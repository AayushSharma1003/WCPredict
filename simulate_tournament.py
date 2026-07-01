"""
simulate_tournament.py — Full WC 2026 Monte Carlo, group stage through final.

For each simulation:
  1. Sample the remaining group-stage scorelines (reuses simulate_groups).
  2. Compute final group standings using FIFA tiebreakers (reuses tiebreaker).
  3. Rank the 12 third-placed teams and take the best 8.
  4. Use bracket_data.get_third_place_assignment to map advancing thirds to
     R32 slots per FIFA Annex C (the 495-scenario lookup).
  5. Simulate R32 → R16 → QF → SF → F, sampling each knockout match.

Output:
  - data/tournament_simulation.csv      (per-team P(reach round X))
  - Console: top 20 teams sorted by P(Cup), plus conservation checks.

Conservation: across all teams, sum P(R32)=32, P(R16)=16, P(QF)=8, P(SF)=4,
P(F)=2, P(Cup)=1. The asserts at the end verify these to within tolerance.
"""

import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from model import load_params, load_elo_table, validate_params
from bracket_data import BRACKET_PAIRINGS
from knockout import simulate_match, build_r32_bracket, load_played_knockouts, resolve_match

# Reuse group-stage simulation building blocks from simulate_groups.py
from standings import WC_GROUPS, load_wc_group_stage
from simulate_groups import (
    split_played_remaining,
    precompute_cdfs,
    sample_all,
    played_records_by_group,
    remaining_records_by_group,
    simulate_once,
)

# ---- Config ----
N_SIMS    = 10_000
DATA_DIR  = Path("data")
RNG_SEED  = 42

ROUNDS         = ["R32", "R16", "QF", "SF", "F", "Cup"]
ROUND_EXPECTED = {"R32": 32, "R16": 16, "QF": 8, "SF": 4, "F": 2, "Cup": 1}


def main():
    rng = np.random.default_rng(RNG_SEED)

    print("Loading data...")
    matches = load_wc_group_stage()
    played, remaining = split_played_remaining(matches)
    print(f"  Group stage — played: {len(played)}, remaining: {len(remaining)}")

    params = load_params()
    elo    = load_elo_table()
    validate_params(params)

    print("Setting up group-stage sampling...")
    cdfs = precompute_cdfs(remaining, params, elo)
    home_goals, away_goals = sample_all(cdfs, N_SIMS, rng)
    played_recs = played_records_by_group(played, WC_GROUPS)
    rem_recs    = remaining_records_by_group(remaining, WC_GROUPS)

    # Any knockout matches that have already been played — we'll use their
    # real outcomes deterministically instead of resampling.
    played_ko = load_played_knockouts()
    if played_ko:
        print(f"  Knockout stage — {len(played_ko)} match(es) already played, "
              f"treating deterministically")

    counts = defaultdict(lambda: {r: 0 for r in ROUNDS})

    print(f"Running {N_SIMS:,} full-tournament simulations...")
    t0 = time.time()
    for sim in range(N_SIMS):
        if sim > 0 and sim % 1000 == 0:
            elapsed = time.time() - t0
            eta = elapsed / sim * (N_SIMS - sim)
            print(f"  {sim:>5} / {N_SIMS}   elapsed={elapsed:.1f}s  eta={eta:.1f}s")

        # 1. Group stage
        final_groups, advancing_thirds = simulate_once(
            played_recs, rem_recs, home_goals[sim], away_goals[sim],
        )

        # 2. Build R32 bracket using Annex C lookup
        r32_matches = build_r32_bracket(final_groups, advancing_thirds)

        # Every R32 participant gets credit for reaching R32
        for _, a, b in r32_matches:
            counts[a]["R32"] += 1
            counts[b]["R32"] += 1

        # 3. Simulate R32 (or use real result if already played)
        winners = {}
        for m_id, a, b in r32_matches:
            res = resolve_match(a, b, params, elo, rng, played_ko)
            winners[m_id] = res["winner"]

        # 4. Simulate R16, QF, SF, F
        for round_name in ["R16", "QF", "SF", "F"]:
            for m_id, feed_a, feed_b in BRACKET_PAIRINGS[round_name]:
                a = winners[feed_a]
                b = winners[feed_b]
                counts[a][round_name] += 1
                counts[b][round_name] += 1
                res = resolve_match(a, b, params, elo, rng, played_ko)
                winners[m_id] = res["winner"]

        # 5. Cup = winner of the final
        final_id = BRACKET_PAIRINGS["F"][0][0]
        counts[winners[final_id]]["Cup"] += 1

    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s ({elapsed * 1000 / N_SIMS:.2f}ms/sim)")

    # ---- Conservation checks ----
    print("\nConservation checks (across all teams, summed):")
    all_pass = True
    for r in ROUNDS:
        total = sum(c[r] for c in counts.values()) / N_SIMS
        expected = ROUND_EXPECTED[r]
        ok = abs(total - expected) < 0.01
        marker = "✓" if ok else "✗"
        all_pass &= ok
        print(f"  {marker} sum P({r:>3}) = {total:.4f}   (expected {expected})")
    assert all_pass, "Conservation failed — bug somewhere upstream"

    # ---- Build results table ----
    rows = []
    for team, c in counts.items():
        rows.append({
            "team": team,
            **{f"P({r})": c[r] / N_SIMS for r in ROUNDS},
        })
    df = (pd.DataFrame(rows)
          .sort_values("P(Cup)", ascending=False)
          .reset_index(drop=True))

    DATA_DIR.mkdir(exist_ok=True)
    out_csv = DATA_DIR / "tournament_simulation.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nWrote {out_csv}")

    # Display top 20 with percentages
    print("\nTop 20 by P(Cup):")
    display = df.head(20).copy()
    for col in display.columns:
        if col.startswith("P("):
            display[col] = display[col].apply(lambda x: f"{x:6.2%}")
    print(display.to_string(index=False))

    print("\nBottom 10 (least likely to advance from groups):")
    bottom = df.sort_values("P(R32)").head(10).copy()
    for col in bottom.columns:
        if col.startswith("P("):
            bottom[col] = bottom[col].apply(lambda x: f"{x:6.2%}")
    print(bottom.to_string(index=False))


if __name__ == "__main__":
    main()