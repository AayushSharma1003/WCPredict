"""
diagnose_tournament.py — Tournament Monte Carlo with diagnostic outputs.

Beyond the per-team round probabilities (which simulate_tournament.py already
produces), this script tracks bracket-level data for visualization:

  - Per match, per side: top-N teams most likely to appear there
  - Per match: top-N most likely actual matchups (joint, not marginal)
  - Top-N most likely final pairings
  - Per top champion: most likely opponents in each round

Outputs three JSON files in data/, designed for direct consumption by the
website's bracket renderer:

  - bracket_view.json    — every knockout match, both sides + joint matchups
  - final_pairings.json  — top final pairings, sorted by probability
  - champion_routes.json — per top champion, modal opponents per round

Also prints a human-readable summary to the terminal.
"""

import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from model import load_params, load_elo_table, validate_params
from bracket_data import BRACKET_PAIRINGS, R32_SCHEDULE
from knockout import simulate_match, build_r32_bracket, load_played_knockouts, resolve_match
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
N_SIMS           = 10_000
DATA_DIR         = Path("data")
RNG_SEED         = 42

TOP_K_PER_SIDE    = 5    # top-N teams per match side in bracket_view.json
TOP_K_JOINT       = 5    # top-N joint matchups per match
TOP_K_FINALS      = 20   # top-N final pairings
TOP_K_CHAMPIONS   = 10   # detailed routes for top-N most likely champions
TOP_K_OPPONENTS   = 5    # top-N opponents per round in champion routes

# Build match-id ↔ round mapping
ROUND_TO_MATCHES = {
    "R32": [m[0] for m in R32_SCHEDULE],
    "R16": [m[0] for m in BRACKET_PAIRINGS["R16"]],
    "QF":  [m[0] for m in BRACKET_PAIRINGS["QF"]],
    "SF":  [m[0] for m in BRACKET_PAIRINGS["SF"]],
    "F":   [m[0] for m in BRACKET_PAIRINGS["F"]],
}


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

    # Deterministic results for KO matches already played (see knockout.py)
    played_ko = load_played_knockouts()
    if played_ko:
        print(f"  Knockout stage — {len(played_ko)} match(es) already played, "
              f"treating deterministically")

    # ---- Diagnostic counters ----
    # (match_id, side) → team → count
    side_counts   = defaultdict(lambda: defaultdict(int))
    # match_id → (team_a, team_b) → count (joint matchup, ordered as scheduled)
    joint_counts  = defaultdict(lambda: defaultdict(int))
    # (team_a, team_b) → count (sorted), top-K final pairings
    final_pairs   = defaultdict(int)
    # team → count, used to identify top champions
    cup_counts    = defaultdict(int)
    # champion → round → opponent → count
    champ_opps    = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

    final_match_id = BRACKET_PAIRINGS["F"][0][0]
    sf1_id         = BRACKET_PAIRINGS["F"][0][1]
    sf2_id         = BRACKET_PAIRINGS["F"][0][2]

    print(f"Running {N_SIMS:,} simulations with diagnostics...")
    t0 = time.time()
    for sim in range(N_SIMS):
        if sim > 0 and sim % 1000 == 0:
            elapsed = time.time() - t0
            eta = elapsed / sim * (N_SIMS - sim)
            print(f"  {sim:>5}/{N_SIMS}   elapsed={elapsed:.1f}s eta={eta:.1f}s")

        final_groups, advancing_thirds = simulate_once(
            played_recs, rem_recs, home_goals[sim], away_goals[sim]
        )
        r32_matches = build_r32_bracket(final_groups, advancing_thirds)

        # Per-team route within this sim: team → [(round, opponent), ...]
        team_route = defaultdict(list)
        winners = {}

        # R32
        for m_id, a, b in r32_matches:
            side_counts[(m_id, 0)][a] += 1
            side_counts[(m_id, 1)][b] += 1
            joint_counts[m_id][(a, b)] += 1
            team_route[a].append(("R32", b))
            team_route[b].append(("R32", a))
            res = resolve_match(a, b, params, elo, rng, played_ko)
            winners[m_id] = res["winner"]

        # R16 → F
        for round_name in ["R16", "QF", "SF", "F"]:
            for m_id, fa, fb in BRACKET_PAIRINGS[round_name]:
                a, b = winners[fa], winners[fb]
                side_counts[(m_id, 0)][a] += 1
                side_counts[(m_id, 1)][b] += 1
                joint_counts[m_id][(a, b)] += 1
                team_route[a].append((round_name, b))
                team_route[b].append((round_name, a))
                res = resolve_match(a, b, params, elo, rng, played_ko)
                winners[m_id] = res["winner"]

        # Final pairing (alphabetically ordered to deduplicate (A,B) and (B,A))
        finalist_a = winners[sf1_id]
        finalist_b = winners[sf2_id]
        pair = tuple(sorted([finalist_a, finalist_b]))
        final_pairs[pair] += 1

        # Champion
        champion = winners[final_match_id]
        cup_counts[champion] += 1
        for round_name, opp in team_route[champion]:
            champ_opps[champion][round_name][opp] += 1

    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s ({elapsed * 1000 / N_SIMS:.2f}ms/sim)")

    DATA_DIR.mkdir(exist_ok=True)

    # ---- Output 1: bracket_view.json ----
    print("\nWriting bracket_view.json...")
    bracket_view = {"n_sims": N_SIMS, "rounds": []}
    for round_name in ["R32", "R16", "QF", "SF", "F"]:
        round_obj = {"name": round_name, "matches": []}
        for m_id in ROUND_TO_MATCHES[round_name]:
            sides = []
            for side in [0, 1]:
                tc = side_counts[(m_id, side)]
                ranked = sorted(tc.items(), key=lambda x: (-x[1], x[0]))
                top    = ranked[:TOP_K_PER_SIDE]
                modal_team, modal_count = (top[0] if top else ("?", 0))
                sides.append({
                    "modal_team": modal_team,
                    "modal_prob": modal_count / N_SIMS,
                    "top_teams": [
                        {"team": t, "prob": c / N_SIMS} for t, c in top
                    ],
                })

            joint = joint_counts[m_id]
            ranked_joint = sorted(joint.items(), key=lambda x: (-x[1], x[0]))
            top_joint = ranked_joint[:TOP_K_JOINT]

            round_obj["matches"].append({
                "match_id": m_id,
                "sides": sides,
                "top_matchups": [
                    {"team_a": a, "team_b": b, "prob": c / N_SIMS}
                    for (a, b), c in top_joint
                ],
            })
        bracket_view["rounds"].append(round_obj)

    _write_json(DATA_DIR / "bracket_view.json", bracket_view)

    # ---- Output 2: final_pairings.json ----
    print("Writing final_pairings.json...")
    ranked_pairs = sorted(final_pairs.items(), key=lambda x: (-x[1], x[0]))
    pairings_out = {
        "n_sims": N_SIMS,
        "n_unique_pairings": len(final_pairs),
        "pairings": [
            {"team_a": a, "team_b": b, "prob": c / N_SIMS, "count": c}
            for (a, b), c in ranked_pairs[:TOP_K_FINALS]
        ],
    }
    _write_json(DATA_DIR / "final_pairings.json", pairings_out)

    # ---- Output 3: champion_routes.json ----
    print("Writing champion_routes.json...")
    top_champs = sorted(cup_counts.items(), key=lambda x: (-x[1], x[0]))[:TOP_K_CHAMPIONS]
    champions_out = {"n_sims": N_SIMS, "champions": []}
    for team, count in top_champs:
        modal_route = []
        for r in ["R32", "R16", "QF", "SF", "F"]:
            opps = champ_opps[team][r]
            if not opps:
                continue
            ranked_opps = sorted(opps.items(), key=lambda x: (-x[1], x[0]))
            top_opp, top_opp_count = ranked_opps[0]
            modal_route.append({
                "round": r,
                "modal_opponent": top_opp,
                "modal_prob_given_champ": top_opp_count / count,
                "top_opponents": [
                    {"opponent": o, "prob_given_champ": c / count}
                    for o, c in ranked_opps[:TOP_K_OPPONENTS]
                ],
            })
        champions_out["champions"].append({
            "team": team,
            "cup_prob": count / N_SIMS,
            "cup_count": count,
            "modal_route": modal_route,
        })
    _write_json(DATA_DIR / "champion_routes.json", champions_out)

    # ---- Terminal summary ----
    print_summary(bracket_view, pairings_out, champions_out)


def _write_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    print(f"  Wrote {path}")


def print_summary(bracket_view, pairings_out, champions_out):
    n = bracket_view["n_sims"]
    print(f"\n{'=' * 70}")
    print(f"=== DIAGNOSTIC SUMMARY  ({n:,} simulations) ===")
    print(f"{'=' * 70}")

    print("\n--- Modal bracket (most likely team on each side per match) ---")
    for round_obj in bracket_view["rounds"]:
        print(f"\n  [{round_obj['name']}]")
        for m in round_obj["matches"]:
            sa, sb = m["sides"]
            print(f"    M{m['match_id']:>3}  "
                  f"{sa['modal_team']:>22} ({sa['modal_prob']:5.1%}) "
                  f"vs {sb['modal_team']:<22} ({sb['modal_prob']:5.1%})")

    print("\n--- Top 10 most likely final pairings ---")
    for p in pairings_out["pairings"][:10]:
        print(f"  {p['team_a']:>20} vs {p['team_b']:<20}  "
              f"{p['prob']:5.2%}  ({p['count']} sims)")
    print(f"  ... ({pairings_out['n_unique_pairings']:,} unique pairings total)")

    print("\n--- Most likely route per top champion (conditional on winning) ---")
    for c in champions_out["champions"][:5]:
        print(f"\n  ★ {c['team']} — P(Cup) = {c['cup_prob']:.2%} "
              f"(won {c['cup_count']:,} sims)")
        for stage in c["modal_route"]:
            mp = stage["modal_prob_given_champ"]
            print(f"      {stage['round']:>4}:  most likely vs "
                  f"{stage['modal_opponent']:<22}  ({mp:5.1%} of {c['team']}'s wins)")


if __name__ == "__main__":
    main()