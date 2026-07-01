"""
simulate_groups.py — Phase A of the tournament simulator (fast path).

Pure-Python hot loop, no DataFrame ops inside the simulation.
"""

import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from model      import (load_params, load_elo_table,
                        predict_lambda, scoreline_matrix, validate_params)
from tiebreaker import compute_group_fast
from standings  import WC_GROUPS, load_wc_group_stage

N_SIMS    = 10_000
DATA_DIR  = Path("data")
MAX_GOALS = 10
RNG       = np.random.default_rng(42)


# ---- Setup ----

def split_played_remaining(matches: pd.DataFrame):
    played = matches[matches.home_score.notna()].copy()
    played["home_score"] = played["home_score"].astype(int)
    played["away_score"] = played["away_score"].astype(int)
    remaining = matches[matches.home_score.isna()].copy()
    return played.reset_index(drop=True), remaining.reset_index(drop=True)


def precompute_cdfs(remaining, params, elo):
    cdfs = []
    for r in remaining.itertuples(index=False):
        h_elo, a_elo = elo[r.home_team], elo[r.away_team]
        lam = predict_lambda(params, h_elo, a_elo, 0)
        mu  = predict_lambda(params, a_elo, h_elo, 0)
        M = scoreline_matrix(lam, mu, params["rho"])
        cdfs.append(np.cumsum(M.flatten()))
    return np.array(cdfs)


def sample_all(cdfs, n_sims, rng):
    # Group stage is fully played → nothing left to sample.
    # Return empty (n_sims, 0) arrays; downstream remaining_recs_by_group
    # is empty for every group, so no index into these is ever attempted.
    if cdfs.ndim < 2 or cdfs.shape[0] == 0:
        empty = np.empty((n_sims, 0), dtype=np.int64)
        return empty, empty
    n_matches = cdfs.shape[0]
    u = rng.random((n_sims, n_matches))
    idx = np.empty((n_sims, n_matches), dtype=np.int64)
    for j in range(n_matches):
        idx[:, j] = np.searchsorted(cdfs[j], u[:, j])
    idx = np.clip(idx, 0, cdfs.shape[1] - 1)
    n_cols = MAX_GOALS + 1
    return idx // n_cols, idx % n_cols


def played_records_by_group(played, WC_GROUPS):
    out = {}
    for letter, teams in WC_GROUPS.items():
        ts = set(teams)
        out[letter] = [(r.home_team, r.away_team, r.home_score, r.away_score)
                       for r in played.itertuples(index=False)
                       if r.home_team in ts and r.away_team in ts]
    return out


def remaining_records_by_group(remaining, WC_GROUPS):
    """Per group, list of (home, away, sample_index) — index into the flat arrays."""
    out = {}
    for letter, teams in WC_GROUPS.items():
        ts = set(teams)
        recs = [(r.home_team, r.away_team, i)
                for i, r in enumerate(remaining.itertuples(index=False))
                if r.home_team in ts and r.away_team in ts]
        out[letter] = recs
    return out


# ---- One simulation iteration ----

def simulate_once(played_recs_by_group, remaining_recs_by_group, home_row, away_row):
    final, thirds = {}, []
    for letter, teams in WC_GROUPS.items():
        played_recs = played_recs_by_group[letter]
        sim_recs    = [(h, a, int(home_row[i]), int(away_row[i]))
                       for h, a, i in remaining_recs_by_group[letter]]
        standings = compute_group_fast(teams, played_recs + sim_recs)
        final[letter] = standings
        if len(standings) >= 3:
            t = dict(standings[2]); t["group"] = letter
            thirds.append(t)
    thirds.sort(key=lambda s: (-s["Pts"], -s["GD"], -s["GF"], s["team"]))
    return final, {t["team"] for t in thirds[:8]}


# ---- Main ----

def main():
    print("Loading data...")
    matches = load_wc_group_stage()
    played, remaining = split_played_remaining(matches)
    print(f"  Played: {len(played)}   Remaining: {len(remaining)}")

    params, elo = load_params(), load_elo_table()
    validate_params(params)

    print("Precomputing scoreline matrices...")
    cdfs = precompute_cdfs(remaining, params, elo)

    print(f"Sampling {N_SIMS:,} simulations...")
    t0 = time.time()
    home_goals, away_goals = sample_all(cdfs, N_SIMS, RNG)
    print(f"  Sampling: {time.time() - t0:.1f}s")

    played_recs = played_records_by_group(played, WC_GROUPS)
    rem_recs    = remaining_records_by_group(remaining, WC_GROUPS)

    counts = defaultdict(lambda: {"1st": 0, "2nd": 0, "3rd": 0, "4th": 0, "adv": 0})

    print("Running simulations...")
    t0 = time.time()
    for sim in range(N_SIMS):
        if sim > 0 and sim % 2000 == 0:
            print(f"  {sim:>5} / {N_SIMS}   ({time.time() - t0:.1f}s)")
        final, advancing_thirds = simulate_once(
            played_recs, rem_recs, home_goals[sim], away_goals[sim],
        )
        for letter, standings in final.items():
            for i, s in enumerate(standings):
                pos = ["1st", "2nd", "3rd", "4th"][i]
                counts[s["team"]][pos] += 1
                if i < 2 or (i == 2 and s["team"] in advancing_thirds):
                    counts[s["team"]]["adv"] += 1
    print(f"  Simulation time: {time.time() - t0:.1f}s")

    # Sanity check: total advancement count must equal 32 (top 2 from 12 groups + 8 thirds)
    total_adv = sum(c["adv"] for c in counts.values()) / N_SIMS
    print(f"  Conservation check: total P(Adv) = {total_adv:.4f} (expected 32.0000)")
    assert abs(total_adv - 32.0) < 0.01, f"Sum mismatch — bug in advancement tracking"

    rows = [{"team": team,
             "P(1st)": c["1st"] / N_SIMS,
             "P(2nd)": c["2nd"] / N_SIMS,
             "P(3rd)": c["3rd"] / N_SIMS,
             "P(4th)": c["4th"] / N_SIMS,
             "P(Adv)": c["adv"] / N_SIMS}
            for team, c in counts.items()]
    df = pd.DataFrame(rows)

    print("\nGroup-stage advancement probabilities:")
    for letter, teams in WC_GROUPS.items():
        g = df[df["team"].isin(teams)].sort_values("P(Adv)", ascending=False).copy()
        for c in ["P(1st)", "P(2nd)", "P(3rd)", "P(4th)", "P(Adv)"]:
            g[c] = g[c].apply(lambda x: f"{x:.1%}")
        print(f"\nGroup {letter}")
        print(g.to_string(index=False))

    df = df.sort_values("P(Adv)", ascending=False).reset_index(drop=True)
    df.to_csv(DATA_DIR / "group_simulation.csv", index=False)
    print(f"\nSaved {DATA_DIR / 'group_simulation.csv'}")


if __name__ == "__main__":
    main()