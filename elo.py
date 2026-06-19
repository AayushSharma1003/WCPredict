"""
elo.py — Compute World Football Elo ratings from international match history.

Reads  data/results.csv  (martj42/international_results format).
Writes data/elo_current.csv (current ratings, sorted)
       data/elo_history.csv (pre/post Elo for every match — useful later)
"""

import pandas as pd
import numpy as np
from collections import defaultdict
from pathlib import Path

# ---- Config ----
DATA_DIR     = Path("data")
RESULTS_CSV  = DATA_DIR / "results.csv"
OUT_CURRENT  = DATA_DIR / "elo_current.csv"
OUT_HISTORY  = DATA_DIR / "elo_history.csv"

INITIAL_RATING = 1500
HOME_ADVANTAGE = 100  # Elo bump for home team on non-neutral pitches


def k_factor(tournament: str) -> int:
    """Tournament importance → K. Follows World Football Elo convention."""
    t = tournament.lower()

    # Order matters: more specific checks first.
    if "qualification" in t or "qualifier" in t:
        return 30
    if "friendly" in t or "fifa series" in t or "concacaf series" in t:
        return 20
    if "fifa world cup" in t:
        return 60
    if "nations league" in t:                       # before continental block
        return 40
    if any(c in t for c in [
        "uefa euro", "copa américa", "copa america",
        "african cup of nations", "africa cup of nations",
        "afc asian cup", "gold cup", "confederations cup",
    ]):
        return 50
    return 30


def gd_multiplier(gd: int) -> float:
    """Goal-difference multiplier: bigger wins move ratings more, with diminishing returns."""
    gd = abs(gd)
    if gd <= 1:
        return 1.0
    if gd == 2:
        return 1.5
    return (11 + gd) / 8.0                         # gd=3 → 1.75, gd=4 → 1.875, gd=5 → 2.0, ...


def expected_score(r_a: float, r_b: float) -> float:
    """Probability A beats B under Elo (treats draws as half-wins for both)."""
    return 1.0 / (1.0 + 10 ** ((r_b - r_a) / 400))


def main():
    df = pd.read_csv(RESULTS_CSV, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df["neutral"] = df["neutral"].astype(str).str.upper() == "TRUE"

    print(f"Loaded {len(df):,} matches from "
          f"{df['date'].min().date()} to {df['date'].max().date()}")

    ratings = defaultdict(lambda: INITIAL_RATING)
    history_rows = []

    for row in df.itertuples(index=False):
        if pd.isna(row.home_score) or pd.isna(row.away_score):
            continue

        home, away = row.home_team, row.away_team
        hs, as_   = int(row.home_score), int(row.away_score)

        r_home, r_away = ratings[home], ratings[away]
        ha = 0 if row.neutral else HOME_ADVANTAGE
        e_home = expected_score(r_home + ha, r_away)
        e_away = 1 - e_home

        if   hs > as_: s_home, s_away = 1.0, 0.0
        elif hs < as_: s_home, s_away = 0.0, 1.0
        else:          s_home, s_away = 0.5, 0.5

        K = k_factor(row.tournament)
        G = gd_multiplier(hs - as_)

        new_r_home = r_home + K * G * (s_home - e_home)
        new_r_away = r_away + K * G * (s_away - e_away)
        ratings[home] = new_r_home
        ratings[away] = new_r_away

        history_rows.append({
            "date": row.date, "home_team": home, "away_team": away,
            "home_score": hs, "away_score": as_,
            "tournament": row.tournament, "neutral": row.neutral,
            "home_elo_pre":  round(r_home, 1),
            "away_elo_pre":  round(r_away, 1),
            "home_elo_post": round(new_r_home, 1),
            "away_elo_post": round(new_r_away, 1),
            "expected_home": round(e_home, 3),
        })

    current = (pd.DataFrame(
                  [(t, round(r, 1)) for t, r in ratings.items()],
                  columns=["team", "elo"])
               .sort_values("elo", ascending=False)
               .reset_index(drop=True))
    current.to_csv(OUT_CURRENT, index=False)
    pd.DataFrame(history_rows).to_csv(OUT_HISTORY, index=False)

    print(f"\nWrote {OUT_CURRENT} ({len(current)} teams)")
    print(f"Wrote {OUT_HISTORY} ({len(history_rows):,} match records)")
    print("\nTop 30 teams by current Elo:")
    print(current.head(30).to_string(index=False))


if __name__ == "__main__":
    main()