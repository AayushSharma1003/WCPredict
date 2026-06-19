"""standings.py — Current WC 2026 group standings using official FIFA letters."""

from collections import defaultdict
from pathlib import Path
import pandas as pd

from tiebreaker import compute_group, rank_third_placed

DATA_DIR    = Path("data")
RESULTS     = DATA_DIR / "results.csv"
ELO_CURRENT = DATA_DIR / "elo_current.csv"
WC_START    = pd.Timestamp("2026-06-01")
KO_START    = pd.Timestamp("2026-06-28")

WC_GROUPS = {
    "A": ["Mexico", "South Africa", "South Korea", "Czech Republic"],
    "B": ["Canada", "Bosnia and Herzegovina", "Qatar", "Switzerland"],
    "C": ["Brazil", "Morocco", "Haiti", "Scotland"],
    "D": ["United States", "Paraguay", "Australia", "Turkey"],
    "E": ["Germany", "Curaçao", "Ivory Coast", "Ecuador"],
    "F": ["Netherlands", "Japan", "Tunisia", "Sweden"],
    "G": ["Belgium", "Egypt", "Iran", "New Zealand"],
    "H": ["Spain", "Cape Verde", "Uruguay", "Saudi Arabia"],
    "I": ["France", "Senegal", "Norway", "Iraq"],
    "J": ["Argentina", "Algeria", "Austria", "Jordan"],
    "K": ["Portugal", "DR Congo", "Uzbekistan", "Colombia"],
    "L": ["England", "Croatia", "Ghana", "Panama"],
}


def load_wc_group_stage() -> pd.DataFrame:
    df = pd.read_csv(RESULTS, parse_dates=["date"])
    return df[(df["tournament"] == "FIFA World Cup")
              & (df["date"] >= WC_START)
              & (df["date"] <  KO_START)].copy()


def infer_clusters(matches: pd.DataFrame) -> set[frozenset]:
    opp = defaultdict(set)
    for _, r in matches.iterrows():
        opp[r.home_team].add(r.away_team)
        opp[r.away_team].add(r.home_team)
    out, seen = set(), set()
    for team in sorted(opp):
        if team in seen:
            continue
        c = sorted({team} | opp[team])
        if len(c) == 4 and all(opp[t] >= set(c) - {t} for t in c):
            out.add(frozenset(c))
            seen.update(c)
    return out


def validate_groups(matches: pd.DataFrame) -> bool:
    elo = set(pd.read_csv(ELO_CURRENT)["team"])
    inferred = infer_clusters(matches)
    ok = True
    for letter, teams in WC_GROUPS.items():
        missing = [t for t in teams if t not in elo]
        if missing:
            print(f"  Group {letter}: not in Elo table → {missing}")
            ok = False
        if frozenset(teams) not in inferred:
            print(f"  Group {letter}: doesn't match fixture-inferred cluster → {teams}")
            ok = False
    return ok


def standings_df(standings):
    df = pd.DataFrame(standings)
    df.insert(0, "Pos", range(1, len(df) + 1))
    df["Status"] = df["Pos"].map(lambda i: "Q" if i <= 2 else "?" if i == 3 else " ")
    return df[["Pos", "Status", "team", "P", "W", "D", "L", "GF", "GA", "GD", "Pts"]]


def thirds_df(thirds):
    df = pd.DataFrame(thirds)
    df.insert(0, "Rank", range(1, len(df) + 1))
    df["Status"] = df["Rank"].map(lambda i: "Q" if i <= 8 else "X")
    return df[["Rank", "Status", "group", "team",
               "P", "W", "D", "L", "GF", "GA", "GD", "Pts"]]


def main():
    matches  = load_wc_group_stage()
    n_played = matches["home_score"].notna().sum()
    print(f"WC 2026 group-stage fixtures: {len(matches)}  (played: {n_played})")

    print("\nValidating hardcoded groups against fixture data...")
    if not validate_groups(matches):
        print("\nValidation failed.")
        return
    print("  All 12 groups match — using official FIFA letters A-L.")

    by_group = {letter: compute_group(matches, teams)
                for letter, teams in WC_GROUPS.items()}

    for letter, st in by_group.items():
        print(f"\nGroup {letter}")
        print(standings_df(st).to_string(index=False))

    thirds = rank_third_placed(by_group)
    print("\n\nThird-placed team ranking (top 8 advance to Round of 32)")
    print(thirds_df(thirds).to_string(index=False))
    print("\nLegend:  Q = advances    ? = third-place lottery    X = eliminated")


if __name__ == "__main__":
    main()