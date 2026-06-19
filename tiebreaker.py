"""
tiebreaker.py — FIFA group standings and tiebreaker logic.

Shared by standings.py and the tournament simulator. Pure functions over
match records: no I/O, no globals.
"""

from __future__ import annotations
import pandas as pd


def empty_stats(team: str) -> dict:
    return {"team": team, "P": 0, "W": 0, "D": 0, "L": 0,
            "GF": 0, "GA": 0, "GD": 0, "Pts": 0}


def compute_group(matches: pd.DataFrame, teams: list[str]) -> list[dict]:
    """Return ordered standings for `teams`, given completed matches."""
    stats  = {t: empty_stats(t) for t in teams}
    played = matches[matches.home_team.isin(teams)
                     & matches.away_team.isin(teams)
                     & matches.home_score.notna()].copy()
    played["home_score"] = played["home_score"].astype(int)
    played["away_score"] = played["away_score"].astype(int)

    for r in played.itertuples(index=False):
        h, a, hs, as_ = r.home_team, r.away_team, r.home_score, r.away_score
        stats[h]["P"]  += 1; stats[a]["P"]  += 1
        stats[h]["GF"] += hs; stats[h]["GA"] += as_
        stats[a]["GF"] += as_; stats[a]["GA"] += hs
        if hs > as_:
            stats[h]["W"] += 1; stats[h]["Pts"] += 3; stats[a]["L"] += 1
        elif hs < as_:
            stats[a]["W"] += 1; stats[a]["Pts"] += 3; stats[h]["L"] += 1
        else:
            stats[h]["D"] += 1; stats[a]["D"] += 1
            stats[h]["Pts"] += 1; stats[a]["Pts"] += 1
    for s in stats.values():
        s["GD"] = s["GF"] - s["GA"]

    ordered = sorted(stats.values(),
                     key=lambda s: (-s["Pts"], -s["GD"], -s["GF"], s["team"]))
    return _apply_h2h(ordered, played)


def _apply_h2h(ordered: list[dict], played: pd.DataFrame) -> list[dict]:
    """Re-sort runs of teams tied on (Pts, GD, GF) using head-to-head."""
    result, i = [], 0
    while i < len(ordered):
        j = i
        while (j < len(ordered)
               and ordered[j]["Pts"] == ordered[i]["Pts"]
               and ordered[j]["GD"]  == ordered[i]["GD"]
               and ordered[j]["GF"]  == ordered[i]["GF"]):
            j += 1
        tied = ordered[i:j]
        if len(tied) > 1:
            tied = _sort_by_h2h(tied, played)
        result.extend(tied)
        i = j
    return result


def _sort_by_h2h(tied: list[dict], played: pd.DataFrame) -> list[dict]:
    teams = {s["team"] for s in tied}
    h2h   = {t: {"Pts": 0, "GD": 0, "GF": 0} for t in teams}
    for r in played.itertuples(index=False):
        if r.home_team not in teams or r.away_team not in teams:
            continue
        h, a, hs, as_ = r.home_team, r.away_team, r.home_score, r.away_score
        h2h[h]["GF"] += hs;  h2h[h]["GD"] += hs - as_
        h2h[a]["GF"] += as_; h2h[a]["GD"] += as_ - hs
        if hs > as_:   h2h[h]["Pts"] += 3
        elif hs < as_: h2h[a]["Pts"] += 3
        else:          h2h[h]["Pts"] += 1; h2h[a]["Pts"] += 1
    return sorted(tied, key=lambda s: (-h2h[s["team"]]["Pts"],
                                        -h2h[s["team"]]["GD"],
                                        -h2h[s["team"]]["GF"],
                                        s["team"]))


def rank_third_placed(by_group: dict[str, list[dict]]) -> list[dict]:
    """Across all groups, rank the third-placed teams (top 8 advance)."""
    thirds = []
    for letter, st in by_group.items():
        if len(st) >= 3:
            t = dict(st[2]); t["group"] = letter
            thirds.append(t)
    thirds.sort(key=lambda s: (-s["Pts"], -s["GD"], -s["GF"], s["team"]))
    return thirds


# ---- Pure-Python "fast" variants for the simulator hot loop ----

def compute_group_fast(teams: list, records: list) -> list:
    """Same as compute_group but operates on a list of (home, away, hs, as_) tuples.
    Skips all DataFrame overhead. ~10x faster for small inputs."""
    stats = {t: {"team": t, "P": 0, "W": 0, "D": 0, "L": 0,
                 "GF": 0, "GA": 0, "GD": 0, "Pts": 0} for t in teams}
    for h, a, hs, as_ in records:
        sh = stats[h]; sa = stats[a]
        sh["P"]  += 1;  sa["P"]  += 1
        sh["GF"] += hs; sh["GA"] += as_
        sa["GF"] += as_;sa["GA"] += hs
        if hs > as_:
            sh["W"] += 1; sh["Pts"] += 3; sa["L"] += 1
        elif hs < as_:
            sa["W"] += 1; sa["Pts"] += 3; sh["L"] += 1
        else:
            sh["D"] += 1; sa["D"] += 1
            sh["Pts"] += 1; sa["Pts"] += 1
    for s in stats.values():
        s["GD"] = s["GF"] - s["GA"]
    ordered = sorted(stats.values(),
                     key=lambda s: (-s["Pts"], -s["GD"], -s["GF"], s["team"]))
    return _apply_h2h_fast(ordered, records)


def _apply_h2h_fast(ordered, records):
    result, i = [], 0
    while i < len(ordered):
        j = i
        while (j < len(ordered)
               and ordered[j]["Pts"] == ordered[i]["Pts"]
               and ordered[j]["GD"]  == ordered[i]["GD"]
               and ordered[j]["GF"]  == ordered[i]["GF"]):
            j += 1
        tied = ordered[i:j]
        if len(tied) > 1:
            tied = _sort_by_h2h_fast(tied, records)
        result.extend(tied)
        i = j
    return result


def _sort_by_h2h_fast(tied, records):
    team_set = {s["team"] for s in tied}
    h2h = {t: {"Pts": 0, "GD": 0, "GF": 0} for t in team_set}
    for h, a, hs, as_ in records:
        if h not in team_set or a not in team_set:
            continue
        h2h[h]["GF"] += hs;  h2h[h]["GD"] += hs - as_
        h2h[a]["GF"] += as_; h2h[a]["GD"] += as_ - hs
        if hs > as_:   h2h[h]["Pts"] += 3
        elif hs < as_: h2h[a]["Pts"] += 3
        else:          h2h[h]["Pts"] += 1; h2h[a]["Pts"] += 1
    return sorted(tied, key=lambda s: (-h2h[s["team"]]["Pts"],
                                        -h2h[s["team"]]["GD"],
                                        -h2h[s["team"]]["GF"],
                                        s["team"]))