"""
fixtures.py — Build the match-by-match fixture feed for the website.

Joins three things the rest of the pipeline already maintains:

  - data/wc_schedule.csv   the official fixture list (date, teams, venue)
  - data/results.csv       what has actually been played (scores)
  - the Dixon-Coles model  (dc_params.json + elo_current.csv)

and writes:

  - data/fixtures.json     one entry per scheduled match, each carrying the
                           model's pre-match prediction, the result if it has
                           been played, plus recent form and head-to-head for
                           both sides.

The website reads fixtures.json directly: the "Today's matches" list filters
by date, and the per-match page renders one entry. No prediction math runs in
the browser.

This is an OFFLINE step — it only reads files the pipeline already produced,
so it cannot make a network call and cannot stall a CI run. If a single match
can't be predicted (e.g. a team missing from the Elo table) that one match is
emitted without a prediction and the run still succeeds.

Knockout stage:
  wc_schedule.csv only contains the 72 group-stage matches. R32 → Final are
  synthesised here from the FIFA-official bracket structure in
  bracket_data.py (R32_SCHEDULE + BRACKET_PAIRINGS + KO_SCHEDULE), resolved
  as far as the played data allows:

    - Once every group has completed all six matches, R32 pairings become
      deterministic via build_r32_bracket. Before that, sides show as
      "Winner Group X" / "Runner-up Group Y" / "Best 3rd (...)".
    - Once an R32 match is played, its winner cascades forward — the R16
      slot fed by it names the actual team; otherwise it stays as
      "Winner Match 73".
    - Same rule applies all the way to the Final.

Usage:
    python fixtures.py
"""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from model import load_params, load_elo_table, predict_match, validate_params
from bracket_data import (
    R32_SCHEDULE,
    BRACKET_PAIRINGS,
    KO_SCHEDULE,
    THIRD_PLACE_SLOTS,
)
from knockout import build_r32_bracket, load_played_knockouts
from standings import WC_GROUPS
from tiebreaker import compute_group, rank_third_placed

# ---- Config ----
DATA_DIR     = Path("data")
SCHEDULE_CSV = DATA_DIR / "wc_schedule.csv"
RESULTS_CSV  = DATA_DIR / "results.csv"
OUT_JSON     = DATA_DIR / "fixtures.json"

FORM_N       = 5     # recent matches per team for the "form" strip
H2H_MAX      = 10    # most recent head-to-head meetings to summarise

# Three-letter codes for the 48 finalists (frontend uses these as the visual id)
CODES = {
    "Argentina":"ARG","Algeria":"ALG","Australia":"AUS","Austria":"AUT","Belgium":"BEL",
    "Bosnia and Herzegovina":"BIH","Brazil":"BRA","Canada":"CAN","Cape Verde":"CPV",
    "Colombia":"COL","Croatia":"CRO","Curaçao":"CUW","Czech Republic":"CZE","DR Congo":"COD",
    "Ecuador":"ECU","Egypt":"EGY","England":"ENG","France":"FRA","Germany":"GER","Ghana":"GHA",
    "Haiti":"HAI","Iran":"IRN","Iraq":"IRQ","Ivory Coast":"CIV","Japan":"JPN","Jordan":"JOR",
    "Mexico":"MEX","Morocco":"MAR","Netherlands":"NED","New Zealand":"NZL","Norway":"NOR",
    "Panama":"PAN","Paraguay":"PAR","Portugal":"POR","Qatar":"QAT","Saudi Arabia":"KSA",
    "Scotland":"SCO","Senegal":"SEN","South Africa":"RSA","South Korea":"KOR","Spain":"ESP",
    "Sweden":"SWE","Switzerland":"SUI","Tunisia":"TUN","Turkey":"TUR","United States":"USA",
    "Uruguay":"URU","Uzbekistan":"UZB",
}

def code(team: str) -> str:
    if team is None:
        return "TBD"
    if team in CODES:
        return CODES[team]
    # Placeholder names ("Winner Group A", "Runner-up Group B", "Winner Match 73",
    # "Best 3rd A/B/C/D/F") are not real teams — the site should render them as
    # TBD rather than mangling them into a 3-letter slug.
    return "TBD"


def is_real_team(team: str) -> bool:
    return team in CODES


# ---- History index (built once, queried per match) ----

def build_history(results: pd.DataFrame) -> dict:
    """team -> list of completed matches (chronological), each from that
    team's own perspective: {date, opp, gf, ga, res}. Used for form + H2H."""
    done = results.dropna(subset=["home_score", "away_score"]).copy()
    done["home_score"] = done["home_score"].astype(int)
    done["away_score"] = done["away_score"].astype(int)
    done = done.sort_values("date")

    hist = defaultdict(list)
    for r in done.itertuples(index=False):
        hist[r.home_team].append({"date": r.date, "opp": r.away_team,
                                  "gf": r.home_score, "ga": r.away_score,
                                  "res": _res(r.home_score, r.away_score)})
        hist[r.away_team].append({"date": r.date, "opp": r.home_team,
                                  "gf": r.away_score, "ga": r.home_score,
                                  "res": _res(r.away_score, r.home_score)})
    return hist


def _res(gf: int, ga: int) -> str:
    return "W" if gf > ga else "L" if gf < ga else "D"


def recent_form(hist: dict, team: str, before: pd.Timestamp, n: int = FORM_N) -> list:
    if not is_real_team(team):
        return []
    games = [g for g in hist.get(team, []) if g["date"] < before]
    out = []
    for g in games[-n:]:
        out.append({"res": g["res"], "opp": code(g["opp"]),
                    "score": f"{g['gf']}-{g['ga']}",
                    "date": g["date"].strftime("%Y-%m-%d")})
    return out


def head_to_head(hist: dict, team: str, opp: str, before: pd.Timestamp) -> dict:
    if not is_real_team(team) or not is_real_team(opp):
        return {"played": 0, "wins": 0, "draws": 0, "losses": 0,
                "recent": [], "last": None}
    games = [g for g in hist.get(team, []) if g["opp"] == opp and g["date"] < before]
    w = sum(g["res"] == "W" for g in games)
    d = sum(g["res"] == "D" for g in games)
    l = sum(g["res"] == "L" for g in games)
    last = None
    if games:
        g = games[-1]
        last = {"date": g["date"].strftime("%Y-%m-%d"),
                "score": f"{g['gf']}-{g['ga']}", "res": g["res"]}
    return {"played": len(games), "wins": w, "draws": d, "losses": l,
            "recent": [{"res": g["res"], "score": f"{g['gf']}-{g['ga']}",
                        "date": g["date"].strftime("%Y-%m-%d")}
                       for g in games[-H2H_MAX:]],
            "last": last}


# ---- Prediction wrapper ----

def predict(home: str, away: str, neutral: bool, params: dict, elo: dict):
    """Return a compact prediction dict, or None if the match can't be modelled."""
    if not (is_real_team(home) and is_real_team(away)):
        return None
    try:
        s = predict_match(home, away, neutral=neutral, params=params, elo=elo)
    except KeyError as e:
        print(f"  ! skipping prediction ({e})")
        return None
    ph, pd_, pa = s["p_home_win"], s["p_draw"], s["p_away_win"]
    fav = home if ph >= pa else away
    return {
        "home_win": round(ph, 4), "draw": round(pd_, 4), "away_win": round(pa, 4),
        "exp_goals_home": round(s["lambda_home"], 2),
        "exp_goals_away": round(s["lambda_away"], 2),
        "favourite": fav, "favourite_code": code(fav),
        "edge": round(abs(ph - pa), 4),
        "top_scorelines": [{"score": f"{h}-{a}", "prob": round(p, 4)}
                           for h, a, p in s["top_scorelines"][:4]],
    }


# ---- Knockout resolution ----

def compute_final_group_standings(results: pd.DataFrame) -> dict:
    """Per group, ordered standings using FIFA tiebreakers. Empty groups
    (no matches played yet) return an empty list."""
    wc = results[(results["tournament"] == "FIFA World Cup")
                 & (results["date"] >= pd.Timestamp("2026-06-11"))
                 & (results["date"] <  pd.Timestamp("2026-06-28"))].copy()
    out = {}
    for letter, teams in WC_GROUPS.items():
        out[letter] = compute_group(wc, teams)
    return out


def _group_stage_complete(final_groups: dict) -> bool:
    """All 12 groups have every team having played all 3 matches."""
    for letter, standings in final_groups.items():
        if len(standings) != 4 or not all(s["P"] == 3 for s in standings):
            return False
    return True


def _r32_pairings(final_groups: dict) -> list:
    """Return list of (match_id, team_a, team_b). Teams are real names if the
    group stage is complete, else FIFA placeholder strings that describe the slot.
    Preserves R32_SCHEDULE match order."""
    if _group_stage_complete(final_groups):
        thirds_ranked = rank_third_placed(final_groups)
        advancing_thirds = {t["team"] for t in thirds_ranked[:8]}
        return build_r32_bracket(final_groups, advancing_thirds)

    # Group stage still in progress → use FIFA placeholder labels
    from bracket_data import SLOT_ALLOWED_SETS
    pos_word = {"1": "Winner Group", "2": "Runner-up Group"}

    def slot_label(slot, other_slot):
        if slot == "3?":
            allowed = SLOT_ALLOWED_SETS[other_slot]
            return "Best 3rd " + "/".join(sorted(allowed))
        pos_char, group_letter = slot[0], slot[1]
        return f"{pos_word[pos_char]} {group_letter}"

    out = []
    for m_id, slot_a, slot_b in R32_SCHEDULE:
        out.append((m_id, slot_label(slot_a, slot_b), slot_label(slot_b, slot_a)))
    return out


def _resolve_ko_teams(final_groups: dict, played_ko: dict):
    """Walk the whole bracket, returning:
      match_teams[m_id] = (team_a, team_b)   (real names or placeholders)
      match_winner[m_id] = winner name       (only for matches actually played)
    """
    match_teams = {}
    match_winner = {}

    # R32
    r32 = _r32_pairings(final_groups)
    for m_id, a, b in r32:
        match_teams[m_id] = (a, b)
        if is_real_team(a) and is_real_team(b):
            actual = played_ko.get(frozenset({a, b}))
            if actual is not None:
                match_winner[m_id] = actual["winner"]

    # R16 → F: each round can be resolved for a given match only if BOTH feeders
    # have been resolved to real teams (i.e. their winners are known).
    for round_name in ["R16", "QF", "SF", "F"]:
        for m_id, feed_a, feed_b in BRACKET_PAIRINGS[round_name]:
            win_a = match_winner.get(feed_a)
            win_b = match_winner.get(feed_b)
            side_a = win_a if win_a is not None else f"Winner Match {feed_a}"
            side_b = win_b if win_b is not None else f"Winner Match {feed_b}"
            match_teams[m_id] = (side_a, side_b)
            if is_real_team(side_a) and is_real_team(side_b):
                actual = played_ko.get(frozenset({side_a, side_b}))
                if actual is not None:
                    match_winner[m_id] = actual["winner"]

    return match_teams, match_winner


def _build_ko_entry(match_id: int, team_a: str, team_b: str,
                    played_ko: dict, hist: dict,
                    params: dict, elo: dict) -> dict:
    """Build one knockout fixture entry in the same schema as group-stage entries."""
    stage, date_str, kickoff_utc, city, country = KO_SCHEDULE[match_id]
    date_ts = pd.Timestamp(date_str)

    sc = None
    if is_real_team(team_a) and is_real_team(team_b):
        actual = played_ko.get(frozenset({team_a, team_b}))
        if actual is not None:
            # played_ko stores scores oriented as (winner=score_a, loser=score_b);
            # flip so score_a matches team_a's goals.
            w = actual["winner"]
            if w == team_a:
                sc = (actual["score_a"], actual["score_b"])
            else:
                sc = (actual["score_b"], actual["score_a"])

    return {
        "match_no":    int(match_id),
        "date":        date_str,
        "kickoff_utc": kickoff_utc,
        "stage":       stage,
        "group":       None,
        "neutral":     True,
        "venue":       {"city": city, "country": country},
        "status":      "played" if sc else "upcoming",
        "score":       {"home": int(sc[0]), "away": int(sc[1])} if sc else None,
        "home": {
            "team": team_a, "code": code(team_a),
            "elo": elo.get(team_a) if is_real_team(team_a) else None,
            "form": recent_form(hist, team_a, date_ts),
        },
        "away": {
            "team": team_b, "code": code(team_b),
            "elo": elo.get(team_b) if is_real_team(team_b) else None,
            "form": recent_form(hist, team_b, date_ts),
        },
        "prediction": predict(team_a, team_b, True, params, elo),
        "h2h":        head_to_head(hist, team_a, team_b, date_ts),
    }


def build_knockout_fixtures(results: pd.DataFrame, hist: dict,
                            params: dict, elo: dict) -> list:
    """Return one fixture entry per knockout match (R32 → Final)."""
    final_groups = compute_final_group_standings(results)
    played_ko    = load_played_knockouts()
    match_teams, _ = _resolve_ko_teams(final_groups, played_ko)

    entries = []
    for m_id in sorted(KO_SCHEDULE.keys()):
        team_a, team_b = match_teams.get(m_id, ("Winner Match ?", "Winner Match ?"))
        entries.append(_build_ko_entry(m_id, team_a, team_b,
                                       played_ko, hist, params, elo))
    return entries


# ---- Main ----

def main():
    if not SCHEDULE_CSV.exists():
        raise SystemExit(f"Missing {SCHEDULE_CSV}. Commit the fixture schedule first.")

    schedule = pd.read_csv(SCHEDULE_CSV, parse_dates=["date"])
    results  = pd.read_csv(RESULTS_CSV, parse_dates=["date"])
    params, elo = load_params(), load_elo_table()
    validate_params(params)
    hist = build_history(results)

    # index played results by (date, home, away) for an O(1) score lookup
    played = results.dropna(subset=["home_score", "away_score"]).copy()
    score_of = {}
    for r in played.itertuples(index=False):
        score_of[(pd.Timestamp(r.date).strftime("%Y-%m-%d"), r.home_team, r.away_team)] = (
            int(r.home_score), int(r.away_score))

    print(f"Building fixtures for {len(schedule)} scheduled group matches + "
          f"{len(KO_SCHEDULE)} knockout matches...")
    matches = []
    for r in schedule.itertuples(index=False):
        d_str   = pd.Timestamp(r.date).strftime("%Y-%m-%d")
        neutral = str(r.neutral).strip().upper() in ("TRUE", "1")
        sc      = score_of.get((d_str, r.home_team, r.away_team))
        if sc is None:                                   # tolerate sides listed in the other order
            swp = score_of.get((d_str, r.away_team, r.home_team))
            if swp is not None:
                sc = (swp[1], swp[0])

        entry = {
            "match_no":   int(r.match_no),
            "date":       d_str,
            "kickoff_utc": (r.kickoff_utc if isinstance(r.kickoff_utc, str) else None) or None,
            "stage":      r.stage,
            "group":      (r.group if isinstance(r.group, str) else None) or None,
            "neutral":    neutral,
            "venue":      {"city": _s(r.city), "country": _s(r.country)},
            "status":     "played" if sc else "upcoming",
            "score":      {"home": sc[0], "away": sc[1]} if sc else None,
            "home": {
                "team": r.home_team, "code": code(r.home_team),
                "elo": elo.get(r.home_team),
                "form": recent_form(hist, r.home_team, r.date),
            },
            "away": {
                "team": r.away_team, "code": code(r.away_team),
                "elo": elo.get(r.away_team),
                "form": recent_form(hist, r.away_team, r.date),
            },
            "prediction": predict(r.home_team, r.away_team, neutral, params, elo),
            "h2h": head_to_head(hist, r.home_team, r.away_team, r.date),
        }
        matches.append(entry)

    # Append knockout fixtures (real teams once resolvable, placeholders before)
    ko_entries = build_knockout_fixtures(results, hist, params, elo)
    matches.extend(ko_entries)

    out = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "schedule_source": str(SCHEDULE_CSV),
        "n_matches": len(matches),
        "matches": matches,
    }
    DATA_DIR.mkdir(exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    n_played      = sum(m["status"] == "played" for m in matches)
    n_group       = sum(m["stage"] == "group" for m in matches)
    n_ko          = len(matches) - n_group
    n_ko_resolved = sum(1 for m in matches
                        if m["stage"] != "group"
                        and is_real_team(m["home"]["team"])
                        and is_real_team(m["away"]["team"]))
    print(f"Wrote {OUT_JSON}  ({n_played} played, {len(matches) - n_played} upcoming)")
    print(f"  Group stage: {n_group} matches")
    print(f"  Knockout:    {n_ko} matches ({n_ko_resolved} with resolved teams)")


def _s(v):
    return v if isinstance(v, str) and v.strip() else None


if __name__ == "__main__":
    main()