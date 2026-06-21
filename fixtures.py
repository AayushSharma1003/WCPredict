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
    return CODES.get(team, team[:3].upper())


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
    games = [g for g in hist.get(team, []) if g["date"] < before]
    out = []
    for g in games[-n:]:
        out.append({"res": g["res"], "opp": code(g["opp"]),
                    "score": f"{g['gf']}-{g['ga']}",
                    "date": g["date"].strftime("%Y-%m-%d")})
    return out


def head_to_head(hist: dict, team: str, opp: str, before: pd.Timestamp) -> dict:
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

    print(f"Building fixtures for {len(schedule)} scheduled matches...")
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

    out = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "schedule_source": str(SCHEDULE_CSV),
        "n_matches": len(matches),
        "matches": matches,
    }
    DATA_DIR.mkdir(exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    n_played = sum(m["status"] == "played" for m in matches)
    print(f"Wrote {OUT_JSON}  ({n_played} played, {len(matches) - n_played} upcoming)")


def _s(v):
    return v if isinstance(v, str) and v.strip() else None


if __name__ == "__main__":
    main()