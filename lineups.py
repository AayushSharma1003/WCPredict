"""
lineups.py — Best-effort confirmed-lineup fetcher (TheSportsDB or API-Football).

Confirmed starting XIs are only published ~1 hour before kickoff. There is no
free, reliable source of *predicted* XIs for national teams, so this step is
best-effort: it stores a confirmed XI when one exists, otherwise marks the
match "not_announced" and the page falls back to the model + form + H2H that
fixtures.py always provides.

Run this on a TIGHT cadence (every ~20 min) from its own lightweight workflow,
not on the 6-hourly modelling cadence. It uses only the standard library, so
that workflow needs no pip install.

PROVIDER SELECTION (automatic)
------------------------------
  - If LINEUPS_API_KEY is set to a real key  -> API-Football (best WC coverage)
  - Otherwise                                -> TheSportsDB free key "3"
  Force either with LINEUPS_PROVIDER = "apifootball" | "thesportsdb".

ENV
---
  LINEUPS_API_KEY     API key (GitHub Actions secret). Default "3" (TheSportsDB).
  LINEUPS_PROVIDER    optional override of the auto-detected provider.
  WC_LEAGUE_ID        API-Football league id for the World Cup (default 1).
  WC_SEASON           API-Football season (default 2026).

PIPELINE SAFETY: every network path is guarded; a missing key, timeout, HTTP
error, or odd payload degrades to "not_announced"; confirmed XIs already on
disk are preserved; the script always exits 0.

Usage:
    python lineups.py
    python lineups.py --days 1
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone, date, timedelta
from pathlib import Path
from urllib.request import urlopen, Request
from urllib.parse import urlencode
from urllib.error import URLError, HTTPError

DATA_DIR      = Path("data")
FIXTURES_JSON = DATA_DIR / "fixtures.json"
OUT_JSON      = DATA_DIR / "lineups.json"

API_KEY  = os.environ.get("LINEUPS_API_KEY", "3")
PROVIDER = os.environ.get("LINEUPS_PROVIDER") or (
    "apifootball" if API_KEY not in ("", "3") else "thesportsdb")
TIMEOUT  = 20
NET_ERRORS = (HTTPError, URLError, ValueError, TimeoutError, OSError)

# API-Football config
AF_BASE   = "https://v3.football.api-sports.io"
AF_LEAGUE = os.environ.get("WC_LEAGUE_ID", "1")     # FIFA World Cup
AF_SEASON = os.environ.get("WC_SEASON", "2026")

# TheSportsDB config
TSD_BASE  = os.environ.get("LINEUPS_API_BASE", "https://www.thesportsdb.com/api/v1/json")

# martj42 spellings -> provider spellings. Extend as you hit misses.
NAME_MAP = {
    "United States": "USA", "South Korea": "South Korea", "Ivory Coast": "Ivory Coast",
    "DR Congo": "DR Congo", "Bosnia and Herzegovina": "Bosnia and Herzegovina",
    "Czech Republic": "Czechia", "Cape Verde": "Cape Verde",
}

def api_name(team: str) -> str:
    return NAME_MAP.get(team, team)

def match_key(m: dict) -> str:
    return f"{m['date']}|{m['home']['team']}|{m['away']['team']}"

def _matches(team_a: str, team_b: str) -> bool:
    """Loose name match: first significant token of one appears in the other."""
    a, b = (team_a or "").lower(), (team_b or "").lower()
    if not a or not b:
        return False
    return a in b or b in a or a.split()[0] == b.split()[0]


# ---- HTTP ----

def _get_json(url: str, headers: dict | None = None):
    req = Request(url, headers={"User-Agent": "WCPredict/1.0", **(headers or {})})
    with urlopen(req, timeout=TIMEOUT) as r:
        return json.loads(r.read().decode("utf-8", "replace"))


# ---- Provider: API-Football ----

def _af_day_events(date_str: str) -> list:
    url = f"{AF_BASE}/fixtures?" + urlencode(
        {"date": date_str, "league": AF_LEAGUE, "season": AF_SEASON})
    try:
        data = _get_json(url, headers={"x-apisports-key": API_KEY})
    except NET_ERRORS:
        return []
    out = []
    for f in (data or {}).get("response", []) or []:
        out.append({"id": f.get("fixture", {}).get("id"),
                    "home": f.get("teams", {}).get("home", {}).get("name"),
                    "away": f.get("teams", {}).get("away", {}).get("name"),
                    "kickoff": f.get("fixture", {}).get("date")})
    return out

def _af_lineup(ev: dict, m: dict) -> dict | None:
    if not ev or not ev.get("id"):
        return None
    url = f"{AF_BASE}/fixtures/lineups?" + urlencode({"fixture": ev["id"]})
    try:
        data = _get_json(url, headers={"x-apisports-key": API_KEY})
    except NET_ERRORS:
        return None
    blocks = (data or {}).get("response", []) or []
    if not blocks:
        return None

    def side_from(block) -> dict:
        xi = [{"name": p.get("player", {}).get("name"),
               "number": p.get("player", {}).get("number"),
               "pos": p.get("player", {}).get("pos")}
              for p in (block.get("startXI") or [])]
        return {"formation": block.get("formation"),
                "xi": xi,
                "coach": (block.get("coach") or {}).get("name")}

    home = away = None
    for block in blocks:
        name = (block.get("team") or {}).get("name")
        if _matches(name, api_name(m["home"]["team"])):
            home = side_from(block)
        elif _matches(name, api_name(m["away"]["team"])):
            away = side_from(block)
    if not home and not away:
        return None
    return {"home": home or _empty(), "away": away or _empty(),
            "kickoff_utc": ev.get("kickoff")}


# ---- Provider: TheSportsDB ----

def _tsd_day_events(date_str: str) -> list:
    try:
        day = _get_json(f"{TSD_BASE}/{API_KEY}/eventsday.php?d={date_str}&s=Soccer")
    except NET_ERRORS:
        return []
    out = []
    for e in (day or {}).get("events") or []:
        out.append({"id": e.get("idEvent"),
                    "home": e.get("strHomeTeam"), "away": e.get("strAwayTeam"),
                    "homeFormation": e.get("strHomeFormation"),
                    "awayFormation": e.get("strAwayFormation"),
                    "kickoff": (e.get("strTimestamp") or None)})
    return out

def _tsd_lineup(ev: dict, m: dict) -> dict | None:
    if not ev or not ev.get("id"):
        return None
    try:
        lu = _get_json(f"{TSD_BASE}/{API_KEY}/lookuplineup.php?id={ev['id']}")
    except NET_ERRORS:
        return None
    players = (lu or {}).get("lineup") or []
    if not players:
        return None

    def side(is_home: bool, formation) -> dict:
        xi = [{"name": p.get("strPlayer"),
               "number": p.get("intSquadNumber"),
               "pos": p.get("strPosition")}
              for p in players
              if (p.get("strHome") == "Yes") == is_home
              and (p.get("strSubstitute") in (None, "No", ""))]
        return {"formation": formation, "xi": xi, "coach": None}

    home = side(True,  ev.get("homeFormation"))
    away = side(False, ev.get("awayFormation"))
    if not home["xi"] and not away["xi"]:
        return None
    return {"home": home, "away": away, "kickoff_utc": ev.get("kickoff")}


def _empty() -> dict:
    return {"formation": None, "xi": [], "coach": None}


# ---- provider dispatch ----

def day_events(date_str: str) -> list:
    return _af_day_events(date_str) if PROVIDER == "apifootball" else _tsd_day_events(date_str)

def find_event(m: dict, events: list) -> dict | None:
    for e in events:
        if _matches(e.get("home"), api_name(m["home"]["team"])) and \
           _matches(e.get("away"), api_name(m["away"]["team"])):
            return e
    return None

def lineup_for(m: dict, ev: dict | None) -> dict | None:
    if ev is None:
        return None
    return _af_lineup(ev, m) if PROVIDER == "apifootball" else _tsd_lineup(ev, m)


# ---- main ----

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=1,
                    help="Only fetch lineups for matches within N days (default 1).")
    args = ap.parse_args()

    print(f"Provider: {PROVIDER}")

    if not FIXTURES_JSON.exists():
        print(f"  ! {FIXTURES_JSON} not found — run fixtures.py first. Skipping.")
        _write({})
        return 0

    fixtures = json.loads(FIXTURES_JSON.read_text())
    existing, existing_ok = {}, False
    if OUT_JSON.exists():
        try:
            existing = json.loads(OUT_JSON.read_text()).get("lineups", {})
            existing_ok = True
        except json.JSONDecodeError:
            existing = {}

    today   = date.today()
    horizon = today + timedelta(days=args.days)

    day_cache: dict[str, list] = {}
    out, n_found, n_checked = {}, 0, 0

    for m in fixtures.get("matches", []):
        key = match_key(m)
        if existing.get(key, {}).get("status") == "confirmed":
            out[key] = existing[key]                      # never lose a confirmed XI
            continue
        if m["status"] == "played":
            continue
        try:
            mdate = datetime.strptime(m["date"], "%Y-%m-%d").date()
        except ValueError:
            continue
        if not (today <= mdate <= horizon):
            continue

        n_checked += 1
        result = None
        try:
            if m["date"] not in day_cache:
                day_cache[m["date"]] = day_events(m["date"])
            result = lineup_for(m, find_event(m, day_cache[m["date"]]))
        except Exception as e:
            print(f"  ! {key}: {type(e).__name__}: {e}")

        if result:
            n_found += 1
            out[key] = {"status": "confirmed",
                        "fetched_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                        **result}
        else:
            out[key] = {"status": "not_announced"}

    if out == existing and existing_ok:
        print(f"Lineups: checked {n_checked} match(es), {n_found} confirmed — no change, file untouched.")
        return 0
    _write(out)
    print(f"Lineups: checked {n_checked} upcoming match(es), {n_found} confirmed XI(s).")
    return 0


def _write(lineups: dict):
    DATA_DIR.mkdir(exist_ok=True)
    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "provider": PROVIDER,
        "lineups": lineups,
    }
    with open(OUT_JSON, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"  ! lineups.py degraded ({type(e).__name__}: {e}); writing empty feed.")
        try:
            _write({})
        except Exception:
            pass
        sys.exit(0)