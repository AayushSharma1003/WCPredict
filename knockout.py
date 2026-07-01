"""
knockout.py — Knockout match simulation and bracket builder.

A knockout match goes:
  1. Regulation: sample scoreline from the DC matrix (with tau correction).
  2. If drawn: extra time, additional Poisson goals at 1/3 the regulation rate
     (since ET = 30 min and regulation = 90 min).
  3. If still drawn: penalty shootout — essentially 50/50 with a small Elo
     nudge (penalties are mostly random in the literature).

The bracket builder uses bracket_data.get_third_place_assignment to turn the
final group standings + the 8 advancing third-placed teams into the 16 R32
matchups, per FIFA's official 495-scenario Annex C.

resolve_match(...) is the wrapper the simulators call — it uses an actually
played result if available, else falls back to Monte Carlo. This is what lets
the pipeline keep producing sensible predictions after knockouts have started
playing (Argentina beating Jordan in R32 should be a certainty, not a coin
flip, in the R16 sims that follow).
"""

from pathlib import Path

import numpy as np
import pandas as pd

from model import predict_lambda, scoreline_matrix, MAX_GOALS
from bracket_data import R32_SCHEDULE, get_third_place_assignment

# Knockouts start June 28, 2026 (R32). Anything on/after this date in the
# WC tournament is a knockout match.
KO_START = pd.Timestamp("2026-06-28")

# Extra time is 30 of 90 minutes.
ET_FRACTION = 30.0 / 90.0

# Penalty shootout: P(team_a wins) = 0.5 + scale * tanh(elo_diff / divisor).
# Max deviation from 0.5 = ±scale. The literature gives Elo very weak signal
# for shootouts, so the scale stays small.
PEN_TANH_SCALE  = 0.03
PEN_ELO_DIVISOR = 200.0


# ---- Knockout match simulation ----

def simulate_match(team_a: str, team_b: str, params: dict, elo: dict,
                   rng: np.random.Generator, *, neutral: bool = True) -> dict:
    """Simulate one knockout match.

    Returns a dict with keys:
        winner, loser     — team names
        score_a, score_b  — final score (including any ET goals)
        went_to_et        — bool, True if regulation ended drawn
        went_to_pens      — bool, True if ET also ended drawn
    """
    elo_a, elo_b = elo[team_a], elo[team_b]
    is_home = 0 if neutral else 1
    lam_a = predict_lambda(params, elo_a, elo_b, is_home)
    lam_b = predict_lambda(params, elo_b, elo_a, 0)

    # Regulation: sample from the joint scoreline matrix (with tau correction)
    M = scoreline_matrix(lam_a, lam_b, params["rho"])
    flat = M.flatten()
    cdf  = np.cumsum(flat)
    idx  = int(np.searchsorted(cdf, rng.random()))
    idx  = min(idx, len(flat) - 1)
    score_a = idx // (MAX_GOALS + 1)
    score_b = idx %  (MAX_GOALS + 1)

    went_to_et = went_to_pens = False

    if score_a == score_b:
        went_to_et = True
        # Independent Poisson goals during extra time
        score_a += int(rng.poisson(lam_a * ET_FRACTION))
        score_b += int(rng.poisson(lam_b * ET_FRACTION))

    if score_a == score_b:
        went_to_pens = True
        p_a = 0.5 + PEN_TANH_SCALE * np.tanh((elo_a - elo_b) / PEN_ELO_DIVISOR)
        winner, loser = (team_a, team_b) if rng.random() < p_a else (team_b, team_a)
    elif score_a > score_b:
        winner, loser = team_a, team_b
    else:
        winner, loser = team_b, team_a

    return {"winner": winner, "loser": loser,
            "score_a": score_a, "score_b": score_b,
            "went_to_et": went_to_et, "went_to_pens": went_to_pens}


# ---- Bracket builder ----

def build_r32_bracket(final_groups: dict, advancing_thirds_set: set) -> list:
    """From the simulated final group standings + the set of advancing third-
    placed team names, assemble the 16 R32 matchups.

    Args:
        final_groups: dict {group_letter: ordered standings list}, where
            each standings entry is a dict with at least key "team".
        advancing_thirds_set: set of team names whose thirds advance.

    Returns:
        list of (match_id, team_a, team_b) tuples in canonical R32 order.
    """
    # Identify the 8 groups that produced advancing thirds + the team in each
    advancing_groups        = []
    third_team_for_group    = {}
    for letter, standings in final_groups.items():
        if len(standings) < 3:
            continue
        third = standings[2]["team"]
        if third in advancing_thirds_set:
            advancing_groups.append(letter)
            third_team_for_group[letter] = third

    if len(advancing_groups) != 8:
        raise RuntimeError(
            f"Expected 8 advancing thirds, got {len(advancing_groups)}: "
            f"{advancing_groups}")

    # Look up the FIFA Annex C slot assignment
    slot_to_source = get_third_place_assignment(advancing_groups)

    def resolve(slot, other_slot):
        if slot == "3?":
            # The third-place source group is determined by the other (1X) slot
            return third_team_for_group[slot_to_source[other_slot]]
        pos_char, group_letter = slot[0], slot[1]
        return final_groups[group_letter][int(pos_char) - 1]["team"]

    matches = []
    for match_id, slot_a, slot_b in R32_SCHEDULE:
        team_a = resolve(slot_a, slot_b)
        team_b = resolve(slot_b, slot_a)
        matches.append((match_id, team_a, team_b))
    return matches


# ---- Actually-played knockout results ----

def load_played_knockouts(results_path="data/results.csv",
                          shootouts_path="data/shootouts.csv") -> dict:
    """Load all played WC knockout matches, keyed by unordered team pair.

    Two files feed into this:
      - results.csv:  regulation/ET final score for every played match.
                      A drawn KO here means it went to penalties.
      - shootouts.csv (optional, best-effort): the sibling martj42 file naming
                      the shootout winner. If it's missing or doesn't cover a
                      particular drawn KO, that match is skipped and the
                      simulators fall back to Monte Carlo for it.

    The frozenset key exploits the fact that any given pair of teams meets
    at most once in the knockout stage, so orientation doesn't matter.

    Returns:
        dict[frozenset[str, str], dict] with fields:
            winner, loser, score_a, score_b, went_to_et, went_to_pens
        oriented as (team_a=winner, team_b=loser).
    """
    results_path   = Path(results_path)
    shootouts_path = Path(shootouts_path)

    if not results_path.exists():
        return {}

    df = pd.read_csv(results_path, parse_dates=["date"])
    ko = df[(df["tournament"] == "FIFA World Cup")
            & (df["date"] >= KO_START)
            & df["home_score"].notna()].copy()
    if ko.empty:
        return {}

    # Best-effort shootout lookup: (date_str, frozenset({home, away})) → winner
    shootout_winner = {}
    if shootouts_path.exists():
        try:
            sh = pd.read_csv(shootouts_path, parse_dates=["date"])
            for r in sh.itertuples(index=False):
                key = (pd.Timestamp(r.date).strftime("%Y-%m-%d"),
                       frozenset({r.home_team, r.away_team}))
                shootout_winner[key] = r.winner
        except Exception as e:
            print(f"  ! shootouts.csv unreadable ({e}), continuing without it")

    out = {}
    for r in ko.itertuples(index=False):
        h, a = r.home_team, r.away_team
        sa, sb = int(r.home_score), int(r.away_score)
        went_to_pens = False

        if sa > sb:
            winner, loser = h, a
        elif sb > sa:
            winner, loser = a, h
        else:
            # Drawn after regulation+ET → shootout. Consult shootouts.csv.
            key = (pd.Timestamp(r.date).strftime("%Y-%m-%d"),
                   frozenset({h, a}))
            sw = shootout_winner.get(key)
            if sw is None:
                print(f"  ! {r.date.date()} {h} {sa}-{sb} {a}: shootout winner "
                      f"unknown, will simulate")
                continue
            winner, loser = sw, (a if sw == h else h)
            went_to_pens = True

        out[frozenset({h, a})] = {
            "winner":       winner,
            "loser":        loser,
            "score_a":      sa if winner == h else sb,
            "score_b":      sb if winner == h else sa,
            "went_to_et":   False,   # cannot reliably infer from results.csv alone
            "went_to_pens": went_to_pens,
        }
    return out


def resolve_match(team_a, team_b,
                  params, elo,
                  rng,
                  played_ko=None,
                  *, neutral=True):
    """Return a knockout match result.

    If (team_a, team_b) is present in `played_ko`, uses the real outcome
    deterministically. Otherwise falls back to Monte Carlo via simulate_match.
    Adds a "played_for_real" flag so callers can tell the two apart.
    """
    if played_ko:
        actual = played_ko.get(frozenset({team_a, team_b}))
        if actual is not None:
            w = actual["winner"]
            # Re-orient scores relative to the (team_a, team_b) framing
            if w == team_a:
                sa, sb = actual["score_a"], actual["score_b"]
            else:
                sa, sb = actual["score_b"], actual["score_a"]
            return {
                "winner": w,
                "loser":  team_b if w == team_a else team_a,
                "score_a": sa, "score_b": sb,
                "went_to_et":   actual.get("went_to_et", False),
                "went_to_pens": actual.get("went_to_pens", False),
                "played_for_real": True,
            }

    r = simulate_match(team_a, team_b, params, elo, rng, neutral=neutral)
    r["played_for_real"] = False
    return r


# ---- Self-test ----

if __name__ == "__main__":
    # Quick smoke test: simulate one match between two known teams.
    from model import load_params, load_elo_table
    p = load_params()
    e = load_elo_table()
    rng = np.random.default_rng(0)
    print("Smoke test: Argentina vs Brazil (neutral)")
    for _ in range(5):
        r = simulate_match("Argentina", "Brazil", p, e, rng)
        et = " (ET)" if r["went_to_et"] else ""
        pen = " (pens)" if r["went_to_pens"] else ""
        print(f"  {r['winner']} beat {r['loser']} {r['score_a']}-{r['score_b']}{et}{pen}")