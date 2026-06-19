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
"""

import numpy as np

from model import predict_lambda, scoreline_matrix, MAX_GOALS
from bracket_data import R32_SCHEDULE, get_third_place_assignment

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
