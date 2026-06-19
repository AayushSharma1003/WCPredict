"""
model.py — Shared Dixon-Coles primitives.

Loaders, fitting, prediction, and scoreline math used by every entry-point
script (dixon_coles.py, predict.py, evaluate.py, and future ones).
"""

from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.optimize import minimize_scalar
from scipy.stats import poisson


# ---- Paths and constants ----

DATA_DIR    = Path("data")
RESULTS_CSV = DATA_DIR / "results.csv"
ELO_HISTORY = DATA_DIR / "elo_history.csv"
ELO_CURRENT = DATA_DIR / "elo_current.csv"
DC_PARAMS   = DATA_DIR / "dc_params.json"

TIME_DECAY = 0.0019    # per day; half-life ≈ 365 days
MAX_GOALS  = 10        # scoreline matrix truncation


# ---- I/O ----

def load_matches() -> pd.DataFrame:
    """Load elo_history.csv as a clean DataFrame of completed matches."""
    df = pd.read_csv(ELO_HISTORY, parse_dates=["date"])
    df["neutral"] = df["neutral"].astype(str).str.lower().isin(["true", "1"])
    df = df.dropna(subset=["home_score", "away_score"]).copy()
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)
    return df


def load_elo_table() -> dict:
    df = pd.read_csv(ELO_CURRENT)
    return dict(zip(df["team"], df["elo"]))


def save_params(p: dict, path: Path = DC_PARAMS) -> None:
    with open(path, "w") as f:
        json.dump(p, f, indent=2)


def load_params(path: Path = DC_PARAMS) -> dict:
    with open(path) as f:
        return json.load(f)


# ---- Fitting ----

def fit_dc(train: pd.DataFrame, ref_date: pd.Timestamp) -> dict:
    """Fit Dixon-Coles Poisson on `train`, time decay relative to `ref_date`."""
    home = pd.DataFrame({
        "date":     train["date"],
        "team_elo": train["home_elo_pre"],
        "opp_elo":  train["away_elo_pre"],
        "is_home":  (~train["neutral"]).astype(int),
        "goals":    train["home_score"],
    })
    away = pd.DataFrame({
        "date":     train["date"],
        "team_elo": train["away_elo_pre"],
        "opp_elo":  train["home_elo_pre"],
        "is_home":  0,
        "goals":    train["away_score"],
    })
    s = pd.concat([home, away], ignore_index=True)
    s["weight"] = np.exp(-TIME_DECAY * (ref_date - s["date"]).dt.days)

    X = pd.DataFrame({
        "team_elo": (s["team_elo"] - 1500) / 100,
        "opp_elo":  (s["opp_elo"]  - 1500) / 100,
        "is_home":  s["is_home"],
    })
    X = sm.add_constant(X)
    res = sm.GLM(s["goals"], X, family=sm.families.Poisson(),
                 freq_weights=s["weight"]).fit()
    p = {k: float(res.params[k]) for k in ("const", "team_elo", "opp_elo", "is_home")}

    # ρ via MLE on training set (vectorised)
    ih   = (~train["neutral"]).astype(int).values
    z_he = (train["home_elo_pre"].values - 1500) / 100
    z_ae = (train["away_elo_pre"].values - 1500) / 100
    lam  = np.exp(p["const"] + p["team_elo"]*z_he + p["opp_elo"]*z_ae + p["is_home"]*ih)
    mu   = np.exp(p["const"] + p["team_elo"]*z_ae + p["opp_elo"]*z_he)
    hg, ag = train["home_score"].values, train["away_score"].values
    w    = np.exp(-TIME_DECAY * (ref_date - train["date"]).dt.days.values)
    p_pois = poisson.pmf(hg, lam) * poisson.pmf(ag, mu)
    m00 = (hg == 0) & (ag == 0); m01 = (hg == 0) & (ag == 1)
    m10 = (hg == 1) & (ag == 0); m11 = (hg == 1) & (ag == 1)

    def neg_ll(rho: float) -> float:
        tau = np.ones_like(p_pois)
        tau[m00] = 1 - lam[m00] * mu[m00] * rho
        tau[m01] = 1 + lam[m01] * rho
        tau[m10] = 1 + mu[m10]  * rho
        tau[m11] = 1 - rho
        joint = p_pois * tau
        if np.any(joint <= 0):
            return 1e10
        return -np.sum(w * np.log(joint))

    p["rho"]        = float(minimize_scalar(neg_ll, bounds=(-0.2, 0.2),
                                            method="bounded").x)
    p["time_decay"] = TIME_DECAY
    return p


# ---- Prediction primitives ----

def predict_lambda(p: dict, team_elo: float, opp_elo: float, is_home: int) -> float:
    z_team = (team_elo - 1500) / 100
    z_opp  = (opp_elo  - 1500) / 100
    return float(np.exp(p["const"] + p["team_elo"]*z_team
                                   + p["opp_elo"] *z_opp
                                   + p["is_home"] *is_home))


def scoreline_matrix(lam: float, mu: float, rho: float,
                     max_goals: int = MAX_GOALS) -> np.ndarray:
    """Joint P(home=i, away=j) with the Dixon-Coles τ correction."""
    hp = poisson.pmf(np.arange(max_goals + 1), lam)
    ap = poisson.pmf(np.arange(max_goals + 1), mu)
    M  = np.outer(hp, ap)
    M[0, 0] *= 1 - lam * mu * rho
    M[0, 1] *= 1 + lam * rho
    M[1, 0] *= 1 + mu  * rho
    M[1, 1] *= 1 - rho
    M /= M.sum()
    return M


def summarise(M: np.ndarray) -> dict:
    """Marginalise a scoreline matrix into outcomes, totals, side bets, etc."""
    n = M.shape[0]
    hg, ag = np.indices(M.shape)
    total  = hg + ag
    flat = [(M[i, j], i, j) for i in range(n) for j in range(n)]
    flat.sort(reverse=True)
    return {
        "p_home_win":         float(M[hg > ag].sum()),
        "p_draw":             float(M[hg == ag].sum()),
        "p_away_win":         float(M[hg < ag].sum()),
        "exp_home_goals":     float((hg * M).sum()),
        "exp_away_goals":     float((ag * M).sum()),
        "p_btts":             float(M[(hg >= 1) & (ag >= 1)].sum()),
        "p_over_1.5":         float(M[total >= 2].sum()),
        "p_over_2.5":         float(M[total >= 3].sum()),
        "p_over_3.5":         float(M[total >= 4].sum()),
        "p_clean_sheet_home": float(M[ag == 0].sum()),
        "p_clean_sheet_away": float(M[hg == 0].sum()),
        "top_scorelines":     [(i, j, p) for p, i, j in flat[:5]],
    }


def predict_match(home_team: str, away_team: str, *, neutral: bool = True,
                  params: dict | None = None,
                  elo: dict | None = None) -> dict:
    """High-level entry: team names + params/elo → full prediction dict."""
    if params is None: params = load_params()
    if elo    is None: elo    = load_elo_table()
    for t in (home_team, away_team):
        if t not in elo:
            raise KeyError(f"Team not in Elo table: {t!r}")

    e_h, e_a = elo[home_team], elo[away_team]
    ih  = 0 if neutral else 1
    lam = predict_lambda(params, e_h, e_a, ih)
    mu  = predict_lambda(params, e_a, e_h, 0)
    M   = scoreline_matrix(lam, mu, params["rho"])
    s   = summarise(M)
    s.update(home_team=home_team, away_team=away_team,
             home_elo=e_h, away_elo=e_a, neutral=neutral,
             lambda_home=lam, lambda_away=mu)
    return s


# ---- Vectorised prediction (for the simulator) ----

def predict_lambdas(p: dict,
                    team_elos: np.ndarray,
                    opp_elos:  np.ndarray,
                    is_homes:  np.ndarray) -> np.ndarray:
    """Vectorised version of predict_lambda. All inputs are arrays of the same length."""
    z_team = (np.asarray(team_elos, dtype=float) - 1500) / 100
    z_opp  = (np.asarray(opp_elos,  dtype=float) - 1500) / 100
    ih     =  np.asarray(is_homes,  dtype=float)
    return np.exp(p["const"]
                  + p["team_elo"] * z_team
                  + p["opp_elo"]  * z_opp
                  + p["is_home"]  * ih)


def validate_params(p: dict) -> None:
    """Raise if params are missing keys or have implausible values."""
    required = {"const", "team_elo", "opp_elo", "is_home", "rho"}
    missing  = required - set(p.keys())
    if missing:
        raise ValueError(f"DC params missing keys: {missing}")
    if not -0.3 < p["rho"] < 0.3:
        raise ValueError(f"rho={p['rho']} outside plausible range (-0.3, 0.3)")
    if p["team_elo"] <= 0 or p["opp_elo"] >= 0:
        raise ValueError(f"Sign of team_elo/opp_elo coefficients is wrong")