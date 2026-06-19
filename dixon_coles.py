"""
dixon_coles.py — Fit a Dixon-Coles Poisson model for international football.

Reads:  data/elo_history.csv
Writes: data/dc_params.json
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.optimize import minimize_scalar
from scipy.stats import poisson

# ---- Config ----
DATA_DIR    = Path("data")
ELO_HISTORY = DATA_DIR / "elo_history.csv"
OUT_PARAMS  = DATA_DIR / "dc_params.json"

TRAIN_FROM  = "2000-01-01"
TIME_DECAY  = 0.0019      # per day; half-life ≈ 365 days
TODAY       = pd.Timestamp.today().normalize()


def load_matches() -> pd.DataFrame:
    df = pd.read_csv(ELO_HISTORY, parse_dates=["date"])
    df["neutral"] = df["neutral"].astype(str).str.lower().isin(["true", "1"])
    df = df[df["date"] >= TRAIN_FROM].copy()
    df = df.dropna(subset=["home_score", "away_score"])
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)
    return df.reset_index(drop=True)


def build_stacked(df: pd.DataFrame) -> pd.DataFrame:
    """Stack each match into two rows: home perspective + away perspective."""
    home = pd.DataFrame({
        "date":     df["date"],
        "team_elo": df["home_elo_pre"],
        "opp_elo":  df["away_elo_pre"],
        "is_home":  (~df["neutral"]).astype(int),
        "goals":    df["home_score"],
    })
    away = pd.DataFrame({
        "date":     df["date"],
        "team_elo": df["away_elo_pre"],
        "opp_elo":  df["home_elo_pre"],
        "is_home":  0,                              # away team is never the host
        "goals":    df["away_score"],
    })
    s = pd.concat([home, away], ignore_index=True)
    days_ago    = (TODAY - s["date"]).dt.days
    s["weight"] = np.exp(-TIME_DECAY * days_ago)
    return s


def fit_poisson(s: pd.DataFrame):
    """Weighted Poisson regression with Elo standardized as (elo - 1500) / 100."""
    X = pd.DataFrame({
        "team_elo": (s["team_elo"] - 1500) / 100,
        "opp_elo":  (s["opp_elo"]  - 1500) / 100,
        "is_home":  s["is_home"],
    })
    X = sm.add_constant(X)
    model  = sm.GLM(s["goals"], X, family=sm.families.Poisson(), freq_weights=s["weight"])
    return model.fit()


def predict_lambda(p: dict, team_elo: float, opp_elo: float, is_home: int) -> float:
    z_team = (team_elo - 1500) / 100
    z_opp  = (opp_elo  - 1500) / 100
    return np.exp(p["const"] + p["team_elo"] * z_team
                             + p["opp_elo"]  * z_opp
                             + p["is_home"]  * is_home)


def fit_rho(df: pd.DataFrame, p: dict) -> float:
    """Find ρ that maximises the Dixon-Coles low-score-corrected likelihood."""
    is_home = (~df["neutral"]).astype(int).values
    lam = predict_lambda(p, df["home_elo_pre"].values, df["away_elo_pre"].values, is_home)
    mu  = predict_lambda(p, df["away_elo_pre"].values, df["home_elo_pre"].values, 0)

    hg, ag  = df["home_score"].values, df["away_score"].values
    days    = (TODAY - df["date"]).dt.days.values
    w       = np.exp(-TIME_DECAY * days)

    p_pois = poisson.pmf(hg, lam) * poisson.pmf(ag, mu)
    m00, m01, m10, m11 = (hg == 0) & (ag == 0), (hg == 0) & (ag == 1), \
                         (hg == 1) & (ag == 0), (hg == 1) & (ag == 1)

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

    res = minimize_scalar(neg_ll, bounds=(-0.2, 0.2), method="bounded")
    return float(res.x)


def main():
    df = load_matches()
    print(f"Training on {len(df):,} matches from {TRAIN_FROM} to {df['date'].max().date()}")

    s = build_stacked(df)
    print(f"Stacked into {len(s):,} team-match observations")

    print("\nFitting Poisson regression...")
    result = fit_poisson(s)
    params = {k: float(result.params[k]) for k in ("const", "team_elo", "opp_elo", "is_home")}
    print("Coefficients (Elo standardized to (elo-1500)/100):")
    for k, v in params.items():
        print(f"  {k:10s} = {v:+.4f}")

    print("\nFitting Dixon-Coles ρ...")
    rho = fit_rho(df, params)
    print(f"  rho = {rho:+.4f}")

    print("\nSanity check — expected goals for a few matchups:")
    samples = [
        ("Argentina (2198) vs Brazil (2052), neutral",          2198, 2052, 0),
        ("France (2127) home vs Germany (2001)",                2127, 2001, 1),
        ("Spain (2172) vs Saudi Arabia (~1500), neutral",       2172, 1500, 0),
        ("Saudi Arabia (~1500) home vs Argentina (2198)",       1500, 2198, 1),
    ]
    for label, e1, e2, ish in samples:
        l1 = predict_lambda(params, e1, e2, ish)
        l2 = predict_lambda(params, e2, e1, 0)
        print(f"  {label}: λ={l1:.2f} vs {l2:.2f}")

    out = {**params, "rho": rho, "time_decay": TIME_DECAY, "train_from": TRAIN_FROM}
    with open(OUT_PARAMS, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {OUT_PARAMS}")


if __name__ == "__main__":
    main()