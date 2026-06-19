"""evaluate.py — Held-out evaluation of the Dixon-Coles model vs baselines."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

from model import (DATA_DIR, load_matches, fit_dc,
                   predict_lambda, scoreline_matrix)

CUTOFF = pd.Timestamp("2024-01-01")


# ---- Outcome models ----

def dc_outcomes(p, h_elo, a_elo, neutral):
    ih  = 0 if neutral else 1
    lam = predict_lambda(p, h_elo, a_elo, ih)
    mu  = predict_lambda(p, a_elo, h_elo, 0)
    M   = scoreline_matrix(lam, mu, p["rho"])
    hgi, agi = np.indices(M.shape)
    return (float(M[hgi > agi].sum()),
            float(M[hgi == agi].sum()),
            float(M[hgi < agi].sum()))


def favorite_outcomes(h, a, neutral):
    ha = 0 if neutral else 100
    if h + ha > a: return 1.0, 0.0, 0.0
    if a > h + ha: return 0.0, 0.0, 1.0
    return 0.5, 0.0, 0.5


# ---- Metrics ----

def actual_class(hs, ag): return 0 if hs > ag else 1 if hs == ag else 2
def actual_onehot(hs, ag):
    o = [0.0, 0.0, 0.0]; o[actual_class(hs, ag)] = 1.0; return o
def brier(probs, actuals):
    return float(((np.array(probs) - np.array(actuals))**2).sum(axis=1).mean())
def logloss(probs, actuals):
    p = np.clip(np.array(probs), 1e-12, 1)
    return float(-(np.array(actuals) * np.log(p)).sum(axis=1).mean())
def accuracy(probs, actuals):
    return float((np.array(probs).argmax(1) == np.array(actuals).argmax(1)).mean())


# ---- Evaluation loop ----

def evaluate_subset(test, dc_p, logit, label):
    actuals, dc_p_, log_p_, fav_p_, uni_p_ = [], [], [], [], []
    for r in test.itertuples():
        actuals.append(actual_onehot(r.home_score, r.away_score))
        dc_p_.append(dc_outcomes(dc_p, r.home_elo_pre, r.away_elo_pre, r.neutral))
        x = np.array([[r.home_elo_pre, r.away_elo_pre, 0 if r.neutral else 1]])
        log_p_.append(tuple(logit.predict_proba(x)[0]))
        fav_p_.append(favorite_outcomes(r.home_elo_pre, r.away_elo_pre, r.neutral))
        uni_p_.append((1/3, 1/3, 1/3))

    print(f"\n=== {label} — {len(test):,} matches ===")
    print(f"{'Model':<22} {'Brier':>8} {'LogLoss':>10} {'Accuracy':>10}")
    print("-" * 52)
    for name, probs in [("Dixon-Coles", dc_p_),
                         ("Logistic on Elo", log_p_),
                         ("Always favorite", fav_p_),
                         ("Uniform 1/3", uni_p_)]:
        print(f"{name:<22} {brier(probs, actuals):>8.4f} "
              f"{logloss(probs, actuals):>10.4f} {accuracy(probs, actuals):>10.2%}")
    return dc_p_, actuals


def plot_calibration(curves: dict, actuals, outfile):
    """curves: {label: probs_list}"""
    a_home = np.array(actuals)[:, 0]
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Perfect calibration")

    for label, probs in curves.items():
        p_home = np.array(probs)[:, 0]
        bins   = np.linspace(0, 1, 11)
        idx    = np.clip(np.digitize(p_home, bins) - 1, 0, 9)
        centers, observed, counts = [], [], []
        for b in range(10):
            mask = (idx == b)
            if mask.sum() >= 10:
                centers.append((bins[b] + bins[b+1]) / 2)
                observed.append(a_home[mask].mean())
                counts.append(int(mask.sum()))
        sizes = [40 + 4*c for c in counts]
        ax.scatter(centers, observed, s=sizes, alpha=0.6, label=label)
        ax.plot(centers, observed, alpha=0.3)

    ax.set_xlabel("Predicted P(home win)")
    ax.set_ylabel("Observed fraction of home wins")
    ax.set_title(f"Calibration on held-out matches (post-{CUTOFF.date()})")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(outfile, dpi=120)
    print(f"\nSaved {outfile}")


def main():
    all_m = load_matches()
    train = all_m[all_m["date"] <  CUTOFF].copy()
    test  = all_m[all_m["date"] >= CUTOFF].copy()
    print(f"Cutoff: {CUTOFF.date()}")
    print(f"Train: {len(train):,} matches    Test: {len(test):,} matches")

    print("\nFitting Dixon-Coles on pre-cutoff data only...")
    dc_p = fit_dc(train, ref_date=CUTOFF)
    print("DC coefficients:", {k: round(v, 4) for k, v in dc_p.items()
                                 if isinstance(v, float)})

    print("\nFitting logistic baseline...")
    X_tr = np.column_stack([
        train["home_elo_pre"].values, train["away_elo_pre"].values,
        (~train["neutral"]).astype(int).values,
    ])
    y_tr = np.array([actual_class(hs, ag) for hs, ag in
                     zip(train["home_score"], train["away_score"])])
    logit = LogisticRegression(max_iter=2000).fit(X_tr, y_tr)

    # Build all four probability lists for the test set
    actuals, dc_probs, log_probs, fav_probs, uni_probs = [], [], [], [], []
    for r in test.itertuples():
        actuals.append(actual_onehot(r.home_score, r.away_score))
        dc_probs.append(dc_outcomes(dc_p, r.home_elo_pre, r.away_elo_pre, r.neutral))
        x = np.array([[r.home_elo_pre, r.away_elo_pre, 0 if r.neutral else 1]])
        log_probs.append(tuple(logit.predict_proba(x)[0]))
        fav_probs.append(favorite_outcomes(r.home_elo_pre, r.away_elo_pre, r.neutral))
        uni_probs.append((1/3, 1/3, 1/3))

    print(f"\n=== All held-out matches — {len(test):,} ===")
    print(f"{'Model':<22} {'Brier':>8} {'LogLoss':>10} {'Accuracy':>10}")
    print("-" * 52)
    for name, probs in [("Dixon-Coles", dc_probs),
                         ("Logistic on Elo", log_probs),
                         ("Always favorite", fav_probs),
                         ("Uniform 1/3", uni_probs)]:
        print(f"{name:<22} {brier(probs, actuals):>8.4f} "
              f"{logloss(probs, actuals):>10.4f} {accuracy(probs, actuals):>10.2%}")

    # WC 2026 subset
    wc = test[test["tournament"] == "FIFA World Cup"]
    if len(wc) > 0:
        wc_idx = wc.index
        wc_actuals = [actuals[test.index.get_loc(i)] for i in wc_idx]
        wc_dc      = [dc_probs[test.index.get_loc(i)] for i in wc_idx]
        wc_log     = [log_probs[test.index.get_loc(i)] for i in wc_idx]
        wc_fav     = [fav_probs[test.index.get_loc(i)] for i in wc_idx]
        wc_uni     = [uni_probs[test.index.get_loc(i)] for i in wc_idx]
        print(f"\n=== WC 2026 only — {len(wc)} matches ===")
        print(f"{'Model':<22} {'Brier':>8} {'LogLoss':>10} {'Accuracy':>10}")
        print("-" * 52)
        for name, probs in [("Dixon-Coles", wc_dc),
                             ("Logistic on Elo", wc_log),
                             ("Always favorite", wc_fav),
                             ("Uniform 1/3", wc_uni)]:
            print(f"{name:<22} {brier(probs, wc_actuals):>8.4f} "
                  f"{logloss(probs, wc_actuals):>10.4f} {accuracy(probs, wc_actuals):>10.2%}")

    plot_calibration({"Dixon-Coles": dc_probs, "Logistic on Elo": log_probs},
                     actuals, DATA_DIR / "calibration.png")


if __name__ == "__main__":
    main()