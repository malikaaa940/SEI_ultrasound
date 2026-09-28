"""
calibrate_scoring.py
--------------------
Calibrate the ultrasound quality score against expert component ratings.

Instead of hand-setting per-feature thresholds from extreme order statistics,
this fits ONE small regularised model that maps the AI's geometric features to
the experts' summed component score, and validates it honestly with
leave-one-out cross-validation (every image is predicted by a model that never
saw it).

Ground truth  : expert_rating.xlsx  (50 images, 6 components each, 0-1 in 0.2 steps)
AI features   : ai_scoring_report.csv (geometry from the YOLOv8-seg pipeline)

Outputs:
  - console: LOO Spearman / MAE, a bootstrap CI, learned weights, per-image table
  - calibration_plot.png: predicted vs expert total, leave-one-out

Everything here is derived scores + geometry (no pixels), so it runs locally.

Dependencies: pandas, numpy, scipy, scikit-learn, matplotlib
"""

import re
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import LeaveOneOut, cross_val_predict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------- #
# CONFIG
# --------------------------------------------------------------------------- #
CSV_PATH   = "ai_scoring_report.csv"
XLSX_PATH  = "expert rating.xlsx"
PLOT_PATH  = "calibration_plot.png"

# Expert component columns (summed to a 0-6 ground-truth total)
EXPERT_COMPONENTS = ["magnification", "fit of outline", "rib",
                     "stomach", "vein position", "vein tubularity"]

# AI features fed to the model. Deliberately small (n=50 -> keep it simple).
# Absence-aware: when no vein is detected, solidity/position are set to 0,
# because absence is itself a quality signal.
AI_FEATURES = ["AC_Area", "AC_Confidence", "n_ribs", "max_rib_len",
               "stomach_ai", "Vein_Solidity", "Vein_Position"]

RIDGE_ALPHA = 5.0     # regularisation strength; higher = simpler/safer at low n
LOGREG_C    = 0.5     # inverse regularisation for the good/poor classifier
N_BOOT      = 2000    # bootstrap resamples for the CI on LOO performance
RANDOM_SEED = 0


# --------------------------------------------------------------------------- #
# LOADING & CLEANING
# --------------------------------------------------------------------------- #
def _to_num(series):
    return pd.to_numeric(series, errors="coerce")


def _max_rib_length(cell):
    """Rib_Lengths is a pipe-joined string like '212.18 | 275.84', or 'N/A'."""
    s = str(cell)
    if "N/A" in s or s == "nan":
        return 0.0
    try:
        return max(float(x) for x in s.split("|"))
    except ValueError:
        return 0.0


def load_and_merge(csv_path=CSV_PATH, xlsx_path=XLSX_PATH):
    csv = pd.read_csv(csv_path)
    exp = pd.read_excel(xlsx_path)
    exp.columns = [c.strip() for c in exp.columns]

    # --- expert side: image number + summed component score ---
    exp["imgnum"] = exp["Image number"].astype(int)
    exp["expert_sum"] = exp[EXPERT_COMPONENTS].sum(axis=1)

    # --- AI side: pull the image number out of the filename ---
    csv["imgnum"] = csv["Filename"].str.extract(r"_(\d+)\.").astype(int)

    # The labelled prefixes (Optimal_/Suboptimal_) duplicate images that also
    # appear as Unlabeled_. Collapse to one row per image number.
    n_before = len(csv)
    csv = csv.sort_values("Filename").drop_duplicates("imgnum")
    n_dropped = n_before - len(csv)

    # --- derive clean numeric features ---
    csv["AC_Area"] = _to_num(csv["AC_Area"])
    # Using .get() to safely handle cases if the script is run before batch_scores is updated
    csv["AC_Confidence"] = _to_num(csv.get("AC_Confidence")).fillna(0.0)
    csv["n_ribs"] = _to_num(csv["Ribs"]).fillna(0)
    csv["max_rib_len"] = csv["Rib_Lengths"].apply(_max_rib_length)
    csv["stomach_ai"] = csv["Stomach"].astype(str).map({"True": 1, "False": 0})
    csv["vein_ai"] = csv["Vein"].astype(str).map({"True": 1, "False": 0})
    csv["Vein_Solidity"]  = _to_num(csv["Vein_Solidity"]).fillna(0.0)
    csv["Vein_Position"]  = _to_num(csv["Vein_Position"]).fillna(0.0)

    merged = exp.merge(csv, on="imgnum", how="inner")
    return merged, n_dropped


# --------------------------------------------------------------------------- #
# MODELLING
# --------------------------------------------------------------------------- #
def leave_one_out_predictions(X, y):
    """Ridge total-score prediction, one held-out image at a time."""
    pipe = make_pipeline(StandardScaler(), Ridge(alpha=RIDGE_ALPHA))
    preds = cross_val_predict(pipe, X, y, cv=LeaveOneOut())
    return preds


def bootstrap_spearman_ci(y, preds, n_boot=N_BOOT, seed=RANDOM_SEED):
    """CI on the LOO Spearman correlation, to show how fragile it is at n=50."""
    rng = np.random.default_rng(seed)
    n = len(y)
    rs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y[idx])) < 3:
            continue
        rs.append(stats.spearmanr(y[idx], preds[idx]).statistic)
    lo, hi = np.percentile(rs, [2.5, 97.5])
    return lo, hi


def logistic_good_poor(X, y):
    """Binary good/poor at the median expert total, LOO accuracy."""
    thr = float(np.median(y))
    yb = (y >= thr).astype(int)
    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(C=LOGREG_C, max_iter=1000))
    pb = cross_val_predict(clf, X, yb, cv=LeaveOneOut())
    acc = float((pb == yb).mean())
    base = float(max(yb.mean(), 1 - yb.mean()))
    return thr, acc, base


def learned_weights(X, y, feature_names):
    """Full-fit standardised ridge weights (interpretability only)."""
    pipe = make_pipeline(StandardScaler(), Ridge(alpha=RIDGE_ALPHA)).fit(X, y)
    coef = pipe.named_steps["ridge"].coef_
    return sorted(zip(feature_names, coef), key=lambda t: -abs(t[1]))


# --------------------------------------------------------------------------- #
# PLOT
# --------------------------------------------------------------------------- #
def make_plot(y, preds, r_loo, mae, path=PLOT_PATH):
    fig, ax = plt.subplots(figsize=(6.2, 6.0))
    ax.scatter(y, preds, s=45, alpha=0.75, edgecolor="black", linewidth=0.5)
    lo = min(y.min(), preds.min()) - 0.3
    hi = max(y.max(), preds.max()) + 0.3
    ax.plot([lo, hi], [lo, hi], "--", color="grey", linewidth=1,
            label="perfect agreement (y = x)")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("Expert summed component score (/6.0)")
    ax.set_ylabel("Model prediction, leave-one-out (/6.0)")
    ax.set_title(f"Calibrated score vs expert judgment\n"
                 f"LOO Spearman r = {r_loo:.2f}   MAE = {mae:.2f}/6.0   n = {len(y)}")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"\nSaved plot -> {path}")


# --------------------------------------------------------------------------- #
# MAIN
# --------------------------------------------------------------------------- #
def main():
    merged, n_dropped = load_and_merge()
    n = len(merged)

    X = merged[AI_FEATURES].values.astype(float)
    y = merged["expert_sum"].values.astype(float)

    print("=" * 68)
    print("EXPERT-CALIBRATED SCORING  |  leave-one-out validation")
    print("=" * 68)
    print(f"Expert-rated images merged with AI outputs : n = {n}")
    print(f"Duplicate prefix rows collapsed            : {n_dropped}")
    print(f"Features                                   : {', '.join(AI_FEATURES)}")

    # --- baseline: how well does the CURRENT pipeline score already track experts? ---
    if "Score" in merged.columns:
        r_base = stats.spearmanr(y, _to_num(merged["Score"])).statistic
        print(f"\nBaseline  (current AI 'Score' vs expert_sum): Spearman r = {r_base:+.2f}")

    # --- calibrated model, honestly validated ---
    preds = leave_one_out_predictions(X, y)
    r_loo = stats.spearmanr(y, preds).statistic
    mae   = float(np.mean(np.abs(y - preds)))
    lo, hi = bootstrap_spearman_ci(y, preds)
    print(f"Calibrated (ridge, LOO)      : Spearman r = {r_loo:+.2f}  "
          f"[95% CI {lo:+.2f}, {hi:+.2f}]   MAE = {mae:.2f}/6.0")

    thr, acc, base = logistic_good_poor(X, y)
    print(f"Good/poor @ median ({thr:.1f}/6): LOO accuracy = {acc:.0%}  "
          f"(base rate {base:.0%})")

    # --- which features drive expert judgment ---
    print("\nStandardised ridge weights (larger |w| = more influence):")
    for f, c in learned_weights(X, y, AI_FEATURES):
        bar = "#" * int(round(abs(c) * 20))
        print(f"   {f:16s} {c:+.3f}  {bar}")

    # --- per-image table ---
    tbl = merged[["imgnum", "expert_sum"]].copy()
    tbl["pred_LOO"] = np.round(preds, 2)
    tbl["abs_err"]  = np.round(np.abs(preds - y), 2)
    tbl = tbl.sort_values("abs_err", ascending=False)
    print("\nPer-image (worst-predicted first) -- inspect the top few for label/segmentation issues:")
    print(tbl.to_string(index=False))

    make_plot(y, preds, r_loo, mae)

    print("\nNotes:")
    print("  * n=50 is small; treat r as 'explains ~half the ranking', not calibrated.")
    print("  * Vein sub-features are 0 when no vein detected (absence = signal).")
    print("  * Circularity was dropped and replaced by AC_Confidence for outline fit.")
    print("  * Lock these weights, then confirm on images that had no part in fitting.")


if __name__ == "__main__":
    main()