"""
infer_thresholds.py
-------------------
Data-driven threshold inference for AI ultrasound scoring.

This script replaces human guesswork by:
1. Finding the natural "valley" in bimodal expert ratings using K-Means clustering.
2. Binarizing the ground truth at that mathematical split.
3. Calculating the optimal AI geometric threshold using Youden's J statistic (ROC curve)
   to maximize True Positives while minimizing False Positives.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve
from sklearn.cluster import KMeans

# --- CONFIG ---
CSV_PATH = "ai_scoring_report.csv"
XLSX_PATH = "expert rating.xlsx"

# Map the exact Expert component name to the AI metric column
FEATURE_MAPPINGS = [
    ("magnification", "AC_Area"),
    ("fit of outline", "AC_CV_Circularity"),
    ("rib", "max_rib_len"),
    ("vein position", "Vein_Position"),
    ("vein tubularity", "Vein_Solidity")
]


def load_and_clean_data():
    """Loads datasets, merges on image number, and ensures numeric types."""
    # Load data
    exp = pd.read_excel(XLSX_PATH)
    csv = pd.read_csv(CSV_PATH)

    # Clean headers
    exp.columns = [c.strip() for c in exp.columns]

    # Extract ID and deduplicate the AI output
    csv["imgnum"] = csv["Filename"].str.extract(r"_(\d+)\.").astype(float)
    csv = csv.sort_values("Filename").drop_duplicates("imgnum")

    # Merge based on image number
    merged = exp.merge(csv, left_on="Image number", right_on="imgnum", how="inner")

    # Force numeric conversion for AI columns to handle 'N/A' strings safely
    ai_cols = ["AC_Area", "AC_CV_Circularity", "Vein_Position", "Vein_Solidity"]
    for c in ai_cols:
        merged[c] = pd.to_numeric(merged[c], errors="coerce").fillna(0.0)

    # Helper to parse Rib_Lengths (e.g., "212.18 | 275.84")
    def get_max_rib(cell):
        s = str(cell)
        if "N/A" in s or s == "nan": return 0.0
        try:
            return max(float(v) for v in s.split("|"))
        except ValueError:
            return 0.0

    merged["max_rib_len"] = merged["Rib_Lengths"].apply(get_max_rib)

    return merged


def calculate_optimal_thresholds(merged_df):
    """Calculates clustering valleys and Youden's J thresholds."""
    results = []

    for exp_col, ai_col in FEATURE_MAPPINGS:
        # 1. Unsupervised split of expert scores (Find the Valley)
        # Use K-Means to find the centers of the 'good' and 'bad' ratings
        exp_scores = merged_df[exp_col].values.reshape(-1, 1)
        kmeans = KMeans(n_clusters=2, random_state=42, n_init=10).fit(exp_scores)

        # Sort centers so centers[0] is the 'bad' cluster, centers[1] is the 'good' cluster
        centers = sorted(kmeans.cluster_centers_.flatten())
        valley_threshold = (centers[0] + centers[1]) / 2.0

        # 2. Binarize the ground truth based on the mathematical valley
        binarized_expert = (merged_df[exp_col] >= valley_threshold).astype(int)

        # Determine correlation to check if the AI metric goes UP or DOWN for a good score
        # e.g., higher area is good, but lower vein position (distance) might be better
        corr = merged_df[[exp_col, ai_col]].corr().iloc[0, 1]

        ai_vals = merged_df[ai_col]
        # Invert values for ROC calculation if the relationship is inverse
        if corr < 0:
            ai_vals = -ai_vals

        # 3. Find AI threshold using Youden's Index (ROC)
        fpr, tpr, ai_thresholds = roc_curve(binarized_expert, ai_vals)
        youden_j = tpr - fpr
        best_idx = np.argmax(youden_j)
        best_ai_thresh = ai_thresholds[best_idx]

        # Revert threshold sign if we inverted it
        if corr < 0:
            best_ai_thresh = -best_ai_thresh

        # Format rules based on directionality
        operator = "<=" if corr < 0 else ">="

        results.append({
            "Component": exp_col.title(),
            "Expert 'Pass' Score": f">= {valley_threshold:.2f}",
            "AI Metric": ai_col,
            "Direction": operator,
            "Optimal AI Threshold": round(best_ai_thresh, 4),
            "Correlation (r)": round(corr, 3)
        })

    return pd.DataFrame(results)


if __name__ == "__main__":
    df = load_and_clean_data()
    threshold_df = calculate_optimal_thresholds(df)

    print("=" * 80)
    print("DATA-DRIVEN AI THRESHOLD INFERENCE")
    print("=" * 80)
    print(threshold_df.to_string(index=False))

    print("\nNotes:")
    print("- 'Expert Pass Score' is the mathematical valley separating poor/good clusters.")
    print("- 'Optimal AI Threshold' is the geometric cut-off that maximizes True Positives.")
    print("- 'Direction' indicates whether the AI metric should be higher or lower than the threshold.")
    print("- Note the correlation for fit of outline; if it's very low, the threshold is statistically weak.")