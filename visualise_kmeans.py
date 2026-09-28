"""
visualise_valley.py
-------------------
This script generates a visual plot showing exactly how the K-Means algorithm
finds the "Valley" (the Pass/Fail cut-off) in the human expert grades.

It saves an image named 'valley_visualization.png' in your folder.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.stats import linregress
import os

# --- CONFIG ---
XLSX_PATH = "expert rating.xlsx"
CSV_PATH = "ai_scoring_report.csv"
COMPONENT_TO_PLOT = "fit of outline"  # You can change this to 'fit of outline', 'vein position', etc.

# Maps the expert component to the AI metric for the X/Y plot
FEATURE_MAPPINGS = {
    "magnification": "AC_Area",
    "fit of outline": "AC_CV_Circularity",
    "rib": "max_rib_len",
    "vein position": "Vein_Position",
    "vein tubularity": "Vein_Solidity"
}

def main():
    print(f"Loading data from {XLSX_PATH} and {CSV_PATH}...")
    try:
        exp = pd.read_excel(XLSX_PATH)
        csv = pd.read_csv(CSV_PATH)
    except FileNotFoundError as e:
        print(f"Error: {e}. Make sure both files are in the same folder.")
        return

    exp.columns = [c.strip() for c in exp.columns]

    if COMPONENT_TO_PLOT not in exp.columns:
        print(f"Error: Column '{COMPONENT_TO_PLOT}' not found in the Excel file.")
        return

    # Clean and merge data to get both AI metrics and Expert grades
    csv["imgnum"] = csv["Filename"].str.extract(r"_(\d+)\.").astype(float)
    csv = csv.sort_values("Filename").drop_duplicates("imgnum")
    merged = exp.merge(csv, left_on="Image number", right_on="imgnum", how="inner")

    # Helper to parse Rib_Lengths
    def get_max_rib(cell):
        s = str(cell)
        if "N/A" in s or s == "nan": return 0.0
        try: return max(float(v) for v in s.split("|"))
        except ValueError: return 0.0

    merged["max_rib_len"] = merged["Rib_Lengths"].apply(get_max_rib)

    # Ensure AI column is numeric
    ai_col = FEATURE_MAPPINGS.get(COMPONENT_TO_PLOT)
    merged[ai_col] = pd.to_numeric(merged[ai_col], errors="coerce").fillna(0.0)

    # Extract the scores for the specific component
    scores = merged[COMPONENT_TO_PLOT].dropna().values.reshape(-1, 1)

    # 1. Run K-Means to find the two "piles"
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10).fit(scores)
    centers = sorted(kmeans.cluster_centers_.flatten())

    # 2. Calculate the exact dead center between the two piles
    valley_threshold = (centers[0] + centers[1]) / 2.0

    # 3. Create the visual plot
    plt.figure(figsize=(10, 4))

    # Add a tiny bit of random vertical scatter ("jitter") so dots with the same score don't perfectly overlap
    jitter = np.random.normal(0, 0.02, size=len(scores))

    # Draw the expert scores as blue dots
    plt.scatter(scores, jitter, alpha=0.6, color='dodgerblue', edgecolor='black', s=80, label="Individual Expert Scores")

    # Draw the centers of the two piles as giant red stars
    plt.scatter(centers, [0, 0], color='red', marker='*', s=400, edgecolor='black', label="Centers of the 'Poor' and 'Great' Piles")

    # Draw the Valley as a green dashed line
    plt.axvline(valley_threshold, color='green', linestyle='dashed', linewidth=3, label=f"The Valley (Calculated Cut-off: {valley_threshold:.2f})")

    # Format the chart to look nice
    plt.title(f"How the AI finds the Pass/Fail Line for '{COMPONENT_TO_PLOT.title()}'", fontsize=14, pad=15)
    plt.xlabel("Expert Grade (0.0 to 1.0)", fontsize=12)
    plt.xlim(-0.1, 1.1)
    plt.yticks([])  # Hide the Y-axis numbers since they just represent our random jitter

    # Move the legend outside the plot so it doesn't cover any dots
    plt.legend(loc='upper right', bbox_to_anchor=(1.45, 1))

    # Save the image
    output_filename = f"valley_{COMPONENT_TO_PLOT.replace(' ', '_')}.png"
    plt.tight_layout()
    plt.savefig(output_filename, dpi=200, bbox_inches="tight")
    print(f"\nSuccess! Open the image file '{output_filename}' to see the visualization.")
    plt.close() # Close figure to prevent overlap

    # ==========================================
    # NEW SECTION: VISUALIZE THE TREND (CORRELATION)
    # ==========================================
    print("Generating the Correlation Trend plot...")

    # Get X (AI Metric) and Y (Expert Grade)
    x = merged[ai_col].values
    y = merged[COMPONENT_TO_PLOT].values

    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, alpha=0.7, color='purple', edgecolor='black', s=60, label="Image Data Point (50 total)")

    # Calculate line of best fit (Correlation)
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    line_x = np.array([x.min(), x.max()])
    line_y = slope * line_x + intercept

    # Determine the pattern name for the title
    if r_value >= 0.3:
        pattern = "Uphill Trend (Positive Connection)"
        line_color = "green"
    elif r_value <= -0.3:
        pattern = "Downhill Trend (Negative Connection)"
        line_color = "green"
    else:
        pattern = "No Strong Connection"
        line_color = "red"

    # Draw the trendline
    plt.plot(line_x, line_y, color=line_color, linestyle='dashed', linewidth=3, label=f"Trendline (r = {r_value:.2f})")

    # Format the chart
    plt.title(f"Does the AI Match the Human?\nPattern: {pattern}", fontsize=14, pad=15)
    plt.xlabel(f"AI Raw Measurement: {ai_col}", fontsize=12)
    plt.ylabel(f"Expert Grade: {COMPONENT_TO_PLOT.title()} (0.0 to 1.0)", fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='best')

    # Save the second image
    trend_filename = f"trend_{COMPONENT_TO_PLOT.replace(' ', '_')}.png"
    plt.tight_layout()
    plt.savefig(trend_filename, dpi=200, bbox_inches="tight")
    print(f"Success! Open the image file '{trend_filename}' to see the trend pattern.\n")

if __name__ == "__main__":
    main()