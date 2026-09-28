"""
quality_evaluator.py
--------------------
Calculates the ultrasound quality score using data-driven thresholds
and strict clinical rules.

FIXES APPLIED (magnification sub-component):
  * Denominator is now the SCAN SECTOR (metrics.actual_scan_area), not the whole
    frame (metrics.total_image_area). The clinical rule is "abdomen fills at least
    1/6 of the ultrasound", not of the screen.
  * Threshold set to 1/6 accordingly.
  * Numerator (metrics.ac_area) is now rasterised in the same full-res frame as
    actual_scan_area -- see the coordinate-frame fix in batch_scores.py.
"""

from metrics_types import SegmentationMetrics, ScoringResult
from scoring_exceptions import MissingDataError

# --- SCORING CONSTANTS ---
WEIGHT_AC = 0.40
WEIGHT_VEIN = 0.25
WEIGHT_STOMACH = 0.20
WEIGHT_RIB = 0.15

SCORE_MULTIPLIER = 100

# --- DATA-DRIVEN THRESHOLDS ---
# Magnification: abdomen must fill >= 1/6 of the SCAN SECTOR (not the frame).
# NOTE: confirm 1/6 vs 0.50 with the expert -- this is a one-line change.
MIN_AC_AREA_RATIO = 1.0 / 6.0     # ~0.1667 of the scan sector
MIN_AC_CONFIDENCE = 0.85          # Replaces Circularity for Outline Fit
MIN_RIB_LENGTH = 147.85           # Optimal Rib validation
MIN_VEIN_SOLIDITY = 0.625         # Vein Tubularity validation
VEIN_POSITION_THRESH = 0.371      # Vein Position validation


def calculate_basic_quality_score(
        metrics: SegmentationMetrics,
        vein_solidity: float = 0.0,
        vein_position: float = 0.0,
        max_rib_len: float = 0.0
) -> ScoringResult:
    """
    Evaluates the segmentation metrics against standardized thresholds.
    Assigns a weighted score based on component pass/fail criteria.
    """
    # Guard the ACTUAL denominator now in use.
    if metrics.actual_scan_area <= 0:
        raise MissingDataError("Scan-sector area is invalid or zero.")

    raw_score = 0.0

    # ---------------------------------------------------------
    # 1. Abdominal Circumference (AC) - 40%
    # ---------------------------------------------------------
    # DENOMINATOR FIX: divide by the scan sector, not the whole frame.
    ac_ratio = metrics.ac_area / metrics.actual_scan_area

    # Sub-component A: Magnification (20%)
    if ac_ratio >= MIN_AC_AREA_RATIO:
        raw_score += (WEIGHT_AC / 2)

    # Sub-component B: Fit of Outline (20%)
    if metrics.ac_conf >= MIN_AC_CONFIDENCE:
        raw_score += (WEIGHT_AC / 2)

    # ---------------------------------------------------------
    # 2. Vein - 25%
    # ---------------------------------------------------------
    if metrics.vein_present:
        # Sub-component A: Tubularity (12.5%)
        if vein_solidity >= MIN_VEIN_SOLIDITY:
            raw_score += (WEIGHT_VEIN / 2)

        # Sub-component B: Position (12.5%)
        if vein_position >= VEIN_POSITION_THRESH:
            raw_score += (WEIGHT_VEIN / 2)

    # ---------------------------------------------------------
    # 3. Stomach - 20%
    # ---------------------------------------------------------
    if metrics.stomach_present:
        raw_score += WEIGHT_STOMACH

    # ---------------------------------------------------------
    # 4. Ribs - 15%
    # ---------------------------------------------------------
    if max_rib_len >= MIN_RIB_LENGTH:
        raw_score += WEIGHT_RIB

    # ---------------------------------------------------------
    # Final Score Calculation
    # ---------------------------------------------------------
    final_score = int(round(raw_score * SCORE_MULTIPLIER))

    if final_score == 100:
        label = "Optimal"
    elif final_score >= 60:
        label = "Acceptable"
    else:
        label = "Suboptimal"

    return ScoringResult(final_score=final_score, quality_label=label)