# --- CONFIDENCE WEIGHTS ---
# Total should equal 1.0
WEIGHT_AC = 0.40
WEIGHT_VEIN = 0.25
WEIGHT_STOMACH = 0.20
WEIGHT_RIB = 0.15

# --- SCORING THRESHOLDS (Data-Driven) ---
MIN_AC_AREA_RATIO = 0.50          # Clinical Magnification override
MIN_RIB_LENGTH = 147.85           # Optimal Rib validation
MIN_VEIN_SOLIDITY = 0.625         # Vein Tubularity validation
VEIN_POSITION_THRESH = 0.371      # Vein Position validation
MIN_AC_CONFIDENCE = 0.85