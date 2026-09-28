from dataclasses import dataclass, field
from typing import Optional, Tuple, List


@dataclass
class SegmentationMetrics:
    total_image_area: int
    actual_scan_area: int
    focus_variance: float
    ac_area: int
    rib_count: int
    vein_present: bool
    stomach_present: bool

    # AI Confidence Scores [0.0 - 1.0]
    ac_conf: float = 0.0
    vein_conf: float = 0.0
    stomach_conf: float = 0.0
    rib_confs: List[float] = field(default_factory=list)

    # Geometric Shape Data
    ac_ellipse: Optional[Tuple[Tuple[float, float], Tuple[float, float], float]] = None
    vein_ellipse: Optional[Tuple[Tuple[float, float], Tuple[float, float], float]] = None

    # Spatial & Linear Data
    stomach_clock: str = "N/A"
    rib_clocks: List[str] = field(default_factory=list)
    rib_lengths: List[float] = field(default_factory=list)


@dataclass
class ScoringResult:
    final_score: int
    quality_label: str