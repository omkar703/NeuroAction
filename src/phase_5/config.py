"""Configuration parameters for Phase 5: Independent Scientific Validation."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple, Dict, Any


@dataclass
class Phase5Config:
    """Master configuration for Phase 5 validation pipeline."""

    # Paths
    project_root: Path = field(default_factory=lambda: Path.cwd())
    manifest_path: Path = field(default_factory=lambda: Path("data/manifests/iackd_manifest.csv"))
    output_dir: Path = field(default_factory=lambda: Path("reports/phase_5"))
    figures_dir: Path = field(default_factory=lambda: Path("reports/phase_5/figures"))
    demo_dir: Path = field(default_factory=lambda: Path("reports/phase_5/demo"))
    model_dir_phase_2: Path = field(default_factory=lambda: Path("models/phase_2"))
    model_dir_phase_3: Path = field(default_factory=lambda: Path("models/phase_3"))
    cache_dir: Path = field(default_factory=lambda: Path("data/cache"))

    # Benchmark Cohort
    benchmark_subjects: List[str] = field(
        default_factory=lambda: ["sub-01", "sub-02", "sub-03", "sub-08", "sub-15"]
    )

    # Kinematic Segmentation Parameters
    speed_threshold_mm_s: float = 20.0
    prefix_duration_ms: float = 200.0
    prefix_sample_count: int = 20
    sampling_rate_kinematics_hz: float = 100.0
    sampling_rate_eeg_hz: float = 250.0

    # Forecasting Horizons
    horizons_ms: List[int] = field(default_factory=lambda: [100, 200, 300, 500])

    # Target Geometry & Tolerances
    target_tolerance_mm: float = 50.0
    target_angle_threshold_deg: float = 45.0
    tolerance_grid_mm: List[float] = field(default_factory=lambda: [25.0, 50.0, 75.0, 100.0])

    # Decision & Abstention Policy
    confidence_threshold: float = 0.60
    distance_abstention_margin_mm: float = 15.0  # Abstain if |d - tolerance| <= margin
    uncertainty_distance_range_mm: Tuple[float, float] = (40.0, 60.0)

    # Prospective Data Collection Planning Defaults
    prospective_target_radius_mm: float = 40.0
    prospective_sampling_rate_hz: float = 100.0
    prospective_mocap_coordinate_frame: str = "optitrack_cartesian_mm"

    def __post_init__(self):
        """Ensure directories exist."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self.demo_dir.mkdir(parents=True, exist_ok=True)
