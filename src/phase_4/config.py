"""Configuration schemas and default hyperparameters for Phase 4 Digital Twin."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class DigitalTwinConfig:
    """Central configuration for Phase 4 Multimodal Digital Twin runtime."""

    # 1. Paths
    manifest_path: Path = Path("data/manifests/iackd_manifest.csv")
    model_dir_phase_2: Path = Path("models/phase_2")
    model_dir_phase_3: Path = Path("models/phase_3")
    output_dir: Path = Path("reports/phase_4")
    figures_dir: Path = Path("reports/phase_4/figures")
    demo_dir: Path = Path("reports/phase_4/demo")

    # 2. Demonstration & Evaluation Defaults
    demo_trial_id: str = "sub-01_run-01_L1_tr00"
    default_protocol: str = "kinematic_onset"  # "kinematic_onset" (Protocol B) or "event_locked" (Protocol A)
    held_out_participant: str = "sub-01"

    # 3. Conflict Detection Thresholds (Configurable Engineering Parameters)
    # Reference distance for continuous score normalization: d_ref = 100 mm (half lateral target separation)
    distance_reference_mm: float = 100.0
    # Discrete threshold for endpoint-target disagreement: d_threshold = 50 mm
    distance_threshold_mm: float = 50.0
    # Angular threshold for directional disagreement: theta_threshold = 45.0 degrees
    directional_threshold_deg: float = 45.0
    # Low confidence threshold for UNCERTAIN classification
    confidence_uncertainty_threshold: float = 0.55

    # 4. Temporal & Kinematic Parameters
    sampling_rate_kinematics_hz: float = 100.0
    sampling_rate_eeg_hz: float = 200.0
    prefix_duration_ms: float = 200.0
    prefix_samples: int = 20  # 200 ms @ 100 Hz
    speed_threshold_mm_s: float = 20.0  # Movement-onset velocity threshold
    horizons_ms: List[int] = field(default_factory=lambda: [100, 200, 300, 500])

    # 5. Spatial Target Coordinates (Centroids in Leap Motion Workspace in mm)
    target_left_coord: List[float] = field(default_factory=lambda: [-188.0, 150.0, 53.0])
    target_right_coord: List[float] = field(default_factory=lambda: [187.5, 149.0, 51.0])
    coordinate_units: str = "mm"
    coordinate_frame: str = "leap_motion_cartesian"

    def __post_init__(self):
        """Ensure directories exist."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self.demo_dir.mkdir(parents=True, exist_ok=True)
