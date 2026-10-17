"""Schema definitions for Phase 3: Continuous 3D Kinematics and Trajectory Forecasting."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np


@dataclass
class TrajectorySample:
    """Canonical kinematic sample containing segmented movement prefix and future targets."""

    participant_id: str
    run_id: str
    trial_id: str
    congruency: str
    target_goal: str
    physical_hand_direction: str
    sampling_rate: float = 100.0
    coordinate_units: str = "mm"
    segmentation_mode: str = "event_locked"  # "event_locked" or "kinematic_onset"
    onset_latency_ms: float = 0.0            # Delay from event marker to physical onset

    # Movement timing
    t_onset_ms: float = 0.0
    onset_sample_idx: int = 0
    movement_duration_ms: float = 0.0

    # Observed 200 ms prefix (20 samples at 100 Hz, shape: (20, 3))
    prefix_positions: np.ndarray = field(default_factory=lambda: np.zeros((20, 3), dtype=np.float32))
    prefix_velocities: np.ndarray = field(default_factory=lambda: np.zeros((20, 3), dtype=np.float32))
    prefix_times: np.ndarray = field(default_factory=lambda: np.zeros((20,), dtype=np.float64))

    # Actual final endpoint (shape: (3,))
    actual_endpoint: np.ndarray = field(default_factory=lambda: np.zeros((3,), dtype=np.float32))
    endpoint_time_ms: float = 0.0

    # Future trajectory horizons relative to prefix end (+100, +200, +300, +500 ms)
    # Positions over the whole horizon window, e.g. horizon_trajectories[100] shape: (10, 3)
    horizon_trajectories: Dict[int, np.ndarray] = field(default_factory=dict)
    # Point target at horizon end, e.g. horizon_points[100] shape: (3,)
    horizon_points: Dict[int, np.ndarray] = field(default_factory=dict)

    # Full trial trajectory starting from movement onset
    full_trajectory: Optional[np.ndarray] = None  # shape: (T, 3)
    full_times: Optional[np.ndarray] = None        # shape: (T,)


@dataclass
class ForecastResult:
    """Unified forecast result representing predictions from a model on a single trial."""

    participant_id: str
    run_id: str
    trial_id: str
    model_name: str
    fold_id: str = ""
    congruency: str = ""
    target_goal: str = ""
    physical_hand_direction: str = ""

    observation_duration_ms: float = 200.0
    forecast_horizon_ms: int = 200
    sampling_rate_hz: float = 100.0
    coordinate_units: str = "mm"

    # 3D arrays
    observed_positions: np.ndarray = field(default_factory=lambda: np.zeros((20, 3), dtype=np.float32))
    predicted_positions: np.ndarray = field(default_factory=lambda: np.zeros((20, 3), dtype=np.float32))
    ground_truth_positions: np.ndarray = field(default_factory=lambda: np.zeros((20, 3), dtype=np.float32))

    predicted_endpoint: np.ndarray = field(default_factory=lambda: np.zeros((3,), dtype=np.float32))
    actual_endpoint: np.ndarray = field(default_factory=lambda: np.zeros((3,), dtype=np.float32))

    # Error metrics (in mm)
    endpoint_error_mm: float = 0.0
    endpoint_error_x_mm: float = 0.0
    endpoint_error_y_mm: float = 0.0
    endpoint_error_z_mm: float = 0.0

    trajectory_error_mm: float = 0.0
    trajectory_error_x_mm: float = 0.0
    trajectory_error_y_mm: float = 0.0
    trajectory_error_z_mm: float = 0.0

    def to_dict(self) -> Dict:
        """Convert scalar fields to dictionary for tabular export."""
        return {
            "participant_id": self.participant_id,
            "run_id": self.run_id,
            "trial_id": self.trial_id,
            "model_name": self.model_name,
            "fold_id": self.fold_id,
            "congruency": self.congruency,
            "target_goal": self.target_goal,
            "physical_hand_direction": self.physical_hand_direction,
            "observation_duration_ms": self.observation_duration_ms,
            "forecast_horizon_ms": self.forecast_horizon_ms,
            "endpoint_error_mm": self.endpoint_error_mm,
            "endpoint_error_x_mm": self.endpoint_error_x_mm,
            "endpoint_error_y_mm": self.endpoint_error_y_mm,
            "endpoint_error_z_mm": self.endpoint_error_z_mm,
            "trajectory_error_mm": self.trajectory_error_mm,
            "trajectory_error_x_mm": self.trajectory_error_x_mm,
            "trajectory_error_y_mm": self.trajectory_error_y_mm,
            "trajectory_error_z_mm": self.trajectory_error_z_mm,
            "predicted_end_x": float(self.predicted_endpoint[0]),
            "predicted_end_y": float(self.predicted_endpoint[1]),
            "predicted_end_z": float(self.predicted_endpoint[2]),
            "actual_end_x": float(self.actual_endpoint[0]),
            "actual_end_y": float(self.actual_endpoint[1]),
            "actual_end_z": float(self.actual_endpoint[2]),
        }
