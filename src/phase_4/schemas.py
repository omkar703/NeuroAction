"""Typed data contracts and schemas for Phase 4 Multimodal Digital Twin."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import numpy as np


@dataclass
class TrialMetadata:
    """Canonical trial metadata capturing ground-truth parameters and integrity flags."""

    participant_id: str
    run_id: str
    trial_id: str
    trial_index: int
    label_code: int
    congruency: str                 # "congruent" or "incongruent"
    target_goal: str                # "left" or "right"
    physical_hand_direction: str    # "left" or "right"
    ball_color: str                 # "red" or "yellow"
    valid: bool
    exclusion_reason: Optional[str] = None
    go_cue_time_ms: float = 0.0
    kinematic_onset_time_ms: Optional[float] = None
    movement_duration_ms: Optional[float] = None
    target_coordinates: Optional[np.ndarray] = None  # shape (3,) [x, y, z] mm
    coordinate_units: str = "mm"
    coordinate_frame: str = "leap_motion_cartesian"
    eeg_sample_range_ms: Tuple[float, float] = (-1000.0, 0.0)
    kinematic_sample_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            "participant_id": self.participant_id,
            "run_id": self.run_id,
            "trial_id": self.trial_id,
            "trial_index": self.trial_index,
            "label_code": self.label_code,
            "congruency": self.congruency,
            "target_goal": self.target_goal,
            "physical_hand_direction": self.physical_hand_direction,
            "ball_color": self.ball_color,
            "valid": self.valid,
            "exclusion_reason": self.exclusion_reason,
            "go_cue_time_ms": self.go_cue_time_ms,
            "kinematic_onset_time_ms": self.kinematic_onset_time_ms,
            "movement_duration_ms": self.movement_duration_ms,
            "target_coordinates": self.target_coordinates.tolist() if self.target_coordinates is not None else None,
            "coordinate_units": self.coordinate_units,
            "coordinate_frame": self.coordinate_frame,
            "eeg_sample_range_ms": list(self.eeg_sample_range_ms),
            "kinematic_sample_count": self.kinematic_sample_count,
        }


@dataclass
class NeuralIntentionResult:
    """Pre-movement scalp EEG intention classification output."""

    participant_id: str
    trial_id: str
    model_name: str
    prediction_time_ms: float       # Visual go-cue (e.g. 0.0 ms)
    window_start_ms: float          # -1000.0 ms
    window_end_ms: float            # 0.0 ms
    predicted_goal: str             # "left" or "right"
    probability_left: float
    probability_right: float
    confidence: float               # max(P(left), P(right))
    score_type: str = "posterior_probability"
    is_calibrated: bool = False
    status: str = "AVAILABLE"       # "AVAILABLE", "UNAVAILABLE", "ERROR"
    error_message: Optional[str] = None
    is_temporally_valid: bool = True
    processing_mode: str = "offline_acausal"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "participant_id": self.participant_id,
            "trial_id": self.trial_id,
            "model_name": self.model_name,
            "prediction_time_ms": self.prediction_time_ms,
            "window_start_ms": self.window_start_ms,
            "window_end_ms": self.window_end_ms,
            "predicted_goal": self.predicted_goal,
            "probability_left": float(self.probability_left),
            "probability_right": float(self.probability_right),
            "confidence": float(self.confidence),
            "score_type": self.score_type,
            "is_calibrated": self.is_calibrated,
            "status": self.status,
            "error_message": self.error_message,
            "is_temporally_valid": self.is_temporally_valid,
            "processing_mode": self.processing_mode,
        }


@dataclass
class KinematicObservation:
    """Single-timestamp 3D hand observation for causal state rendering."""

    participant_id: str
    trial_id: str
    timestamp_ms: float
    position: np.ndarray            # shape (3,) [x, y, z] mm
    velocity: Optional[np.ndarray] = None  # shape (3,) mm/s
    speed: Optional[float] = None   # mm/s
    coordinate_units: str = "mm"
    coordinate_frame: str = "leap_motion_cartesian"
    is_valid: bool = True
    is_missing: bool = False
    available_at_clock: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "participant_id": self.participant_id,
            "trial_id": self.trial_id,
            "timestamp_ms": self.timestamp_ms,
            "position": self.position.tolist() if self.position is not None else None,
            "velocity": self.velocity.tolist() if self.velocity is not None else None,
            "speed": float(self.speed) if self.speed is not None else None,
            "coordinate_units": self.coordinate_units,
            "coordinate_frame": self.coordinate_frame,
            "is_valid": self.is_valid,
            "is_missing": self.is_missing,
            "available_at_clock": self.available_at_clock,
        }


@dataclass
class TrajectoryForecast:
    """Continuous 3D kinematic trajectory and endpoint forecast."""

    participant_id: str
    trial_id: str
    model_name: str
    forecast_protocol: str          # "event_locked" or "kinematic_onset"
    forecast_generation_time_ms: float
    prefix_start_time_ms: float
    prefix_end_time_ms: float
    observed_prefix: np.ndarray     # shape (20, 3) mm
    predicted_endpoint: np.ndarray  # shape (3,) mm
    actual_endpoint: Optional[np.ndarray] = None  # shape (3,) mm
    horizon_trajectories: Dict[int, np.ndarray] = field(default_factory=dict)
    horizon_points: Dict[int, np.ndarray] = field(default_factory=dict)
    coordinate_units: str = "mm"
    coordinate_frame: str = "leap_motion_cartesian"
    is_held_out: bool = False
    held_out_participant: str = ""
    status: str = "AVAILABLE"       # "AVAILABLE", "UNAVAILABLE", "ERROR"
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "participant_id": self.participant_id,
            "trial_id": self.trial_id,
            "model_name": self.model_name,
            "forecast_protocol": self.forecast_protocol,
            "forecast_generation_time_ms": self.forecast_generation_time_ms,
            "prefix_start_time_ms": self.prefix_start_time_ms,
            "prefix_end_time_ms": self.prefix_end_time_ms,
            "observed_prefix": self.observed_prefix.tolist() if self.observed_prefix is not None else None,
            "predicted_endpoint": self.predicted_endpoint.tolist() if self.predicted_endpoint is not None else None,
            "actual_endpoint": self.actual_endpoint.tolist() if self.actual_endpoint is not None else None,
            "coordinate_units": self.coordinate_units,
            "coordinate_frame": self.coordinate_frame,
            "is_held_out": self.is_held_out,
            "held_out_participant": self.held_out_participant,
            "status": self.status,
            "error_message": self.error_message,
        }


@dataclass
class ConflictAssessment:
    """Interpretable intention-action discordance measurement and state."""

    participant_id: str
    trial_id: str
    replay_time_ms: float
    conflict_state: str             # "ALIGNED", "POTENTIAL_DISAGREEMENT", "UNCERTAIN", "UNAVAILABLE", "INVALID_INPUT"
    endpoint_target_distance_mm: Optional[float] = None
    directional_disagreement_deg: Optional[float] = None
    normalized_distance_score: Optional[float] = None
    neural_goal_agreement: Optional[bool] = None     # Decoded goal vs Instructed target goal
    motor_forecast_agreement: Optional[bool] = None  # Predicted endpoint lateral sign vs Physical hand direction
    retrospective_endpoint_error_mm: Optional[float] = None
    explanation: str = ""
    is_conflict: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "participant_id": self.participant_id,
            "trial_id": self.trial_id,
            "replay_time_ms": self.replay_time_ms,
            "conflict_state": self.conflict_state,
            "endpoint_target_distance_mm": self.endpoint_target_distance_mm,
            "directional_disagreement_deg": self.directional_disagreement_deg,
            "normalized_distance_score": self.normalized_distance_score,
            "neural_goal_agreement": self.neural_goal_agreement,
            "motor_forecast_agreement": self.motor_forecast_agreement,
            "retrospective_endpoint_error_mm": self.retrospective_endpoint_error_mm,
            "explanation": self.explanation,
            "is_conflict": self.is_conflict,
        }


@dataclass
class FusionState:
    """Unified multimodal state vector uniting neural, kinematic, and conflict layers."""

    participant_id: str
    trial_id: str
    current_replay_time_ms: float
    trial_metadata: Optional[TrialMetadata] = None
    neural_result: Optional[NeuralIntentionResult] = None
    current_kinematics: Optional[KinematicObservation] = None
    forecast_result: Optional[TrajectoryForecast] = None
    conflict_assessment: Optional[ConflictAssessment] = None
    alignment_status: str = "ALIGNED"
    coordinate_mapping_status: str = "VERIFIED"
    active_protocol: str = "kinematic_onset"
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "participant_id": self.participant_id,
            "trial_id": self.trial_id,
            "current_replay_time_ms": self.current_replay_time_ms,
            "trial_metadata": self.trial_metadata.to_dict() if self.trial_metadata else None,
            "neural_result": self.neural_result.to_dict() if self.neural_result else None,
            "current_kinematics": self.current_kinematics.to_dict() if self.current_kinematics else None,
            "forecast_result": self.forecast_result.to_dict() if self.forecast_result else None,
            "conflict_assessment": self.conflict_assessment.to_dict() if self.conflict_assessment else None,
            "alignment_status": self.alignment_status,
            "coordinate_mapping_status": self.coordinate_mapping_status,
            "active_protocol": self.active_protocol,
            "warnings": self.warnings,
        }


@dataclass
class ReplayFrame:
    """Discrete simulation timestep frame yielded by the ReplayEngine."""

    frame_index: int
    timestamp_ms: float
    fusion_state: FusionState
    observed_path_so_far: np.ndarray  # shape (N, 3) mm
    forecast_path: Optional[np.ndarray] = None  # shape (M, 3) mm
    target_pos: Optional[np.ndarray] = None     # shape (3,) mm
    actual_future_path: Optional[np.ndarray] = None  # shape (K, 3) mm (only for retrospective mode)
