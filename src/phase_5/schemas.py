"""Typed data contracts and schemas for Phase 5: Independent Scientific Validation."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import numpy as np


@dataclass
class TargetReference:
    """Explicitly specified physical action target reference."""

    target_id: str                      # e.g. "left", "right", "target_01"
    coordinates: np.ndarray             # shape (3,) [x, y, z] mm
    tolerance_mm: float = 50.0
    geometry_type: str = "centroid_proxy"  # "authoritative_sphere", "authoritative_box", "centroid_proxy"
    coordinate_frame: str = "leap_motion_cartesian"
    coordinate_units: str = "mm"
    is_authoritative: bool = False       # True ONLY if measured independently from physical setup
    provenance: str = "training_subs_proxy"
    description: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "target_id": self.target_id,
            "coordinates": self.coordinates.tolist() if self.coordinates is not None else None,
            "tolerance_mm": float(self.tolerance_mm),
            "geometry_type": self.geometry_type,
            "coordinate_frame": self.coordinate_frame,
            "coordinate_units": self.coordinate_units,
            "is_authoritative": bool(self.is_authoritative),
            "provenance": self.provenance,
            "description": self.description,
        }


@dataclass
class OutcomeLabel:
    """Measurable physical target-reaching outcome ground truth."""

    trial_id: str
    target_id: str
    actual_endpoint: np.ndarray         # shape (3,) [x, y, z] mm
    distance_to_target_mm: float
    target_success: bool                # True if endpoint inside target tolerance
    target_failure: bool                # True if endpoint outside target tolerance
    status: str = "VALID"               # "VALID", "UNAVAILABLE", "AMBIGUOUS"
    is_authoritative: bool = False       # Inherited from TargetReference
    provenance: str = "retrospective_endpoint_measurement"
    exclusion_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trial_id": self.trial_id,
            "target_id": self.target_id,
            "actual_endpoint": self.actual_endpoint.tolist() if self.actual_endpoint is not None else None,
            "distance_to_target_mm": float(self.distance_to_target_mm),
            "target_success": bool(self.target_success),
            "target_failure": bool(self.target_failure),
            "status": self.status,
            "is_authoritative": bool(self.is_authoritative),
            "provenance": self.provenance,
            "exclusion_reason": self.exclusion_reason,
        }


@dataclass
class EligibilityRecord:
    """Machine-readable trial eligibility record across Phase 5 tasks."""

    trial_id: str
    participant_id: str
    run_id: str
    has_valid_pre_eeg: bool
    has_valid_prefix: bool
    has_valid_endpoint: bool
    has_independent_target_coordinates: bool
    has_independent_outcome_label: bool
    is_authoritative_ground_truth: bool

    # Task Eligibility
    eligible_task_a_neural: bool            # Neural stimulus target decoding
    eligible_task_b_kinematic_forecast: bool # Kinematic trajectory forecasting
    eligible_task_c_target_forecast: bool   # Target-success forecast (exploratory proxy or auth)
    eligible_task_d_neural_outcome: bool    # Neural-only outcome prediction
    eligible_task_e_kinematic_outcome: bool # Kinematic-only outcome prediction
    eligible_task_f_fusion_outcome: bool    # Multimodal fusion outcome prediction

    unsupported_reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trial_id": self.trial_id,
            "participant_id": self.participant_id,
            "run_id": self.run_id,
            "has_valid_pre_eeg": self.has_valid_pre_eeg,
            "has_valid_prefix": self.has_valid_prefix,
            "has_valid_endpoint": self.has_valid_endpoint,
            "has_independent_target_coordinates": self.has_independent_target_coordinates,
            "has_independent_outcome_label": self.has_independent_outcome_label,
            "is_authoritative_ground_truth": self.is_authoritative_ground_truth,
            "eligible_task_a_neural": self.eligible_task_a_neural,
            "eligible_task_b_kinematic_forecast": self.eligible_task_b_kinematic_forecast,
            "eligible_task_c_target_forecast": self.eligible_task_c_target_forecast,
            "eligible_task_d_neural_outcome": self.eligible_task_d_neural_outcome,
            "eligible_task_e_kinematic_outcome": self.eligible_task_e_kinematic_outcome,
            "eligible_task_f_fusion_outcome": self.eligible_task_f_fusion_outcome,
            "unsupported_reasons": self.unsupported_reasons,
        }


@dataclass
class Phase5Prediction:
    """Prospective prediction record capturing strictly available information."""

    trial_id: str
    participant_id: str
    prediction_time_ms: float           # Timestamp when forecast was generated (t_prefix_end)
    input_window_ms: Tuple[float, float]# (t_onset, t_onset + 200 ms)
    forecast_model_id: str
    neural_model_id: str
    fusion_model_id: str
    fold_id: str                        # e.g. "held_out_sub01"
    training_participants: List[str]

    # Model Outputs
    neural_goal_pred: str
    neural_confidence: float
    pred_endpoint: np.ndarray           # shape (3,) [x, y, z] mm
    distance_to_target_mm: float
    directional_angle_deg: float
    continuous_score: float             # Continuous score for AUROC (e.g. c_norm or probability)

    # Decision Policy
    decision_policy_state: str          # PREDICTED_SUCCESS, PREDICTED_FAILURE, UNCERTAIN, UNAVAILABLE, INVALID_INPUT
    is_abstention: bool
    is_coverage: bool                   # True if a definite decision (PREDICTED_SUCCESS or PREDICTED_FAILURE) was made
    calibration_status: str = "uncalibrated"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trial_id": self.trial_id,
            "participant_id": self.participant_id,
            "prediction_time_ms": float(self.prediction_time_ms),
            "input_window_ms": list(self.input_window_ms),
            "forecast_model_id": self.forecast_model_id,
            "neural_model_id": self.neural_model_id,
            "fusion_model_id": self.fusion_model_id,
            "fold_id": self.fold_id,
            "training_participants": self.training_participants,
            "neural_goal_pred": self.neural_goal_pred,
            "neural_confidence": float(self.neural_confidence),
            "pred_endpoint": self.pred_endpoint.tolist() if self.pred_endpoint is not None else None,
            "distance_to_target_mm": float(self.distance_to_target_mm),
            "directional_angle_deg": float(self.directional_angle_deg),
            "continuous_score": float(self.continuous_score),
            "decision_policy_state": self.decision_policy_state,
            "is_abstention": bool(self.is_abstention),
            "is_coverage": bool(self.is_coverage),
            "calibration_status": self.calibration_status,
        }


@dataclass
class Phase5EvaluationRow:
    """Unified evaluation row pairing prospective prediction with retrospective ground truth."""

    trial_id: str
    participant_id: str
    run_id: str
    congruency: str                     # "congruent" or "incongruent"
    target_goal: str                    # "left" or "right" (instructed stimulus location)
    physical_hand_direction: str        # "left" or "right" (physical movement direction)
    is_held_out: bool
    fold_id: str

    # Ground Truth References
    target_reference_provenance: str
    is_target_authoritative: bool
    actual_endpoint_x: float
    actual_endpoint_y: float
    actual_endpoint_z: float
    true_target_success: bool
    true_target_failure: bool
    outcome_status: str

    # Prospective Predictions
    prediction_time_ms: float
    neural_goal_pred: str
    neural_confidence: float
    pred_endpoint_x: float
    pred_endpoint_y: float
    pred_endpoint_z: float
    endpoint_error_mm: float
    pred_distance_to_target_mm: float
    decision_policy_state: str
    is_abstention: bool
    continuous_score: float

    # Outcome Prediction Agreement
    predicted_success: bool
    outcome_agreement: bool             # predicted_success == true_target_success
    evaluation_eligibility: str         # "ELIGIBLE_PROXY", "ELIGIBLE_AUTHORITATIVE", "EXCLUDED"


@dataclass
class TaskMetricsSummary:
    """Statistical evaluation metrics for a specific prediction task."""

    task_name: str
    total_trials: int
    eligible_trials: int
    evaluated_trials: int
    abstained_trials: int
    coverage_rate: float
    balanced_accuracy: float
    sensitivity: float
    specificity: float
    precision: float
    f1_score: float
    auroc: float
    brier_score: Optional[float] = None
    endpoint_rmse_mm: Optional[float] = None
    mean_endpoint_error_mm: Optional[float] = None
    is_authoritative: bool = False
    evidence_level: str = "Level 2 (Existing-Data Exploratory)"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_name": self.task_name,
            "total_trials": self.total_trials,
            "eligible_trials": self.eligible_trials,
            "evaluated_trials": self.evaluated_trials,
            "abstained_trials": self.abstained_trials,
            "coverage_rate": float(self.coverage_rate),
            "balanced_accuracy": float(self.balanced_accuracy),
            "sensitivity": float(self.sensitivity),
            "specificity": float(self.specificity),
            "precision": float(self.precision),
            "f1_score": float(self.f1_score),
            "auroc": float(self.auroc),
            "brier_score": float(self.brier_score) if self.brier_score is not None else None,
            "endpoint_rmse_mm": float(self.endpoint_rmse_mm) if self.endpoint_rmse_mm is not None else None,
            "mean_endpoint_error_mm": float(self.mean_endpoint_error_mm) if self.mean_endpoint_error_mm is not None else None,
            "is_authoritative": bool(self.is_authoritative),
            "evidence_level": self.evidence_level,
        }
