"""Interpretable Intention-Action Conflict Engine for Phase 4."""

from typing import Dict, Optional, Tuple, Any
import numpy as np

from src.phase_4.config import DigitalTwinConfig
from src.phase_4.coordinate_mapping import validate_coordinate_compatibility
from src.phase_4.schemas import (
    ConflictAssessment,
    NeuralIntentionResult,
    TrajectoryForecast,
    TrialMetadata,
)


class ConflictEngine:
    """Computes transparent, explainable discordance between neural intention and kinematic forecast."""

    def __init__(self, config: Optional[DigitalTwinConfig] = None):
        self.config = config or DigitalTwinConfig()
        if self.config.distance_reference_mm <= 0:
            raise ValueError(
                f"Reference distance must be strictly positive, got {self.config.distance_reference_mm}"
            )

    def evaluate_conflict(
        self,
        neural_result: Optional[NeuralIntentionResult],
        forecast_result: Optional[TrajectoryForecast],
        trial_metadata: Optional[TrialMetadata] = None,
        replay_time_ms: float = 0.0,
        initial_position: Optional[np.ndarray] = None,
    ) -> ConflictAssessment:
        """Evaluate intention-action conflict across 4 complementary measurements.
        
        Args:
            neural_result: Pre-movement neural intention decoding result
            forecast_result: Continuous 3D trajectory and endpoint forecast
            trial_metadata: Optional ground truth trial metadata
            replay_time_ms: Current simulation replay clock timestamp
            initial_position: Initial hand position at onset for direction vector (defaults to [0, 150, 50])
        """
        # --- 1. Check Availability and Trial Identity Matching ---
        if neural_result is None or forecast_result is None:
            return ConflictAssessment(
                participant_id=trial_metadata.participant_id if trial_metadata else "unknown",
                trial_id=trial_metadata.trial_id if trial_metadata else "unknown",
                replay_time_ms=replay_time_ms,
                conflict_state="UNAVAILABLE",
                explanation="Conflict scoring unavailable: either neural intention or kinematic forecast is missing.",
                is_conflict=False,
            )

        # Check Trial ID Alignment (Test Case 12)
        if neural_result.trial_id != forecast_result.trial_id:
            return ConflictAssessment(
                participant_id=neural_result.participant_id,
                trial_id=f"{neural_result.trial_id}!={forecast_result.trial_id}",
                replay_time_ms=replay_time_ms,
                conflict_state="INVALID_INPUT",
                explanation=(
                    f"Trial ID mismatch: neural intention trial '{neural_result.trial_id}' "
                    f"differs from kinematic forecast trial '{forecast_result.trial_id}'."
                ),
                is_conflict=False,
            )

        trial_id = neural_result.trial_id
        participant_id = neural_result.participant_id

        # Check Status of Input Models
        if neural_result.status != "AVAILABLE" or forecast_result.status != "AVAILABLE":
            reason = neural_result.error_message or forecast_result.error_message or "Model status not AVAILABLE"
            return ConflictAssessment(
                participant_id=participant_id,
                trial_id=trial_id,
                replay_time_ms=replay_time_ms,
                conflict_state="UNAVAILABLE",
                explanation=f"Inputs not available: {reason}",
                is_conflict=False,
            )

        # Check Finite Coordinates (Test Case 13)
        pred_end = forecast_result.predicted_endpoint
        if pred_end is None or np.isnan(pred_end).any() or np.isinf(pred_end).any():
            return ConflictAssessment(
                participant_id=participant_id,
                trial_id=trial_id,
                replay_time_ms=replay_time_ms,
                conflict_state="INVALID_INPUT",
                explanation="Predicted endpoint contains non-finite (NaN or Inf) coordinates.",
                is_conflict=False,
            )

        # Check Units and Coordinate Compatibility (Test Cases 8 & 9)
        coord_ok, coord_err = validate_coordinate_compatibility(
            frame_a=forecast_result.coordinate_frame,
            frame_b=self.config.coordinate_frame,
            units_a=forecast_result.coordinate_units,
            units_b=self.config.coordinate_units,
        )
        if not coord_ok:
            return ConflictAssessment(
                participant_id=participant_id,
                trial_id=trial_id,
                replay_time_ms=replay_time_ms,
                conflict_state="INVALID_INPUT",
                explanation=coord_err or "Coordinate frame or unit incompatibility.",
                is_conflict=False,
            )

        # Check Target Coordinates Availability (Test Cases 7 & 11)
        target_coords = None
        if trial_metadata is not None and trial_metadata.target_coordinates is not None:
            target_coords = trial_metadata.target_coordinates
        else:
            # Fallback to mapped coordinates from decoded neural intention
            if neural_result.predicted_goal == "left":
                target_coords = np.array(self.config.target_left_coord, dtype=np.float32)
            elif neural_result.predicted_goal == "right":
                target_coords = np.array(self.config.target_right_coord, dtype=np.float32)

        if target_coords is None or np.isnan(target_coords).any():
            return ConflictAssessment(
                participant_id=participant_id,
                trial_id=trial_id,
                replay_time_ms=replay_time_ms,
                conflict_state="UNAVAILABLE",
                explanation="Conflict scoring unavailable: target-coordinate mapping is not verified or target is missing.",
                is_conflict=False,
            )

        # --- 2. Measurement A: Endpoint-to-Target Distance ---
        d_target_mm = float(np.linalg.norm(pred_end - target_coords))
        c_distance_norm = float(min(1.0, d_target_mm / self.config.distance_reference_mm))

        # --- 3. Measurement B: Directional Disagreement Angle ---
        p_origin = initial_position
        if p_origin is None:
            if forecast_result.observed_prefix is not None and len(forecast_result.observed_prefix) > 0:
                p_origin = forecast_result.observed_prefix[0]
            else:
                p_origin = np.array([0.0, 150.0, 50.0], dtype=np.float32)

        v_pred = pred_end - p_origin
        v_target = target_coords - p_origin

        norm_pred = float(np.linalg.norm(v_pred))
        norm_target = float(np.linalg.norm(v_target))

        # Near-zero vector norm handling (Test Case 6)
        if norm_pred < 1e-4 or norm_target < 1e-4:
            theta_deg = 0.0
            is_zero_vector = True
        else:
            is_zero_vector = False
            cos_theta = float(np.dot(v_pred, v_target) / (norm_pred * norm_target))
            cos_theta_clipped = np.clip(cos_theta, -1.0, 1.0)
            theta_deg = float(np.degrees(np.arccos(cos_theta_clipped)))

        # --- 4. Measurement C: Neural Goal Agreement ---
        neural_agreement = None
        if trial_metadata is not None and trial_metadata.target_goal:
            neural_agreement = bool(neural_result.predicted_goal == trial_metadata.target_goal)

        # Motor Forecast Directional Agreement with Physical Hand Direction
        motor_agreement = None
        if trial_metadata is not None and trial_metadata.physical_hand_direction:
            pred_lat_dir = "left" if pred_end[0] < 0 else "right"
            motor_agreement = bool(pred_lat_dir == trial_metadata.physical_hand_direction)

        # --- 5. Measurement D: Retrospective Forecast Error ---
        retrospective_error_mm = None
        if forecast_result.actual_endpoint is not None and not np.isnan(forecast_result.actual_endpoint).any():
            retrospective_error_mm = float(np.linalg.norm(pred_end - forecast_result.actual_endpoint))

        # --- 6. Discrete State Determination ---
        if is_zero_vector:
            conflict_state = "UNCERTAIN"
            explanation = "Directional angle uncertain: movement or target vector has near-zero magnitude."
            is_conflict = False
        elif neural_result.confidence < self.config.confidence_uncertainty_threshold:
            conflict_state = "UNCERTAIN"
            explanation = (
                f"Intention confidence ({neural_result.confidence:.3f}) is below the dependable "
                f"threshold ({self.config.confidence_uncertainty_threshold:.2f})."
            )
            is_conflict = False
        elif (
            d_target_mm > self.config.distance_threshold_mm
            or theta_deg > self.config.directional_threshold_deg
        ):
            conflict_state = "POTENTIAL_DISAGREEMENT"
            is_conflict = True
            explanation = (
                f"Potential disagreement: predicted endpoint is {d_target_mm:.1f} mm from the mapped target "
                f"(exceeds {self.config.distance_threshold_mm:.1f} mm threshold) or directional angle is "
                f"{theta_deg:.1f}° (exceeds {self.config.directional_threshold_deg:.1f}° threshold)."
            )
        else:
            conflict_state = "ALIGNED"
            is_conflict = False
            explanation = (
                f"Aligned: predicted endpoint is {d_target_mm:.1f} mm from target (within "
                f"{self.config.distance_threshold_mm:.1f} mm) and angle is {theta_deg:.1f}°."
            )

        return ConflictAssessment(
            participant_id=participant_id,
            trial_id=trial_id,
            replay_time_ms=replay_time_ms,
            conflict_state=conflict_state,
            endpoint_target_distance_mm=d_target_mm,
            directional_disagreement_deg=theta_deg,
            normalized_distance_score=c_distance_norm,
            neural_goal_agreement=neural_agreement,
            motor_forecast_agreement=motor_agreement,
            retrospective_endpoint_error_mm=retrospective_error_mm,
            explanation=explanation,
            is_conflict=is_conflict,
        )
