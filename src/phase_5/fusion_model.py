"""Interpretable multimodal fusion model combining pre-movement EEG with kinematic trajectory forecast."""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np

from src.phase_4.schemas import NeuralIntentionResult, TrajectoryForecast
from src.phase_5.schemas import TargetReference, Phase5Prediction
from src.phase_5.prediction_policy import PredictionPolicy, DecisionPolicyState


class MultimodalOutcomeFusionModel:
    """Interpretable fusion combining neural goal classification and kinematic destination forecasting."""

    def __init__(
        self,
        policy: Optional[PredictionPolicy] = None,
        neural_weight: float = 0.35,
        kinematic_weight: float = 0.65,
    ):
        self.policy = policy or PredictionPolicy()
        self.neural_weight = neural_weight
        self.kinematic_weight = kinematic_weight

    def predict(
        self,
        trial_id: str,
        participant_id: str,
        neural_result: NeuralIntentionResult,
        forecast_result: TrajectoryForecast,
        target_reference: TargetReference,
        fold_id: str,
        training_participants: List[str],
    ) -> Phase5Prediction:
        """Generate a prospective prediction using strictly information available at t_prefix_end."""
        pred_end = forecast_result.predicted_endpoint
        target_coords = target_reference.coordinates

        # 1. Kinematic Target Distance & Directional Deviation
        d_target = float(np.linalg.norm(pred_end - target_coords))
        d_norm = min(d_target / (2.0 * target_reference.tolerance_mm), 1.0)

        # Directional angle between predicted trajectory vector and target vector
        p_prefix_start = forecast_result.observed_prefix[0]
        p_prefix_end = forecast_result.observed_prefix[-1]
        v_pred = pred_end - p_prefix_end
        v_target = target_coords - p_prefix_end

        norm_pred = np.linalg.norm(v_pred)
        norm_target = np.linalg.norm(v_target)
        if norm_pred > 1e-4 and norm_target > 1e-4:
            cos_theta = np.clip(np.dot(v_pred, v_target) / (norm_pred * norm_target), -1.0, 1.0)
            angle_deg = float(np.degrees(np.arccos(cos_theta)))
        else:
            angle_deg = 0.0

        # 2. Neural Goal Compatibility
        # If target is "left", target_prob is P(left); if "right", P(right)
        if target_reference.target_id == "left":
            target_prob = neural_result.probability_left
        elif target_reference.target_id == "right":
            target_prob = neural_result.probability_right
        else:
            target_prob = neural_result.confidence

        neural_discordance = 1.0 - target_prob

        # 3. Continuous Fused Score for AUROC
        # 0 = highly aligned with target (Success), 1 = highly discordant (Failure)
        continuous_score = (
            self.kinematic_weight * d_norm + self.neural_weight * neural_discordance
        )
        continuous_score = float(np.clip(continuous_score, 0.0, 1.0))

        # 4. Decision Policy & Abstention Evaluation
        decision_state = self.policy.evaluate_target_outcome(
            distance_to_target_mm=d_target,
            confidence_score=neural_result.confidence,
            is_input_valid=True,
            has_target_reference=(target_reference is not None),
        )

        is_abstention = self.policy.is_abstention(decision_state)
        is_coverage = self.policy.is_coverage_decision(decision_state)

        t_pred = float(forecast_result.forecast_generation_time_ms)
        t_prefix_start = float(forecast_result.prefix_start_time_ms)
        t_prefix_end = float(forecast_result.prefix_end_time_ms)

        return Phase5Prediction(
            trial_id=trial_id,
            participant_id=participant_id,
            prediction_time_ms=t_pred,
            input_window_ms=(t_prefix_start, t_prefix_end),
            forecast_model_id=forecast_result.model_name,
            neural_model_id=neural_result.model_name,
            fusion_model_id="multimodal_outcome_fusion_v1",
            fold_id=fold_id,
            training_participants=training_participants,
            neural_goal_pred=neural_result.predicted_goal,
            neural_confidence=neural_result.confidence,
            pred_endpoint=pred_end,
            distance_to_target_mm=d_target,
            directional_angle_deg=angle_deg,
            continuous_score=continuous_score,
            decision_policy_state=decision_state,
            is_abstention=is_abstention,
            is_coverage=is_coverage,
            calibration_status="uncalibrated",
        )
