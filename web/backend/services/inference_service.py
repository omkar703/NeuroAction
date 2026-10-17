"""Inference service wrapping ReplayEngine and Phase 4/5 forecasting pipelines."""

from typing import List, Optional
import numpy as np

from src.phase_4.replay_engine import ReplayEngine
from web.backend.schemas import (
    ConflictAssessmentSummary,
    ConflictStep,
    ForecastSummary,
    LiveInferenceResponse,
    NeuralSummary,
    TrialDataResponse,
)


class InferenceService:
    """Manages trial simulation preparation, causal frame evaluation, and live inference."""

    def __init__(self, engine: Optional[ReplayEngine] = None):
        self.engine = engine or ReplayEngine()

    def get_trial_data(
        self,
        trial_id: str,
        protocol: str = "kinematic_onset",
    ) -> TrialDataResponse:
        """Assemble full recorded trial timeseries, kinematics, forecasts, and conflict states."""
        trial, meta, neural, forecast = self.engine.prepare_trial_simulation(
            trial_id=trial_id, protocol=protocol
        )

        times_ms = [float(t) for t in trial.times_me]
        hand_x = [float(x) for x in trial.hand_x]
        hand_y = [float(y) for y in trial.hand_y]
        hand_z = [float(z) for z in trial.hand_z]

        # Horizon path extraction (prefer 500ms max horizon)
        horizon_path = None
        if forecast.horizon_trajectories:
            max_h = max(forecast.horizon_trajectories.keys())
            horizon_path = forecast.horizon_trajectories[max_h].tolist()
        elif forecast.predicted_endpoint is not None and forecast.observed_prefix is not None:
            horizon_path = [
                forecast.observed_prefix[-1].tolist(),
                forecast.predicted_endpoint.tolist(),
            ]

        # Neural summary
        neural_summary = NeuralSummary(
            predicted_goal=str(neural.predicted_goal),
            probability_left=float(neural.probability_left),
            probability_right=float(neural.probability_right),
            confidence=float(neural.confidence),
            prediction_time_ms=float(neural.prediction_time_ms),
            model_name=str(neural.model_name),
            status=str(neural.status),
        )

        # Causal conflict evaluation at each recorded timestep
        conflict_timeseries: List[ConflictStep] = []
        for t in times_ms:
            frame = self.engine.get_frame_at_time(
                trial=trial,
                metadata=meta,
                neural_result=neural,
                forecast_result=forecast,
                replay_time_ms=t,
                mode="causal_replay",
            )
            ca = frame.fusion_state.conflict_assessment
            conflict_timeseries.append(
                ConflictStep(
                    time_ms=t,
                    state=ca.conflict_state,
                    dist_mm=float(ca.endpoint_target_distance_mm) if ca.endpoint_target_distance_mm is not None else None,
                    norm_dist=float(ca.normalized_distance_score) if ca.normalized_distance_score is not None else None,
                    explanation=ca.explanation,
                )
            )

        target_coords = meta.target_coordinates.tolist() if meta.target_coordinates is not None else None

        return TrialDataResponse(
            trial_id=trial.trial_id,
            participant_id=trial.participant_id,
            congruency=trial.congruency,
            target_goal=trial.target_goal,
            physical_hand_direction=trial.physical_hand_direction,
            ball_color=trial.ball_color,
            times_ms=times_ms,
            hand_x=hand_x,
            hand_y=hand_y,
            hand_z=hand_z,
            predicted_endpoint=forecast.predicted_endpoint.tolist(),
            forecast_generation_time_ms=float(forecast.forecast_generation_time_ms),
            horizon_path=horizon_path,
            neural=neural_summary,
            conflict_timeseries=conflict_timeseries,
            target_coordinates=target_coords,
            coordinate_units=meta.coordinate_units,
            coordinate_frame=meta.coordinate_frame,
            data_source="recorded_replay",
        )

    def run_live_inference(
        self,
        trial_id: str,
        protocol: str = "kinematic_onset",
    ) -> LiveInferenceResponse:
        """Run real-time inference on pre-movement EEG and early kinematic trajectory prefix."""
        trial, meta, neural, forecast = self.engine.prepare_trial_simulation(
            trial_id=trial_id, protocol=protocol
        )

        # Actual terminal endpoint and error calculation
        actual_endpoint = None
        endpoint_error_mm = None
        if trial.hand_x is not None and len(trial.hand_x) > 0:
            actual_endpoint = [float(trial.hand_x[-1]), float(trial.hand_y[-1]), float(trial.hand_z[-1])]
            endpoint_error_mm = float(
                np.linalg.norm(forecast.predicted_endpoint - np.array(actual_endpoint))
            )

        # Horizon path
        horizon_path = None
        if forecast.horizon_trajectories:
            max_h = max(forecast.horizon_trajectories.keys())
            horizon_path = forecast.horizon_trajectories[max_h].tolist()

        # Evaluate final frame conflict state
        last_t = float(trial.times_me[-1]) if trial.times_me is not None and len(trial.times_me) > 0 else float(forecast.forecast_generation_time_ms)
        final_frame = self.engine.get_frame_at_time(
            trial=trial,
            metadata=meta,
            neural_result=neural,
            forecast_result=forecast,
            replay_time_ms=last_t,
            mode="causal_replay",
        )
        ca = final_frame.fusion_state.conflict_assessment

        neural_summary = NeuralSummary(
            predicted_goal=str(neural.predicted_goal),
            probability_left=float(neural.probability_left),
            probability_right=float(neural.probability_right),
            confidence=float(neural.confidence),
            prediction_time_ms=float(neural.prediction_time_ms),
            model_name=str(neural.model_name),
            status=str(neural.status),
        )

        forecast_summary = ForecastSummary(
            model_name=forecast.model_name,
            forecast_protocol=forecast.forecast_protocol,
            forecast_generation_time_ms=float(forecast.forecast_generation_time_ms),
            predicted_endpoint=forecast.predicted_endpoint.tolist(),
            actual_endpoint=actual_endpoint,
            endpoint_error_mm=endpoint_error_mm,
            horizon_path=horizon_path,
            status=forecast.status,
        )

        conflict_summary = ConflictAssessmentSummary(
            conflict_state=ca.conflict_state,
            endpoint_target_distance_mm=float(ca.endpoint_target_distance_mm) if ca.endpoint_target_distance_mm is not None else None,
            directional_disagreement_deg=float(ca.directional_disagreement_deg) if ca.directional_disagreement_deg is not None else None,
            normalized_distance_score=float(ca.normalized_distance_score) if ca.normalized_distance_score is not None else None,
            neural_goal_agreement=ca.neural_goal_agreement,
            motor_forecast_agreement=ca.motor_forecast_agreement,
            retrospective_endpoint_error_mm=float(ca.retrospective_endpoint_error_mm) if ca.retrospective_endpoint_error_mm is not None else None,
            explanation=ca.explanation,
            is_conflict=ca.is_conflict,
        )

        return LiveInferenceResponse(
            trial_id=trial.trial_id,
            participant_id=trial.participant_id,
            protocol=protocol,
            neural=neural_summary,
            trajectory_forecast=forecast_summary,
            conflict_assessment=conflict_summary,
            data_source="live_inference",
        )


# Global singleton instance
_inference_service: Optional[InferenceService] = None


def get_inference_service() -> InferenceService:
    """Dependency injection provider for InferenceService."""
    global _inference_service
    if _inference_service is None:
        _inference_service = InferenceService()
    return _inference_service
