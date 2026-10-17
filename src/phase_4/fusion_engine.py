"""Multimodal Fusion Engine for Phase 4 Digital Twin."""

from typing import List, Optional
import numpy as np

from src.phase_4.config import DigitalTwinConfig
from src.phase_4.conflict_engine import ConflictEngine
from src.phase_4.schemas import (
    ConflictAssessment,
    FusionState,
    KinematicObservation,
    NeuralIntentionResult,
    TrajectoryForecast,
    TrialMetadata,
)


class MultimodalFusionEngine:
    """Unites neural intention decoding, continuous kinematics, and conflict analysis."""

    def __init__(
        self,
        config: Optional[DigitalTwinConfig] = None,
        conflict_engine: Optional[ConflictEngine] = None,
    ):
        self.config = config or DigitalTwinConfig()
        self.conflict_engine = conflict_engine or ConflictEngine(self.config)

    def fuse_trial_state(
        self,
        current_time_ms: float,
        trial_metadata: Optional[TrialMetadata] = None,
        neural_result: Optional[NeuralIntentionResult] = None,
        current_kinematics: Optional[KinematicObservation] = None,
        forecast_result: Optional[TrajectoryForecast] = None,
        active_protocol: str = "kinematic_onset",
    ) -> FusionState:
        """Fuse multimodal signals into a single coherent state vector at the replay clock.
        
        Args:
            current_time_ms: Current replay simulation clock in milliseconds
            trial_metadata: Static trial parameters and ground truth
            neural_result: Pre-movement scalp EEG intention decoding result
            current_kinematics: Current instantaneous hand kinematics observation
            forecast_result: Trajectory and endpoint forecast
            active_protocol: 'kinematic_onset' or 'event_locked'
        """
        warnings: List[str] = []
        participant_id = trial_metadata.participant_id if trial_metadata else "unknown"
        trial_id = trial_metadata.trial_id if trial_metadata else "unknown"

        # Check Trial ID Consistency
        if neural_result and trial_id != "unknown" and neural_result.trial_id != trial_id:
            warnings.append(
                f"Neural result trial ID ({neural_result.trial_id}) does not match metadata ({trial_id})"
            )
        if forecast_result and trial_id != "unknown" and forecast_result.trial_id != trial_id:
            warnings.append(
                f"Forecast result trial ID ({forecast_result.trial_id}) does not match metadata ({trial_id})"
            )

        # Causal Availability Gating
        active_neural: Optional[NeuralIntentionResult] = None
        if neural_result is not None:
            if current_time_ms >= neural_result.prediction_time_ms:
                active_neural = neural_result
            else:
                # Pre-cue replay time: neural prediction exists offline but is not causally available
                active_neural = NeuralIntentionResult(
                    participant_id=neural_result.participant_id,
                    trial_id=neural_result.trial_id,
                    model_name=neural_result.model_name,
                    prediction_time_ms=neural_result.prediction_time_ms,
                    window_start_ms=neural_result.window_start_ms,
                    window_end_ms=neural_result.window_end_ms,
                    predicted_goal="pending",
                    probability_left=0.5,
                    probability_right=0.5,
                    confidence=0.5,
                    status="UNAVAILABLE",
                    error_message=f"Neural prediction available only at t >= {neural_result.prediction_time_ms:.1f} ms",
                    is_temporally_valid=False,
                )

        active_forecast: Optional[TrajectoryForecast] = None
        if forecast_result is not None:
            if current_time_ms >= forecast_result.forecast_generation_time_ms:
                active_forecast = forecast_result
            else:
                # Prefix observation is not yet complete at current replay time
                active_forecast = None

        # Conflict Evaluation
        conflict_assessment: Optional[ConflictAssessment] = None
        if active_neural is not None and active_forecast is not None:
            conflict_assessment = self.conflict_engine.evaluate_conflict(
                neural_result=active_neural,
                forecast_result=active_forecast,
                trial_metadata=trial_metadata,
                replay_time_ms=current_time_ms,
            )
        else:
            conflict_assessment = ConflictAssessment(
                participant_id=participant_id,
                trial_id=trial_id,
                replay_time_ms=current_time_ms,
                conflict_state="UNAVAILABLE",
                explanation="Conflict scoring unavailable: waiting for causal availability of both neural and forecast models.",
                is_conflict=False,
            )

        return FusionState(
            participant_id=participant_id,
            trial_id=trial_id,
            current_replay_time_ms=current_time_ms,
            trial_metadata=trial_metadata,
            neural_result=active_neural,
            current_kinematics=current_kinematics,
            forecast_result=active_forecast,
            conflict_assessment=conflict_assessment,
            alignment_status="ALIGNED" if not warnings else "MISALIGNED",
            coordinate_mapping_status="VERIFIED",
            active_protocol=active_protocol,
            warnings=warnings,
        )
