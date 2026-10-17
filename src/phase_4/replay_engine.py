"""Deterministic Trial Replay and Simulation Engine for Phase 4 Digital Twin."""

from typing import Generator, List, Optional, Tuple
import numpy as np

from src.data.schema import CanonicalTrial
from src.phase_4.config import DigitalTwinConfig
from src.phase_4.coordinate_mapping import get_target_coordinates
from src.phase_4.fusion_engine import MultimodalFusionEngine
from src.phase_4.kinematic_adapter import KinematicTwinAdapter
from src.phase_4.neural_adapter import NeuralIntentionAdapter
from src.phase_4.schemas import (
    FusionState,
    KinematicObservation,
    NeuralIntentionResult,
    ReplayFrame,
    TrajectoryForecast,
    TrialMetadata,
)
from src.phase_4.trial_registry import TrialRegistry


class ReplayEngine:
    """Simulates real-world trial execution advancing causally through discrete timesteps."""

    def __init__(
        self,
        registry: Optional[TrialRegistry] = None,
        neural_adapter: Optional[NeuralIntentionAdapter] = None,
        kinematic_adapter: Optional[KinematicTwinAdapter] = None,
        fusion_engine: Optional[MultimodalFusionEngine] = None,
        config: Optional[DigitalTwinConfig] = None,
    ):
        self.config = config or DigitalTwinConfig()
        self.registry = registry or TrialRegistry(manifest_path=self.config.manifest_path)
        self.neural_adapter = neural_adapter or NeuralIntentionAdapter.load(
            model_dir=self.config.model_dir_phase_2
        )
        self.kinematic_adapter = kinematic_adapter or KinematicTwinAdapter.load(
            model_dir=self.config.model_dir_phase_3
        )
        self.fusion_engine = fusion_engine or MultimodalFusionEngine(config=self.config)

    def prepare_trial_simulation(
        self,
        trial_id: str,
        protocol: str = "kinematic_onset",
    ) -> Tuple[CanonicalTrial, TrialMetadata, NeuralIntentionResult, TrajectoryForecast]:
        """Load and run baseline models for a trial once, caching outputs for deterministic replay."""
        trial = self.registry.get_trial(trial_id)
        if trial is None:
            raise ValueError(f"Trial '{trial_id}' not found in registry.")

        metadata = self.registry.get_metadata(trial_id)
        if metadata is None:
            raise ValueError(f"Metadata could not be constructed for trial '{trial_id}'.")

        # Select held-out or cohort models based on participant ID
        if trial.participant_id == "sub-01":
            n_adapter = NeuralIntentionAdapter.load_held_out(
                participant_id=trial.participant_id,
                model_dir=self.config.model_dir_phase_2,
            )
            k_adapter = KinematicTwinAdapter.load_held_out(
                participant_id=trial.participant_id,
                model_dir=self.config.model_dir_phase_3,
            )
        else:
            n_adapter = self.neural_adapter
            k_adapter = self.kinematic_adapter

        neural_result = n_adapter.predict_trial(trial=trial, current_time_ms=0.0)
        forecast_result = k_adapter.forecast_trial(
            trial=trial,
            protocol=protocol,
            speed_threshold_mm_s=self.config.speed_threshold_mm_s,
            horizons_ms=self.config.horizons_ms,
        )

        return trial, metadata, neural_result, forecast_result

    def get_frame_at_time(
        self,
        trial: CanonicalTrial,
        metadata: TrialMetadata,
        neural_result: NeuralIntentionResult,
        forecast_result: TrajectoryForecast,
        replay_time_ms: float,
        mode: str = "causal_replay",  # "causal_replay" or "retrospective"
    ) -> ReplayFrame:
        """Construct a ReplayFrame strictly using information available at replay_time_ms."""
        times = trial.times_me
        if times is None or len(times) == 0:
            raise ValueError(f"Trial {trial.trial_id} contains no kinematic timestamps.")

        # Find hand samples available up to replay_time_ms
        valid_indices = np.where(times <= replay_time_ms)[0]
        if len(valid_indices) == 0:
            idx_curr = 0
            observed_path = np.stack([trial.hand_x[:1], trial.hand_y[:1], trial.hand_z[:1]], axis=1)
        else:
            idx_curr = valid_indices[-1]
            observed_path = np.stack([
                trial.hand_x[: idx_curr + 1],
                trial.hand_y[: idx_curr + 1],
                trial.hand_z[: idx_curr + 1],
            ], axis=1)

        # Instantaneous Hand Kinematics Observation
        curr_pos = np.array([trial.hand_x[idx_curr], trial.hand_y[idx_curr], trial.hand_z[idx_curr]], dtype=np.float32)
        if idx_curr > 0:
            dt_s = (times[idx_curr] - times[idx_curr - 1]) / 1000.0
            dt_s = max(dt_s, 1e-4)
            prev_pos = np.array([trial.hand_x[idx_curr - 1], trial.hand_y[idx_curr - 1], trial.hand_z[idx_curr - 1]], dtype=np.float32)
            curr_vel = (curr_pos - prev_pos) / dt_s
            curr_speed = float(np.linalg.norm(curr_vel))
        else:
            curr_vel = np.zeros(3, dtype=np.float32)
            curr_speed = 0.0

        current_obs = KinematicObservation(
            participant_id=trial.participant_id,
            trial_id=trial.trial_id,
            timestamp_ms=float(times[idx_curr]),
            position=curr_pos,
            velocity=curr_vel,
            speed=curr_speed,
            coordinate_units="mm",
            coordinate_frame="leap_motion_cartesian",
            is_valid=not np.isnan(curr_pos).any(),
            available_at_clock=True,
        )

        # Multimodal State Fusion
        fusion_state = self.fusion_engine.fuse_trial_state(
            current_time_ms=replay_time_ms,
            trial_metadata=metadata,
            neural_result=neural_result,
            current_kinematics=current_obs,
            forecast_result=forecast_result,
            active_protocol=forecast_result.forecast_protocol,
        )

        # Forecast Path (only rendered if forecast is causally available)
        forecast_path = None
        if fusion_state.forecast_result is not None:
            # Use maximum available horizon trajectory (e.g. 500 ms)
            max_h = max(fusion_state.forecast_result.horizon_trajectories.keys()) if fusion_state.forecast_result.horizon_trajectories else None
            if max_h is not None:
                forecast_path = fusion_state.forecast_result.horizon_trajectories[max_h]
            else:
                forecast_path = np.stack([
                    fusion_state.forecast_result.observed_prefix[-1],
                    fusion_state.forecast_result.predicted_endpoint,
                ], axis=0)

        # Target Coordinates
        target_pos = metadata.target_coordinates

        # Future Ground Truth (strictly None in causal mode!)
        actual_future = None
        if mode == "retrospective":
            actual_future = np.stack([
                trial.hand_x[idx_curr:],
                trial.hand_y[idx_curr:],
                trial.hand_z[idx_curr:],
            ], axis=1)

        return ReplayFrame(
            frame_index=int(idx_curr),
            timestamp_ms=float(times[idx_curr]),
            fusion_state=fusion_state,
            observed_path_so_far=observed_path,
            forecast_path=forecast_path,
            target_pos=target_pos,
            actual_future_path=actual_future,
        )

    def stream_trial(
        self,
        trial_id: str,
        protocol: str = "kinematic_onset",
        mode: str = "causal_replay",
        step_sample: int = 2,  # Step every 20 ms @ 100 Hz
    ) -> Generator[ReplayFrame, None, None]:
        """Stream sequential ReplayFrame objects for UI animation or simulation playback."""
        trial, metadata, neural_result, forecast_result = self.prepare_trial_simulation(
            trial_id=trial_id, protocol=protocol
        )

        times = trial.times_me
        for idx in range(0, len(times), step_sample):
            t_ms = float(times[idx])
            frame = self.get_frame_at_time(
                trial=trial,
                metadata=metadata,
                neural_result=neural_result,
                forecast_result=forecast_result,
                replay_time_ms=t_ms,
                mode=mode,
            )
            yield frame
