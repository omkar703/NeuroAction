"""Adapter for Phase 3 Kinematic Digital Twin Forecaster."""

from pathlib import Path
from typing import Dict, List, Optional
import numpy as np

from src.data.schema import CanonicalTrial
from src.trajectory.schema import TrajectorySample
from src.trajectory.segmentation import TrajectorySegmenter
from src.trajectory.inference import KinematicTwinForecaster
from src.phase_4.schemas import TrajectoryForecast


class KinematicTwinAdapter:
    """Production runtime adapter wrapping trained Phase 3 trajectory forecasters."""

    def __init__(
        self,
        forecaster: Optional[KinematicTwinForecaster] = None,
        model_name: str = "ridge_trajectory_forecaster",
        is_held_out: bool = False,
        held_out_participant: str = "",
    ):
        self.forecaster = forecaster or KinematicTwinForecaster.load()
        self.model_name = model_name
        self.is_held_out = is_held_out
        self.held_out_participant = held_out_participant
        self.segmenter = TrajectorySegmenter()

    @classmethod
    def load(
        cls,
        model_dir: Path = Path("models/phase_3"),
        filename: str = "kinematic_forecaster.pkl",
    ) -> "KinematicTwinAdapter":
        """Load standard cohort kinematic forecaster."""
        forecaster = KinematicTwinForecaster.load(model_dir=model_dir, filename=filename)
        return cls(
            forecaster=forecaster,
            model_name=filename.replace(".pkl", ""),
            is_held_out=False,
            held_out_participant="",
        )

    @classmethod
    def load_held_out(
        cls,
        participant_id: str = "sub-01",
        model_dir: Path = Path("models/phase_3"),
    ) -> "KinematicTwinAdapter":
        """Load honest held-out model strictly trained without the target participant."""
        norm_sub = participant_id.replace("-", "")
        held_out_filename = f"kinematic_forecaster_held_out_{norm_sub}.pkl"
        held_out_path = model_dir / held_out_filename
        if held_out_path.exists():
            forecaster = KinematicTwinForecaster.load(model_dir=model_dir, filename=held_out_filename)
            return cls(
                forecaster=forecaster,
                model_name=held_out_filename.replace(".pkl", ""),
                is_held_out=True,
                held_out_participant=participant_id,
            )
        # Fallback to standard model if held-out artifact is not present
        return cls.load(model_dir=model_dir, filename="kinematic_forecaster.pkl")

    def forecast_trial(
        self,
        trial: CanonicalTrial,
        protocol: str = "kinematic_onset",
        speed_threshold_mm_s: float = 20.0,
        horizons_ms: Optional[List[int]] = None,
    ) -> TrajectoryForecast:
        """Segment trial kinematics and generate multi-horizon + endpoint forecasts.
        
        Args:
            trial: CanonicalTrial with movement-execution kinematics
            protocol: 'kinematic_onset' (Protocol B) or 'event_locked' (Protocol A)
            speed_threshold_mm_s: Velocity threshold in mm/s for physical kinematic onset
            horizons_ms: Target forecast horizons in ms (defaults to [100, 200, 300, 500])
        """
        horizons = horizons_ms or [100, 200, 300, 500]

        # 1. Segment CanonicalTrial using the verified Phase 3 TrajectorySegmenter
        sample: Optional[TrajectorySample] = self.segmenter.process_trial(
            trial=trial,
            mode=protocol,
            speed_threshold_mm_s=speed_threshold_mm_s,
        )

        if sample is None:
            return TrajectoryForecast(
                participant_id=trial.participant_id,
                trial_id=trial.trial_id,
                model_name=self.model_name,
                forecast_protocol=protocol,
                forecast_generation_time_ms=0.0,
                prefix_start_time_ms=0.0,
                prefix_end_time_ms=0.0,
                observed_prefix=np.zeros((20, 3), dtype=np.float32),
                predicted_endpoint=np.zeros((3,), dtype=np.float32),
                actual_endpoint=None,
                is_held_out=self.is_held_out,
                held_out_participant=self.held_out_participant,
                status="UNAVAILABLE",
                error_message="Kinematic prefix could not be segmented (missing tracking or short trial).",
            )

        # 2. Extract timing
        prefix_start_ms = float(sample.t_onset_ms)
        prefix_end_ms = prefix_start_ms + 200.0
        forecast_generation_ms = prefix_end_ms

        try:
            # 3. Model Inference via internal RidgeTrajectoryForecaster
            pred_endpoint = self.forecaster.model.predict_endpoint(sample)

            pred_horizons = {}
            pred_horizon_pts = {}
            for h in horizons:
                if h in self.forecaster.model.horizon_models:
                    pred_traj_h = self.forecaster.model.predict_horizon(sample, h)
                    pred_horizons[h] = pred_traj_h
                    pred_horizon_pts[h] = pred_traj_h[-1]

            return TrajectoryForecast(
                participant_id=trial.participant_id,
                trial_id=trial.trial_id,
                model_name=self.model_name,
                forecast_protocol=protocol,
                forecast_generation_time_ms=forecast_generation_ms,
                prefix_start_time_ms=prefix_start_ms,
                prefix_end_time_ms=prefix_end_ms,
                observed_prefix=sample.prefix_positions,
                predicted_endpoint=pred_endpoint,
                actual_endpoint=sample.actual_endpoint,
                horizon_trajectories=pred_horizons,
                horizon_points=pred_horizon_pts,
                coordinate_units="mm",
                coordinate_frame="leap_motion_cartesian",
                is_held_out=self.is_held_out,
                held_out_participant=self.held_out_participant,
                status="AVAILABLE",
                error_message=None,
            )
        except Exception as exc:
            return TrajectoryForecast(
                participant_id=trial.participant_id,
                trial_id=trial.trial_id,
                model_name=self.model_name,
                forecast_protocol=protocol,
                forecast_generation_time_ms=forecast_generation_ms,
                prefix_start_time_ms=prefix_start_ms,
                prefix_end_time_ms=prefix_end_ms,
                observed_prefix=sample.prefix_positions,
                predicted_endpoint=np.zeros((3,), dtype=np.float32),
                actual_endpoint=sample.actual_endpoint,
                is_held_out=self.is_held_out,
                held_out_participant=self.held_out_participant,
                status="ERROR",
                error_message=f"Forecaster inference failure: {str(exc)}",
            )
