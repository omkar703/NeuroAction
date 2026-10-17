"""Reusable inference interface and Kinematic Digital Twin runtime forecaster."""

from pathlib import Path
from typing import Dict, Optional, Union
import pickle
import numpy as np

from src.trajectory.schema import ForecastResult, TrajectorySample
from src.trajectory.models import RidgeTrajectoryForecaster
from src.trajectory.features import extract_kinematic_features


class KinematicTwinForecaster:
    """Production inference runtime for the Kinematic Digital Twin forecaster."""

    def __init__(self, model: Optional[RidgeTrajectoryForecaster] = None):
        self.model = model or RidgeTrajectoryForecaster(alpha=10.0)

    def save(self, model_dir: Path = Path("models/phase_3"), filename: str = "kinematic_forecaster.pkl"):
        """Persist model and fitted parameters to disk."""
        model_dir.mkdir(parents=True, exist_ok=True)
        model_file = model_dir / filename
        with open(model_file, "wb") as f:
            pickle.dump(self.model, f)
        print(f"Saved KinematicTwinForecaster to {model_file}")

    @classmethod
    def load(cls, model_dir: Path = Path("models/phase_3"), filename: str = "kinematic_forecaster.pkl") -> "KinematicTwinForecaster":
        """Load trained forecaster from disk."""
        model_file = model_dir / filename
        if not model_file.exists():
            raise FileNotFoundError(f"Model file {model_file} not found. Ensure models are trained.")
        with open(model_file, "rb") as f:
            model = pickle.load(f)
        return cls(model=model)

    def forecast(
        self,
        observed_trajectory: np.ndarray,
        sampling_rate_hz: float = 100.0,
        forecast_horizon_ms: int = 200,
        trial_id: str = "live_stream",
    ) -> ForecastResult:
        """Forecast future trajectory and final endpoint from an observed movement snippet.
        
        Args:
            observed_trajectory: Array of shape (N, 3) in millimetres. Must contain at least 20 samples.
            sampling_rate_hz: Nominal sample rate in Hz (expected 100 Hz).
            forecast_horizon_ms: Prediction horizon in milliseconds (e.g. 100, 200, 300, 500 ms).
            trial_id: Identifier for logging.
        Returns:
            ForecastResult: Struct containing predicted trajectory, predicted endpoint, and metadata.
        """
        # 1. Validation
        if not isinstance(observed_trajectory, np.ndarray):
            observed_trajectory = np.array(observed_trajectory, dtype=np.float32)

        if observed_trajectory.ndim != 2 or observed_trajectory.shape[1] != 3:
            raise ValueError(
                f"Expected observed_trajectory of shape (N, 3), got {observed_trajectory.shape}"
            )

        if len(observed_trajectory) < 20:
            raise ValueError(
                f"Observed trajectory must contain at least 20 samples (200 ms @ 100 Hz), got {len(observed_trajectory)}"
            )

        if np.isnan(observed_trajectory).any() or np.isinf(observed_trajectory).any():
            raise ValueError("Input trajectory contains NaN or non-finite values.")

        # Take first 20 samples for the 200 ms prefix
        prefix_pos = observed_trajectory[:20].astype(np.float32)

        # Approximate velocities
        dt_s = 1.0 / sampling_rate_hz
        vel = np.zeros_like(prefix_pos)
        vel[0] = (prefix_pos[1] - prefix_pos[0]) / dt_s
        vel[-1] = (prefix_pos[-1] - prefix_pos[-2]) / dt_s
        vel[1:-1] = (prefix_pos[2:] - prefix_pos[:-2]) / (2.0 * dt_s)

        # Construct dummy sample for feature extraction
        sample = TrajectorySample(
            participant_id="online",
            run_id="online",
            trial_id=trial_id,
            congruency="unknown",
            target_goal="unknown",
            physical_hand_direction="unknown",
            sampling_rate=sampling_rate_hz,
            prefix_positions=prefix_pos,
            prefix_velocities=vel,
        )

        # Run model inference
        pred_endpoint = self.model.predict_endpoint(sample)
        pred_horizon = self.model.predict_horizon(sample, forecast_horizon_ms)

        return ForecastResult(
            participant_id="online",
            run_id="online",
            trial_id=trial_id,
            model_name=self.model.name,
            observation_duration_ms=200.0,
            forecast_horizon_ms=forecast_horizon_ms,
            sampling_rate_hz=sampling_rate_hz,
            coordinate_units="mm",
            observed_positions=prefix_pos,
            predicted_positions=pred_horizon,
            predicted_endpoint=pred_endpoint,
        )


def forecast_trajectory(
    observed_trajectory: np.ndarray,
    sampling_rate_hz: float = 100.0,
    forecast_horizon_ms: int = 200,
    forecaster: Optional[KinematicTwinForecaster] = None,
) -> ForecastResult:
    """Convenience functional wrapper for trajectory forecasting."""
    fc = forecaster or KinematicTwinForecaster.load()
    return fc.forecast(
        observed_trajectory=observed_trajectory,
        sampling_rate_hz=sampling_rate_hz,
        forecast_horizon_ms=forecast_horizon_ms,
    )
