"""Baseline models and single-modality predictors for Tasks A and B."""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np

from src.data.schema import CanonicalTrial
from src.phase_4.neural_adapter import NeuralIntentionAdapter
from src.phase_4.kinematic_adapter import KinematicTwinAdapter
from src.phase_4.schemas import NeuralIntentionResult, TrajectoryForecast
from src.phase_5.temporal_validation import TemporalValidator


class PersistenceBaselineModel:
    """Zero-order persistence baseline: assumes hand remains stationary at prefix end."""

    @staticmethod
    def predict_endpoint(prefix_positions: np.ndarray) -> np.ndarray:
        """Forecast resting endpoint as last observed position."""
        return np.asarray(prefix_positions[-1], dtype=np.float32)


class FairConstantVelocityBaselineModel:
    """First-order linear extrapolation baseline using strictly training-fold duration."""

    @staticmethod
    def predict_endpoint(
        prefix_positions: np.ndarray,
        dt_train_mean_s: float,
        dt_prefix_s: float = 0.19,
    ) -> np.ndarray:
        """Extrapolate forward using prefix mean velocity and training-cohort duration.
        
        CRITICAL: Never accesses test trial's future duration.
        """
        TemporalValidator.verify_duration_leakage_free(
            duration_strategy="fair_training_fold_mean",
            uses_test_trial_future_duration=False,
        )
        p_start = prefix_positions[0]
        p_end = prefix_positions[-1]
        v_prefix = (p_end - p_start) / max(dt_prefix_s, 1e-4)
        return (p_end + v_prefix * dt_train_mean_s).astype(np.float32)


class NeuralTargetClassifier:
    """Task A: Pre-movement scalp EEG goal classification (evaluating visual stimulus location)."""

    def __init__(self, adapter: NeuralIntentionAdapter):
        self.adapter = adapter

    def predict(self, trial: CanonicalTrial) -> NeuralIntentionResult:
        """Run held-out LDA goal decoder at visual go-cue (t = 0.0 ms)."""
        return self.adapter.predict_trial(trial=trial, current_time_ms=0.0)


class KinematicTrajectoryForecaster:
    """Task B: Early kinematic 3D trajectory and resting endpoint forecasting."""

    def __init__(self, adapter: KinematicTwinAdapter):
        self.adapter = adapter

    def forecast(
        self,
        trial: CanonicalTrial,
        protocol: str = "kinematic_onset",
        speed_threshold_mm_s: float = 20.0,
        horizons_ms: Optional[List[int]] = None,
    ) -> TrajectoryForecast:
        """Run held-out Ridge forecaster on 200 ms movement prefix."""
        horizons = horizons_ms or [100, 200, 300, 500]
        return self.adapter.forecast_trial(
            trial=trial,
            protocol=protocol,
            speed_threshold_mm_s=speed_threshold_mm_s,
            horizons_ms=horizons,
        )
