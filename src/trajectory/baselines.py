"""Baseline kinematic trajectory predictors: Persistence and Constant Velocity."""

from typing import Dict, List, Optional, Tuple
import numpy as np

from src.trajectory.schema import TrajectorySample, ForecastResult


class PersistenceBaseline:
    """Predicts that the hand remains stationary at the last observed position."""

    def __init__(self):
        self.name = "persistence"

    def fit(self, train_samples: List[TrajectorySample]):
        """No parameters to fit for persistence."""
        pass

    def predict_endpoint(self, sample: TrajectorySample) -> np.ndarray:
        """Predict endpoint: p_end = p(t_obs)."""
        return sample.prefix_positions[-1].copy()

    def predict_horizon(self, sample: TrajectorySample, horizon_ms: int) -> np.ndarray:
        """Predict multi-step trajectory over horizon: repeats p(t_obs). Shape: (n_steps, 3)."""
        n_steps = int(horizon_ms / 10)
        p_last = sample.prefix_positions[-1]
        return np.tile(p_last, (n_steps, 1))

    def predict_horizon_point(self, sample: TrajectorySample, horizon_ms: int) -> np.ndarray:
        """Predict coordinate at exactly horizon_ms: p(t_obs). Shape: (3,)."""
        return sample.prefix_positions[-1].copy()


class ConstantVelocityBaseline:
    """Extrapolates future positions linearly using velocity estimated from the observed prefix."""

    def __init__(self, velocity_method: str = "least_squares"):
        self.name = "constant_velocity"
        self.velocity_method = velocity_method
        self.mean_remaining_time_s: float = 2.5  # default remaining duration

    def fit(self, train_samples: List[TrajectorySample]):
        """Estimate mean movement completion duration from training fold."""
        durations_s = [
            (s.endpoint_time_ms - s.t_onset_ms - 200.0) / 1000.0
            for s in train_samples
            if s.endpoint_time_ms > s.t_onset_ms + 200.0
        ]
        if durations_s:
            self.mean_remaining_time_s = float(np.mean(durations_s))

    def _estimate_velocity(self, sample: TrajectorySample) -> np.ndarray:
        """Estimate velocity in mm/s from the observed prefix."""
        pos = sample.prefix_positions
        if self.velocity_method == "finite_difference":
            return (pos[-1] - pos[0]) / 0.19
        else:  # least squares trend over all 20 samples
            t = np.arange(20) * 0.01
            t_c = t - np.mean(t)
            denom = np.sum(t_c ** 2) + 1e-12
            v_ls = np.sum(t_c[:, None] * (pos - np.mean(pos, axis=0, keepdims=True)), axis=0) / denom
            return v_ls

    def predict_endpoint(self, sample: TrajectorySample) -> np.ndarray:
        """Predict endpoint using constant velocity over mean remaining duration: p(t_obs) + v * T_rem."""
        p_last = sample.prefix_positions[-1]
        v_est = self._estimate_velocity(sample)
        return p_last + v_est * self.mean_remaining_time_s

    def predict_horizon(self, sample: TrajectorySample, horizon_ms: int) -> np.ndarray:
        """Predict trajectory over horizon steps: p(t_obs + tau) = p(t_obs) + v * tau * dt. Shape: (n_steps, 3)."""
        n_steps = int(horizon_ms / 10)
        p_last = sample.prefix_positions[-1]
        v_est = self._estimate_velocity(sample)
        t_future = (np.arange(1, n_steps + 1) * 0.01)[:, None]  # shape (n_steps, 1)
        pred_traj = p_last[None, :] + v_est[None, :] * t_future
        return pred_traj.astype(np.float32)

    def predict_horizon_point(self, sample: TrajectorySample, horizon_ms: int) -> np.ndarray:
        """Predict position at exactly horizon_ms: p(t_obs) + v * (horizon_ms / 1000). Shape: (3,)."""
        p_last = sample.prefix_positions[-1]
        v_est = self._estimate_velocity(sample)
        dt = horizon_ms / 1000.0
        return (p_last + v_est * dt).astype(np.float32)
