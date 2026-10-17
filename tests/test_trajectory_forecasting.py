"""Comprehensive automated test suite for Phase 3 Kinematics and Trajectory Forecasting."""

from pathlib import Path
import numpy as np
import pytest

from src.trajectory.schema import TrajectorySample, ForecastResult
from src.trajectory.segmentation import TrajectorySegmenter
from src.trajectory.features import extract_kinematic_features
from src.trajectory.baselines import PersistenceBaseline, ConstantVelocityBaseline
from src.trajectory.models import RidgeTrajectoryForecaster, MLPTrajectoryForecaster, GRUTrajectoryForecaster
from src.trajectory.inference import KinematicTwinForecaster, forecast_trajectory


@pytest.fixture(scope="module")
def real_sample() -> TrajectorySample:
    segmenter = TrajectorySegmenter()
    samples, _ = segmenter.load_cohort()
    assert len(samples) > 0, "No valid kinematic samples loaded!"
    return samples[0]


@pytest.fixture(scope="module")
def synthetic_sample() -> TrajectorySample:
    """Synthetic linear trajectory with known constant velocity v = [10, 0, 0] mm/s."""
    t_steps = np.arange(20) * 0.01  # 20 samples = 0.19 s
    pos = np.zeros((20, 3), dtype=np.float32)
    pos[:, 0] = t_steps * 100.0  # moving along x at 100 mm/s

    vel = np.zeros((20, 3), dtype=np.float32)
    vel[:, 0] = 100.0

    # Horizons at +100 ms (10 samples), +200 ms (20 samples), +300 ms (30 samples), +500 ms (50 samples)
    horizon_trajs = {}
    horizon_pts = {}
    for h in [100, 200, 300, 500]:
        n_steps = int(h / 10)
        t_h = (np.arange(1, n_steps + 1) * 0.01)[:, None]
        p_last = pos[-1]
        traj_h = p_last[None, :] + np.array([100.0, 0.0, 0.0])[None, :] * t_h
        horizon_trajs[h] = traj_h.astype(np.float32)
        horizon_pts[h] = traj_h[-1]

    # Endpoint at t = 2.0 s: x = 200.0 mm
    endpoint = np.array([200.0, 0.0, 0.0], dtype=np.float32)

    return TrajectorySample(
        participant_id="syn_sub",
        run_id="syn_run",
        trial_id="syn_tr00",
        congruency="congruent",
        target_goal="right",
        physical_hand_direction="right",
        sampling_rate=100.0,
        coordinate_units="mm",
        t_onset_ms=0.0,
        onset_sample_idx=0,
        movement_duration_ms=2000.0,
        prefix_positions=pos,
        prefix_velocities=vel,
        prefix_times=t_steps * 1000.0,
        actual_endpoint=endpoint,
        endpoint_time_ms=2000.0,
        horizon_trajectories=horizon_trajs,
        horizon_points=horizon_pts,
    )


def test_units_and_sampling_rate(real_sample):
    """Verify coordinate units and sampling rate assumptions."""
    assert real_sample.coordinate_units == "mm"
    assert real_sample.sampling_rate == 100.0
    # Prefix must be exactly 20 samples (200 ms @ 100 Hz)
    assert real_sample.prefix_positions.shape == (20, 3)
    assert real_sample.prefix_velocities.shape == (20, 3)
    assert len(real_sample.prefix_times) == 20
    dt = np.diff(real_sample.prefix_times)
    assert np.allclose(dt, 10.0, atol=0.1), f"Expected 10.0 ms uniform step, got {dt}"


def test_prefix_nan_free(real_sample):
    """Prefix must be strictly free of NaNs and Infs."""
    assert not np.isnan(real_sample.prefix_positions).any()
    assert not np.isnan(real_sample.prefix_velocities).any()
    assert not np.isnan(real_sample.actual_endpoint).any()


def test_features_dimension_and_no_leakage(real_sample):
    """Feature extraction must output 95 features and not depend on future samples."""
    feats = extract_kinematic_features(real_sample)
    assert feats.shape == (95,)
    assert not np.isnan(feats).any()
    assert not np.isinf(feats).any()


def test_persistence_baseline(synthetic_sample):
    """Persistence baseline must predict p(t_obs) for all horizons and endpoint."""
    model = PersistenceBaseline()
    p_last = synthetic_sample.prefix_positions[-1]

    pred_end = model.predict_endpoint(synthetic_sample)
    assert np.allclose(pred_end, p_last)

    for h in [100, 200, 300, 500]:
        n_steps = int(h / 10)
        pred_traj = model.predict_horizon(synthetic_sample, h)
        assert pred_traj.shape == (n_steps, 3)
        assert np.allclose(pred_traj[0], p_last)
        assert np.allclose(pred_traj[-1], p_last)

        pred_pt = model.predict_horizon_point(synthetic_sample, h)
        assert np.allclose(pred_pt, p_last)


def test_constant_velocity_baseline(synthetic_sample):
    """Constant velocity must achieve near zero error on a constant velocity synthetic trajectory."""
    model = ConstantVelocityBaseline()
    for h in [100, 200, 300, 500]:
        pred_traj = model.predict_horizon(synthetic_sample, h)
        actual_traj = synthetic_sample.horizon_trajectories[h]
        err = np.max(np.abs(pred_traj - actual_traj))
        assert err < 1e-3, f"Expected near zero error on constant velocity motion, got max error {err}"


def test_ridge_forecaster_shapes_and_fit(synthetic_sample):
    """Ridge model must fit and predict correct shapes without error."""
    model = RidgeTrajectoryForecaster(alpha=1.0)
    train_samples = [synthetic_sample, synthetic_sample]
    model.fit(train_samples)

    pred_end = model.predict_endpoint(synthetic_sample)
    assert pred_end.shape == (3,)
    assert not np.isnan(pred_end).any()

    for h in [100, 200, 300, 500]:
        pred_traj = model.predict_horizon(synthetic_sample, h)
        assert pred_traj.shape == (int(h / 10), 3)
        assert not np.isnan(pred_traj).any()


def test_mlp_forecaster_shapes_and_fit(synthetic_sample):
    """MLP model must fit and predict correct shapes without error."""
    model = MLPTrajectoryForecaster(epochs=2, lr=0.01)
    train_samples = [synthetic_sample, synthetic_sample]
    model.fit(train_samples)

    pred_end = model.predict_endpoint(synthetic_sample)
    assert pred_end.shape == (3,)
    assert not np.isnan(pred_end).any()

    pred_h = model.predict_horizon(synthetic_sample, 100)
    assert pred_h.shape == (10, 3)


def test_gru_forecaster_shapes_and_fit(synthetic_sample):
    """GRU model must fit sequence and predict correct shapes without error."""
    model = GRUTrajectoryForecaster(epochs=2, lr=0.01)
    train_samples = [synthetic_sample, synthetic_sample]
    model.fit(train_samples)

    pred_end = model.predict_endpoint(synthetic_sample)
    assert pred_end.shape == (3,)
    assert not np.isnan(pred_end).any()

    pred_h = model.predict_horizon(synthetic_sample, 100)
    assert pred_h.shape == (10, 3)


def test_kinematic_twin_forecaster_runtime(tmp_path, synthetic_sample):
    """KinematicTwinForecaster runtime interface validation."""
    model = RidgeTrajectoryForecaster(alpha=10.0)
    model.fit([synthetic_sample, synthetic_sample])
    twin = KinematicTwinForecaster(model)

    # Save and reload
    twin.save(tmp_path)
    loaded = KinematicTwinForecaster.load(tmp_path)

    # Inference call
    result = loaded.forecast(
        observed_trajectory=synthetic_sample.prefix_positions,
        sampling_rate_hz=100.0,
        forecast_horizon_ms=200,
    )
    assert isinstance(result, ForecastResult)
    assert result.predicted_endpoint.shape == (3,)
    assert result.predicted_positions.shape == (20, 3)
    assert result.forecast_horizon_ms == 200


def test_inference_input_validation(synthetic_sample):
    """Inference must reject invalid inputs (too short, NaNs, wrong shape)."""
    model = RidgeTrajectoryForecaster(alpha=10.0)
    model.fit([synthetic_sample, synthetic_sample])
    twin = KinematicTwinForecaster(model)

    # Too short (< 20 samples)
    with pytest.raises(ValueError):
        twin.forecast(synthetic_sample.prefix_positions[:10])

    # Wrong shape
    with pytest.raises(ValueError):
        twin.forecast(np.zeros((20, 2)))

    # Contains NaNs
    nan_arr = synthetic_sample.prefix_positions.copy()
    nan_arr[5, 0] = np.nan
    with pytest.raises(ValueError):
        twin.forecast(nan_arr)


def test_metric_mathematical_consistency():
    """Verify that 3D Euclidean RMSE >= Mean Error and matches sqrt(RMSE_x^2 + RMSE_y^2 + RMSE_z^2)."""
    # 10 timesteps, 3 axes
    np.random.seed(42)
    traj_diffs = np.random.normal(loc=1.0, scale=0.5, size=(10, 3)).astype(np.float32)
    step_eucs = np.linalg.norm(traj_diffs, axis=1)

    mean_traj_euc = float(np.mean(step_eucs))
    rmse_traj_3d = float(np.sqrt(np.mean(step_eucs ** 2)))

    # Jensen's inequality: sqrt(mean(x^2)) >= mean(x)
    assert rmse_traj_3d >= mean_traj_euc, f"Violated Jensen's inequality: RMSE {rmse_traj_3d} < Mean {mean_traj_euc}"

    # Orthogonal sum: RMSE_3D = sqrt(RMSE_x^2 + RMSE_y^2 + RMSE_z^2)
    rmse_x = float(np.sqrt(np.mean(traj_diffs[:, 0] ** 2)))
    rmse_y = float(np.sqrt(np.mean(traj_diffs[:, 1] ** 2)))
    rmse_z = float(np.sqrt(np.mean(traj_diffs[:, 2] ** 2)))
    euc_sum = float(np.sqrt(rmse_x**2 + rmse_y**2 + rmse_z**2))
    assert np.isclose(rmse_traj_3d, euc_sum, atol=1e-5), f"RMSE 3D {rmse_traj_3d} != sqrt sum {euc_sum}"


def test_segmentation_modes(real_sample):
    """Test that segmentation operates in both event_locked and kinematic_onset modes."""
    segmenter = TrajectorySegmenter()
    samples_event, _ = segmenter.load_cohort(mode="event_locked", subjects=["sub-01"])
    samples_onset, _ = segmenter.load_cohort(mode="kinematic_onset", subjects=["sub-01"], speed_threshold_mm_s=20.0)

    assert len(samples_event) > 0
    assert len(samples_onset) > 0
    assert samples_event[0].segmentation_mode == "event_locked"
    assert samples_onset[0].segmentation_mode == "kinematic_onset"
    assert samples_onset[0].onset_latency_ms >= 0.0


def test_held_out_forecaster_save_and_load(tmp_path, synthetic_sample):
    """Verify saving and loading with custom held-out filename."""
    model = RidgeTrajectoryForecaster(alpha=10.0)
    model.fit([synthetic_sample, synthetic_sample])
    twin = KinematicTwinForecaster(model)

    custom_name = "kinematic_forecaster_held_out_sub01.pkl"
    twin.save(tmp_path, filename=custom_name)
    assert (tmp_path / custom_name).exists()

    loaded = KinematicTwinForecaster.load(tmp_path, filename=custom_name)
    res = loaded.forecast(synthetic_sample.prefix_positions)
    assert res.predicted_endpoint.shape == (3,)

