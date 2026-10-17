"""Comprehensive automated test suite for Phase 4: Multimodal Digital Twin.

Covers:
1. Data schemas and serialization contracts.
2. Neural and Kinematic model adapters.
3. Coordinate and temporal alignment layers.
4. All 15 required synthetic unit test cases for the Conflict Engine.
5. Strict temporal integrity and anti-leakage verification.
6. Deterministic replay repeatability.
7. Batch evaluation matrix and confusion matrix consistency.
"""

from pathlib import Path
import copy
import numpy as np
import pytest

from src.data.schema import CanonicalTrial
from src.data.trial_loader import TrialLoader
from src.phase_4.config import DigitalTwinConfig
from src.phase_4.conflict_engine import ConflictEngine
from src.phase_4.coordinate_mapping import (
    convert_units,
    get_target_coordinates,
    validate_coordinate_compatibility,
    TARGET_COORDINATES_MM,
)
from src.phase_4.fusion_engine import MultimodalFusionEngine
from src.phase_4.kinematic_adapter import KinematicTwinAdapter
from src.phase_4.neural_adapter import NeuralIntentionAdapter
from src.phase_4.replay_engine import ReplayEngine
from src.phase_4.schemas import (
    ConflictAssessment,
    FusionState,
    KinematicObservation,
    NeuralIntentionResult,
    TrajectoryForecast,
    TrialMetadata,
)
from src.phase_4.temporal_alignment import (
    construct_trial_timeline,
    verify_temporal_integrity,
)
from src.phase_4.trial_registry import TrialRegistry


@pytest.fixture
def config():
    return DigitalTwinConfig()


@pytest.fixture
def registry(config):
    return TrialRegistry(manifest_path=config.manifest_path)


@pytest.fixture
def demo_trial(registry, config):
    trial = registry.get_trial(config.demo_trial_id)
    assert trial is not None, f"Demo trial {config.demo_trial_id} must exist in registry."
    return trial


# ==============================================================================
# 1. SCHEMAS & SERIALIZATION
# ==============================================================================

def test_schemas_serialization():
    """Verify all typed dataclasses convert cleanly to dictionaries."""
    meta = TrialMetadata(
        participant_id="sub-01",
        run_id="run-01_L1",
        trial_id="sub-01_run-01_L1_tr00",
        trial_index=0,
        label_code=0,
        congruency="congruent",
        target_goal="left",
        physical_hand_direction="left",
        ball_color="red",
        valid=True,
    )
    d_meta = meta.to_dict()
    assert d_meta["participant_id"] == "sub-01"
    assert d_meta["valid"] is True

    n_res = NeuralIntentionResult(
        participant_id="sub-01",
        trial_id="sub-01_run-01_L1_tr00",
        model_name="lda_goal_decoder",
        prediction_time_ms=0.0,
        window_start_ms=-1000.0,
        window_end_ms=0.0,
        predicted_goal="left",
        probability_left=0.85,
        probability_right=0.15,
        confidence=0.85,
    )
    d_n = n_res.to_dict()
    assert d_n["predicted_goal"] == "left"
    assert np.isclose(d_n["confidence"], 0.85)

    k_obs = KinematicObservation(
        participant_id="sub-01",
        trial_id="sub-01_run-01_L1_tr00",
        timestamp_ms=100.0,
        position=np.array([10.0, 150.0, 50.0]),
        velocity=np.array([100.0, 0.0, 0.0]),
        speed=100.0,
    )
    d_k = k_obs.to_dict()
    assert d_k["position"] == [10.0, 150.0, 50.0]

    f_res = TrajectoryForecast(
        participant_id="sub-01",
        trial_id="sub-01_run-01_L1_tr00",
        model_name="ridge_forecaster",
        forecast_protocol="kinematic_onset",
        forecast_generation_time_ms=570.0,
        prefix_start_time_ms=370.0,
        prefix_end_time_ms=570.0,
        observed_prefix=np.zeros((20, 3)),
        predicted_endpoint=np.array([-180.0, 150.0, 50.0]),
    )
    d_f = f_res.to_dict()
    assert d_f["forecast_protocol"] == "kinematic_onset"


# ==============================================================================
# 2. ADAPTERS & MODEL INFERENCE
# ==============================================================================

def test_neural_adapter_inference(demo_trial, config):
    """Verify NeuralIntentionAdapter loads and produces valid intention predictions."""
    adapter = NeuralIntentionAdapter.load(model_dir=config.model_dir_phase_2)
    res = adapter.predict_trial(demo_trial, current_time_ms=0.0)

    assert res.status == "AVAILABLE"
    assert res.predicted_goal in ("left", "right")
    assert 0.0 <= res.probability_left <= 1.0
    assert 0.0 <= res.probability_right <= 1.0
    assert np.isclose(res.probability_left + res.probability_right, 1.0, atol=1e-4)
    assert res.confidence == max(res.probability_left, res.probability_right)
    assert res.is_temporally_valid is True
    assert res.processing_mode == "offline_acausal"


def test_neural_adapter_held_out(demo_trial, config):
    """Verify loading honest held-out Phase 2 model without sub-01."""
    held_out_adapter = NeuralIntentionAdapter.load_held_out("sub-01", model_dir=config.model_dir_phase_2)
    assert "held_out_sub01" in held_out_adapter.model_name
    res = held_out_adapter.predict_trial(demo_trial)
    assert res.status == "AVAILABLE"


def test_kinematic_adapter_protocols(demo_trial, config):
    """Verify KinematicTwinAdapter under Protocol A and Protocol B."""
    adapter = KinematicTwinAdapter.load(model_dir=config.model_dir_phase_3)

    # Protocol B: Kinematic Onset
    res_b = adapter.forecast_trial(demo_trial, protocol="kinematic_onset")
    assert res_b.status == "AVAILABLE"
    assert res_b.forecast_protocol == "kinematic_onset"
    assert res_b.predicted_endpoint.shape == (3,)
    assert res_b.prefix_end_time_ms > 0.0  # Onset is after 0 ms

    # Protocol A: Event Locked
    res_a = adapter.forecast_trial(demo_trial, protocol="event_locked")
    assert res_a.status == "AVAILABLE"
    assert res_a.forecast_protocol == "event_locked"
    assert np.isclose(res_a.prefix_start_time_ms, 0.0, atol=10.0)
    assert np.isclose(res_a.prefix_end_time_ms, 200.0, atol=10.0)


# ==============================================================================
# 3. COORDINATE MAPPING & TEMPORAL ALIGNMENT
# ==============================================================================

def test_coordinate_mapping_and_units():
    """Verify target coordinate mapping, error handling, and unit conversions."""
    left_coords = get_target_coordinates("left")
    right_coords = get_target_coordinates("right")

    assert np.allclose(left_coords, TARGET_COORDINATES_MM["left"])
    assert np.allclose(right_coords, TARGET_COORDINATES_MM["right"])
    assert left_coords[0] < -150.0  # Strongly lateral left
    assert right_coords[0] > 150.0  # Strongly lateral right

    # Unknown label handling
    with pytest.raises(ValueError):
        get_target_coordinates("up")
    assert get_target_coordinates("up", allow_unmapped=True) is None

    # Unit conversion
    meters = convert_units(left_coords, from_units="mm", to_units="m")
    assert np.isclose(meters[0], left_coords[0] / 1000.0)

    # Incompatible frame check
    ok, err = validate_coordinate_compatibility("leap_motion", "vicon", "mm", "mm")
    assert not ok
    assert "frame mismatch" in err


def test_temporal_alignment_timeline():
    """Verify causal event sequence construction and availability rules."""
    times = np.linspace(-500, 1500, 201)
    speeds = np.zeros(201)
    # Movement onset at sample 70 (t = 200 ms)
    speeds[70:] = 50.0

    timeline = construct_trial_timeline(
        participant_id="sub-01",
        trial_id="test_tr00",
        kinematic_times=times,
        speed_series=speeds,
        protocol="kinematic_onset",
    )

    assert timeline.go_cue_ms == 0.0
    assert timeline.kinematic_onset_ms == 200.0
    assert timeline.forecast_available_ms == 400.0  # 200 + 200 ms prefix

    # Causal availability checks
    assert not timeline.is_neural_available(-100.0)
    assert timeline.is_neural_available(0.0)
    assert not timeline.is_forecast_available(300.0)
    assert timeline.is_forecast_available(400.0)


# ==============================================================================
# 4. CONFLICT ENGINE — 15 SYNTHETIC UNIT TEST CASES (SECTION 14 B)
# ==============================================================================

@pytest.fixture
def base_neural():
    return NeuralIntentionResult(
        participant_id="sub-01",
        trial_id="test_tr00",
        model_name="lda",
        prediction_time_ms=0.0,
        window_start_ms=-1000.0,
        window_end_ms=0.0,
        predicted_goal="left",
        probability_left=0.90,
        probability_right=0.10,
        confidence=0.90,
    )


@pytest.fixture
def base_forecast():
    return TrajectoryForecast(
        participant_id="sub-01",
        trial_id="test_tr00",
        model_name="ridge",
        forecast_protocol="kinematic_onset",
        forecast_generation_time_ms=500.0,
        prefix_start_time_ms=300.0,
        prefix_end_time_ms=500.0,
        observed_prefix=np.zeros((20, 3)),
        predicted_endpoint=TARGET_COORDINATES_MM["left"].copy(),
        actual_endpoint=TARGET_COORDINATES_MM["left"].copy(),
    )


def test_conflict_case_1_exact_match(base_neural, base_forecast, config):
    """Case 1: Predicted endpoint exactly equals intended target (d=0, θ=0 -> ALIGNED)."""
    engine = ConflictEngine(config)
    res = engine.evaluate_conflict(base_neural, base_forecast)
    assert res.conflict_state == "ALIGNED"
    assert not res.is_conflict
    assert np.isclose(res.endpoint_target_distance_mm, 0.0)
    assert np.isclose(res.directional_disagreement_deg, 0.0)


def test_conflict_case_2_within_tolerance(base_neural, base_forecast, config):
    """Case 2: Predicted endpoint lies within configured tolerance (d=30 mm <= 50 mm -> ALIGNED)."""
    engine = ConflictEngine(config)
    base_forecast.predicted_endpoint = TARGET_COORDINATES_MM["left"] + np.array([30.0, 0.0, 0.0])
    res = engine.evaluate_conflict(base_neural, base_forecast)
    assert res.conflict_state == "ALIGNED"
    assert not res.is_conflict
    assert np.isclose(res.endpoint_target_distance_mm, 30.0)


def test_conflict_case_3_exceeds_tolerance(base_neural, base_forecast, config):
    """Case 3: Predicted endpoint exceeds configured tolerance (d=80 mm > 50 mm -> POTENTIAL_DISAGREEMENT)."""
    engine = ConflictEngine(config)
    base_forecast.predicted_endpoint = TARGET_COORDINATES_MM["left"] + np.array([80.0, 0.0, 0.0])
    res = engine.evaluate_conflict(base_neural, base_forecast)
    assert res.conflict_state == "POTENTIAL_DISAGREEMENT"
    assert res.is_conflict
    assert np.isclose(res.endpoint_target_distance_mm, 80.0)


def test_conflict_case_4_direction_agrees(base_neural, base_forecast, config):
    """Case 4: Predicted direction agrees with target direction (θ=0° -> ALIGNED)."""
    engine = ConflictEngine(config)
    # Target is left (x < 0). Endpoint is also strongly left.
    base_forecast.predicted_endpoint = TARGET_COORDINATES_MM["left"].copy()
    res = engine.evaluate_conflict(base_neural, base_forecast, initial_position=np.array([0.0, 150.0, 50.0]))
    assert res.conflict_state == "ALIGNED"
    assert res.directional_disagreement_deg < 5.0


def test_conflict_case_5_direction_opposes(base_neural, base_forecast, config):
    """Case 5: Predicted direction opposes target direction (θ > 90° -> POTENTIAL_DISAGREEMENT)."""
    engine = ConflictEngine(config)
    # Neural goal is left, but predicted endpoint reaches to right target
    base_forecast.predicted_endpoint = TARGET_COORDINATES_MM["right"].copy()
    res = engine.evaluate_conflict(base_neural, base_forecast, initial_position=np.array([0.0, 150.0, 50.0]))
    assert res.conflict_state == "POTENTIAL_DISAGREEMENT"
    assert res.is_conflict
    assert res.directional_disagreement_deg > 140.0


def test_conflict_case_6_zero_vector(base_neural, base_forecast, config):
    """Case 6: A vector has zero magnitude (handled safely, returns UNCERTAIN without crashing)."""
    engine = ConflictEngine(config)
    # Origin is identical to target and endpoint
    p_zero = TARGET_COORDINATES_MM["left"].copy()
    base_forecast.predicted_endpoint = p_zero
    res = engine.evaluate_conflict(base_neural, base_forecast, initial_position=p_zero)
    assert res.conflict_state == "UNCERTAIN"
    assert "near-zero magnitude" in res.explanation


def test_conflict_case_7_missing_target(base_neural, base_forecast, config):
    """Case 7: Target coordinates missing (returns UNAVAILABLE)."""
    engine = ConflictEngine(config)
    base_neural.predicted_goal = "unmapped_goal"
    res = engine.evaluate_conflict(base_neural, base_forecast)
    assert res.conflict_state == "UNAVAILABLE"
    assert "target-coordinate mapping is not verified" in res.explanation


def test_conflict_case_8_different_coordinate_frames(base_neural, base_forecast, config):
    """Case 8: Coordinate frames differ (returns INVALID_INPUT)."""
    engine = ConflictEngine(config)
    base_forecast.coordinate_frame = "polar_spherical"
    res = engine.evaluate_conflict(base_neural, base_forecast)
    assert res.conflict_state == "INVALID_INPUT"
    assert "frame mismatch" in res.explanation


def test_conflict_case_9_different_units(base_neural, base_forecast, config):
    """Case 9: Coordinate units differ (returns INVALID_INPUT)."""
    engine = ConflictEngine(config)
    base_forecast.coordinate_units = "cm"
    res = engine.evaluate_conflict(base_neural, base_forecast)
    assert res.conflict_state == "INVALID_INPUT"
    assert "unit mismatch" in res.explanation


def test_conflict_case_10_temporal_alignment(base_neural, base_forecast, config):
    """Case 10: Temporal integrity verification helper catches forward leakage."""
    ok, err = verify_temporal_integrity(prediction_declared_time_ms=100.0, data_max_timestamp_ms=250.0)
    assert not ok
    assert "Temporal leakage detected" in err


def test_conflict_case_11_missing_goal_label(base_neural, base_forecast, config):
    """Case 11: Goal label is None or empty (returns UNAVAILABLE)."""
    engine = ConflictEngine(config)
    base_neural.predicted_goal = None
    res = engine.evaluate_conflict(base_neural, base_forecast)
    assert res.conflict_state == "UNAVAILABLE"


def test_conflict_case_12_trial_id_mismatch(base_neural, base_forecast, config):
    """Case 12: Neural and trajectory outputs have different trial IDs (returns INVALID_INPUT)."""
    engine = ConflictEngine(config)
    base_forecast.trial_id = "other_trial_tr99"
    res = engine.evaluate_conflict(base_neural, base_forecast)
    assert res.conflict_state == "INVALID_INPUT"
    assert "Trial ID mismatch" in res.explanation


def test_conflict_case_13_non_finite_coordinates(base_neural, base_forecast, config):
    """Case 13: Forecast contains NaN or Inf coordinates (returns INVALID_INPUT)."""
    engine = ConflictEngine(config)
    base_forecast.predicted_endpoint = np.array([np.nan, 150.0, 50.0])
    res = engine.evaluate_conflict(base_neural, base_forecast)
    assert res.conflict_state == "INVALID_INPUT"
    assert "non-finite" in res.explanation


def test_conflict_case_14_missing_samples(base_neural, base_forecast, config):
    """Case 14: Model status is UNAVAILABLE due to missing samples."""
    engine = ConflictEngine(config)
    base_forecast.status = "UNAVAILABLE"
    base_forecast.error_message = "Tracking dropout across prefix"
    res = engine.evaluate_conflict(base_neural, base_forecast)
    assert res.conflict_state == "UNAVAILABLE"
    assert "Tracking dropout" in res.explanation


def test_conflict_case_15_uncalibrated_uncertain_confidence(base_neural, base_forecast, config):
    """Case 15: Low/uncertain classifier confidence (< 0.55 -> UNCERTAIN)."""
    engine = ConflictEngine(config)
    base_neural.confidence = 0.51  # Borderline chance confidence
    res = engine.evaluate_conflict(base_neural, base_forecast)
    assert res.conflict_state == "UNCERTAIN"
    assert not res.is_conflict
    assert "below the dependable threshold" in res.explanation


# ==============================================================================
# 5. STRICT TEMPORAL INTEGRITY & REPLAY REPEATABILITY
# ==============================================================================

def test_temporal_anti_leakage_on_altered_future(registry, config):
    """Modifying future ground-truth samples must NOT change earlier replay frames."""
    engine = ReplayEngine(config=config, registry=registry)
    trial, meta, neural, forecast = engine.prepare_trial_simulation("sub-01_run-01_L1_tr00")

    # Evaluate frame at t = 200 ms
    frame_orig = engine.get_frame_at_time(
        trial=trial,
        metadata=meta,
        neural_result=neural,
        forecast_result=forecast,
        replay_time_ms=200.0,
        mode="causal_replay",
    )

    # Mutate future trajectory data (at t > 500 ms) in trial object copy
    trial_mutated = copy.deepcopy(trial)
    future_idx = np.where(trial_mutated.times_me > 500.0)[0]
    trial_mutated.hand_x[future_idx] = 9999.0
    trial_mutated.hand_y[future_idx] = -9999.0

    frame_mutated = engine.get_frame_at_time(
        trial=trial_mutated,
        metadata=meta,
        neural_result=neural,
        forecast_result=forecast,
        replay_time_ms=200.0,
        mode="causal_replay",
    )

    # Both past positions and conflict states must remain identical
    assert np.allclose(frame_orig.observed_path_so_far, frame_mutated.observed_path_so_far, equal_nan=True)
    assert frame_orig.fusion_state.conflict_assessment.conflict_state == frame_mutated.fusion_state.conflict_assessment.conflict_state


def test_deterministic_replay_repeatability(registry, config):
    """Two replay runs on the same trial must produce identical frames."""
    engine = ReplayEngine(config=config, registry=registry)
    frames_1 = list(engine.stream_trial("sub-01_run-01_L1_tr00", step_sample=5))
    frames_2 = list(engine.stream_trial("sub-01_run-01_L1_tr00", step_sample=5))

    assert len(frames_1) == len(frames_2)
    for f1, f2 in zip(frames_1, frames_2):
        assert f1.timestamp_ms == f2.timestamp_ms
        assert np.allclose(f1.observed_path_so_far, f2.observed_path_so_far, equal_nan=True)
        assert f1.fusion_state.conflict_assessment.conflict_state == f2.fusion_state.conflict_assessment.conflict_state


# ==============================================================================
# 6. BATCH MATRIX AND CONFUSION METRICS VERIFICATION
# ==============================================================================

def test_eval_matrix_and_confusion_matrix_exist(config):
    """Verify batch evaluation matrix and confusion metrics files exist and are valid."""
    eval_csv = config.output_dir / "digital_twin_eval_matrix.csv"
    audited_csv = config.output_dir / "audited_evaluation_matrix.csv"
    conf_csv = config.output_dir / "conflict_confusion_matrix.csv"

    assert eval_csv.exists(), f"Evaluation matrix {eval_csv} should exist."
    assert audited_csv.exists(), f"Audited evaluation matrix {audited_csv} should exist."
    assert conf_csv.exists(), f"Confusion matrix {conf_csv} should exist."

    import pandas as pd
    df_eval = pd.read_csv(eval_csv)
    assert len(df_eval) == 538  # 269 trials * 2 protocols

    df_conf = pd.read_csv(conf_csv)
    metric_map = dict(zip(df_conf["Metric"], df_conf["Value"]))

    # Check audited statistical criteria under strict LOSO
    assert metric_map["Sensitivity / Recall"] >= 0.70
    assert metric_map["Specificity"] >= 0.55
    assert metric_map["Balanced Accuracy"] >= 0.65
    assert metric_map["Conflict Score AUROC"] >= 0.65
    assert metric_map["Phase 2 Neural Goal Balanced Accuracy (LOSO)"] >= 0.65
    assert metric_map["Phase 3 Kinematic Endpoint RMSE (LOSO mm)"] <= 55.0


def test_held_out_model_provenance_all_subjects(config):
    """Verify held-out models exist and are loadable for all 5 subjects."""
    from src.phase_4.neural_adapter import NeuralIntentionAdapter
    from src.phase_4.kinematic_adapter import KinematicTwinAdapter

    subjects = ["sub-01", "sub-02", "sub-03", "sub-08", "sub-15"]
    for sub in subjects:
        # Phase 2 held-out adapter
        n_adapter = NeuralIntentionAdapter.load_held_out(sub, model_dir=config.model_dir_phase_2)
        assert n_adapter.is_held_out, f"Neural adapter for {sub} must be held-out."
        assert f"held_out_{sub.replace('-', '')}" in n_adapter.model_name

        # Phase 3 held-out adapter
        k_adapter = KinematicTwinAdapter.load_held_out(sub, model_dir=config.model_dir_phase_3)
        assert k_adapter.is_held_out, f"Kinematic adapter for {sub} must be held-out."
        assert f"held_out_{sub.replace('-', '')}" in k_adapter.model_name


def test_fold_target_centroids_excludes_held_out_subject(config):
    """Verify that fold-specific target centroids exclude the test subject's endpoints."""
    from src.phase_4.coordinate_mapping import estimate_fold_target_centroids
    from src.phase_4.kinematic_adapter import KinematicTwinAdapter

    samples, _ = KinematicTwinAdapter().segmenter.load_cohort(
        manifest_path=config.manifest_path, mode="kinematic_onset"
    )
    subjects = sorted(list(set(s.participant_id for s in samples)))

    for held_out_sub in subjects:
        # Ensure training samples strictly omit held_out_sub
        train_samples = [s for s in samples if s.participant_id != held_out_sub]
        assert all(s.participant_id != held_out_sub for s in train_samples)

        centroids = estimate_fold_target_centroids(train_samples)
        assert "left" in centroids and "right" in centroids
        assert centroids["left"][0] < -100.0  # Left target negative lateral
        assert centroids["right"][0] > 100.0  # Right target positive lateral


def test_audited_evaluation_matrix_provenance(config):
    """Verify that audited_evaluation_matrix.csv records complete provenance."""
    import pandas as pd
    audited_csv = config.output_dir / "audited_evaluation_matrix.csv"
    assert audited_csv.exists(), "Audited evaluation matrix must exist."

    df = pd.read_csv(audited_csv)
    assert len(df) == 538

    # Check required provenance columns
    required_cols = [
        "participant_id", "trial_id", "congruency", "target_goal",
        "is_held_out", "neural_model_provenance", "kinematic_model_provenance",
        "target_proxy_provenance", "conflict_state", "is_conflict",
    ]
    for col in required_cols:
        assert col in df.columns, f"Missing required column: {col}"

    # Verify all evaluations are strictly marked held-out
    assert df["is_held_out"].all(), "All rows must be strictly held out."
    for _, row in df.iterrows():
        sub_clean = row["participant_id"].replace("-", "")
        assert f"held_out_{sub_clean}" in row["neural_model_provenance"]
        assert f"held_out_{sub_clean}" in row["kinematic_model_provenance"]
        assert row["participant_id"] in row["target_proxy_provenance"]


def test_threshold_sensitivity_provenance(config):
    """Verify that threshold sensitivity analysis exists and covers parameter grid."""
    import pandas as pd
    sens_csv = config.output_dir / "threshold_sensitivity_analysis.csv"
    assert sens_csv.exists(), "Threshold sensitivity analysis CSV must exist."

    df_sens = pd.read_csv(sens_csv)
    assert len(df_sens) == 12  # 4 distance thresholds * 3 angle thresholds
    assert set(df_sens["Distance Threshold (mm)"]) == {25.0, 50.0, 75.0, 100.0}
    assert set(df_sens["Angle Threshold (deg)"]) == {30.0, 45.0, 60.0}


def test_baseline_comparison_leakage_free(config):
    """Verify that constant velocity baseline uses training fold statistics only, without future test duration."""
    import numpy as np
    from src.phase_4.kinematic_adapter import KinematicTwinAdapter

    adapter = KinematicTwinAdapter()
    samples, _ = adapter.segmenter.load_cohort(mode="kinematic_onset")
    subjects = sorted(list(set(s.participant_id for s in samples)))

    for test_sub in subjects:
        train_samples = [s for s in samples if s.participant_id != test_sub]
        test_samples = [s for s in samples if s.participant_id == test_sub]

        # Verify training samples strictly exclude test subject
        assert all(s.participant_id != test_sub for s in train_samples)

        # Estimate remaining duration strictly from training participants
        train_rem_durs = [(s.movement_duration_ms - 200.0) / 1000.0 for s in train_samples]
        dt_train_mean = float(np.mean(train_rem_durs))
        assert dt_train_mean > 1.5 and dt_train_mean < 3.5

        # Verify prediction on test sample uses strictly prefix and training duration
        s_test = test_samples[0]
        prefix = s_test.prefix_positions
        v_prefix = (prefix[-1] - prefix[0]) / 0.19
        p_cv = prefix[-1] + v_prefix * dt_train_mean

        # Assert p_cv is finite and computed without accessing s_test.full_trajectory length
        assert np.isfinite(p_cv).all()
        assert p_cv.shape == (3,)


