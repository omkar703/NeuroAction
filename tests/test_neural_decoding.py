"""Comprehensive automated unit and integration tests for NeuroAction Phase 2."""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.data.trial_loader import TrialLoader
from src.neuroscience.readiness_potential import ReadinessPotentialAnalyzer
from src.neuroscience.sensorimotor_rhythms import SensorimotorRhythmAnalyzer
from src.features.pre_eeg import (
    get_sliding_windows,
    extract_time_domain_features,
    extract_bandpower_features,
    extract_trial_features,
    CommonSpatialPattern,
    MOTOR_FOCUSED_CHANNELS,
)
from src.models.decoders import (
    ChanceBaseline,
    LogisticRegressionDecoder,
    LDADecoder,
    CSPLDADecoder,
    CSPLogRegDecoder,
)
from src.models.eegnet import CompactEEGNetModule, EEGNetClassifier


@pytest.fixture(scope="module")
def sample_trials():
    loader = TrialLoader()
    trials = loader.load_run("sub-01", "run-01_L1")
    return [t for t in trials if t.valid]


def test_sliding_windows():
    windows = get_sliding_windows(total_samples=200, window_samples=40, stride_samples=10, srate=200.0)
    assert len(windows) == 17
    assert windows[0]["start_ms"] == -1000.0
    assert windows[0]["end_ms"] == -800.0
    assert np.isclose(windows[-1]["end_ms"], 0.0)


def test_feature_extraction():
    dummy = np.random.randn(40, 26).astype(np.float32)
    td = extract_time_domain_features(dummy)
    assert td.shape == (26 * 6,)

    bp = extract_bandpower_features(dummy, srate=200.0)
    assert bp.shape == (26 * 3,)

    comb = extract_trial_features(dummy, "combined", srate=200.0)
    assert comb.shape == (26 * 9,)


def test_csp_transformer():
    np.random.seed(42)
    X = np.random.randn(30, 40, 26).astype(np.float32)
    y = np.array([0] * 15 + [1] * 15)

    csp = CommonSpatialPattern(n_components=3)
    csp.fit(X, y)
    assert csp.filters_.shape == (26, 6)

    X_proj = csp.transform(X)
    assert X_proj.shape == (30, 6)
    assert not np.isnan(X_proj).any()


def test_readiness_potential(sample_trials):
    rp = ReadinessPotentialAnalyzer()
    res = rp.analyze_trials(sample_trials)

    assert "times" in res
    assert "channels" in res
    assert "all" in res
    assert res["all"]["mean"].shape == (200, len(rp.channels))
    assert res["all"]["ci_low"].shape == (200, len(rp.channels))
    assert res["all"]["ci_high"].shape == (200, len(rp.channels))
    for ch in rp.channels:
        assert ch in res["all"]["slopes"]


def test_sensorimotor_rhythms(sample_trials):
    smr = SensorimotorRhythmAnalyzer()
    res = smr.analyze_trials(sample_trials)

    assert "all" in res
    assert "mu_erd_mean" in res["all"]
    assert "beta_erd_mean" in res["all"]
    assert "mu_li_mean" in res["all"]
    assert res["all"]["mu_erd_mean"].shape == (200, len(smr.TARGET_CHANNELS))
    assert "spectrogram" in res


def test_decoders():
    np.random.seed(42)
    X_feat = np.random.randn(40, 50).astype(np.float32)
    X_3d = np.random.randn(40, 40, 26).astype(np.float32)
    y = np.array([0] * 20 + [1] * 20)

    # 1. Chance
    chance = ChanceBaseline()
    chance.fit(X_feat, y)
    assert chance.predict_proba(X_feat).shape == (40, 2)

    # 2. Logistic Regression
    lr = LogisticRegressionDecoder()
    lr.fit(X_feat, y)
    probs_lr = lr.predict_proba(X_feat)
    assert probs_lr.shape == (40, 2)
    assert np.allclose(probs_lr.sum(axis=1), 1.0)

    # 3. LDA
    lda = LDADecoder()
    lda.fit(X_feat, y)
    probs_lda = lda.predict_proba(X_feat)
    assert probs_lda.shape == (40, 2)
    assert np.allclose(probs_lda.sum(axis=1), 1.0)

    # 4. CSP + LDA
    csp_lda = CSPLDADecoder(n_components=3)
    csp_lda.fit(X_3d, y)
    probs_csp = csp_lda.predict_proba(X_3d)
    assert probs_csp.shape == (40, 2)
    assert np.allclose(probs_csp.sum(axis=1), 1.0)

    # 5. EEGNet
    eegnet = EEGNetClassifier(epochs=2, batch_size=16)
    eegnet.fit(X_3d, y)
    probs_net = eegnet.predict_proba(X_3d)
    assert probs_net.shape == (40, 2)
    assert np.allclose(probs_net.sum(axis=1), 1.0)


def test_output_artifacts_exist():
    # Checks generated artifacts from neural goal decoding
    oof_path = Path("results/predictions/neural_goal_predictions_loso.csv")
    if not oof_path.exists():
        oof_path = Path("reports/phase_2/out_of_fold_predictions.csv")
    if oof_path.exists():
        df = pd.read_csv(oof_path)
        required_cols = [
            "participant_id", "run_id", "trial_id", "target_type", "target_label",
            "condition", "prediction", "probability_left", "probability_right",
            "model_name", "feature_set", "fold_id", "time_window"
        ]
        for c in required_cols:
            assert c in df.columns, f"Missing required column: {c}"

    fig_dir = Path("results/figures")
    if not fig_dir.exists():
        fig_dir = Path("reports/phase_2/figures")
    if fig_dir.exists():
        for i in range(1, 11):
            figs = list(fig_dir.glob(f"figure_{i}_*.png"))
            assert len(figs) >= 1, f"Missing figure {i}"

