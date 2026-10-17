import os
import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from src.data.trial_loader import TrialLoader
from src.data.manifest_builder import ManifestBuilder


@pytest.fixture(scope="module")
def manifest_df():
    builder = ManifestBuilder()
    df = builder.build_cached_manifest()
    return df


@pytest.fixture(scope="module")
def loader():
    return TrialLoader()


def test_manifest_unique_trial_ids(manifest_df):
    """Every trial in the manifest must have a unique identifier."""
    assert manifest_df["trial_id"].is_unique, "Duplicate trial_id found in manifest!"


def test_standardized_trial_indexing(manifest_df, loader):
    """Verify trial_index (0-based) and trial_number (1-based) relationship:
    trial_number == trial_index + 1.
    """
    for _, row in manifest_df.iterrows():
        assert row["trial_number"] == row["trial_index"] + 1, "trial_number must equal trial_index + 1"

    # Test load_trial vs load_trial_by_number
    t_idx0 = loader.load_trial("sub-01", "run-01_L1", 0)
    t_num1 = loader.load_trial_by_number("sub-01", "run-01_L1", 1)
    assert t_idx0.trial_id == t_num1.trial_id
    assert t_idx0.trial_index == 0
    assert t_idx0.trial_number == 1


def test_manifest_required_columns(manifest_df):
    """Manifest must contain all required separated semantic fields and indexing columns."""
    required = [
        "trial_id", "participant_id", "run_id", "trial_index", "trial_number",
        "eeg_seg_idx", "label_code", "ball_color", "target_goal",
        "physical_hand_direction", "ball_direction", "congruency",
        "pre_eeg_available", "execution_eeg_available",
        "kinematics_available", "coverage", "valid", "alignment_status",
    ]
    for col in required:
        assert col in manifest_df.columns, f"Missing column {col} in manifest"


def test_channel_count_and_provenance(loader):
    """Pre-movement EEG must contain exactly 26 scalp channels with C3, Cz, C4 present."""
    t1 = loader.load_trial("sub-01", "run-01_L1", 0)
    assert len(t1.channels) == 26
    for motor_ch in ["C3", "CZ", "C4", "FC3", "FC4", "CP3", "CP4"]:
        assert motor_ch in t1.channels, f"Motor channel {motor_ch} missing from channels!"


def test_pre_eeg_dimensions_and_srate(loader):
    """Pre-movement EEG must be 200 samples @ 200 Hz (1.0 second duration)."""
    t1 = loader.load_trial("sub-01", "run-01_L1", 0)
    assert t1.eeg_pre.shape == (200, 26)
    assert t1.srate_pre == 200.0
    assert len(t1.times_pre) == 200
    assert np.isclose(t1.times_pre[0], -1.0, atol=0.01)
    assert np.isclose(t1.times_pre[-1], -0.005, atol=0.01)


def test_kinematics_dimensions_match_timestamps(loader):
    """Kinematics length must exactly match timestamps length."""
    t1 = loader.load_trial("sub-01", "run-01_L1", 0)
    n_samples = len(t1.times_me)
    assert len(t1.hand_x) == n_samples
    assert len(t1.hand_y) == n_samples
    assert len(t1.hand_z) == n_samples
    assert len(t1.ball_x_pix) == n_samples
    assert t1.eeg_me.shape[0] == n_samples


def test_monotonic_timestamps(loader):
    """Execution timestamps must be strictly monotonic."""
    t1 = loader.load_trial("sub-01", "run-01_L1", 0)
    dt = np.diff(t1.times_me)
    assert np.all(dt > 0), "Non-monotonic timestamps detected in execution data!"
    assert np.allclose(dt, 10.0, atol=0.1), f"Expected 10.0 ms uniform step @ 100 Hz, got {dt[:5]}"


def test_condition_and_direction_semantics(manifest_df):
    """Verify ground truth label semantics and congruency relationship across all trials."""
    for _, row in manifest_df.iterrows():
        lc = row["label_code"]
        assert lc in (0, 1, 2, 3), f"Invalid label_code {lc}"
        
        # Color rule
        expected_color = "red" if lc in (0, 1) else "yellow"
        assert row["ball_color"] == expected_color, f"Color mismatch for label {lc}"
        
        # Target goal & Ball direction rule (in this paradigm, target goal = ball direction)
        expected_ball = "left" if lc in (0, 2) else "right"
        assert row["target_goal"] == expected_ball, f"Target goal mismatch for label {lc}"
        assert row["ball_direction"] == expected_ball, f"Ball direction mismatch for label {lc}"
        
        # Physical hand direction rule
        expected_hand = "left" if lc in (0, 3) else "right"
        assert row["physical_hand_direction"] == expected_hand, f"Hand direction mismatch for label {lc}"
        
        # Congruency rule
        if lc in (0, 1):
            assert row["congruency"] == "congruent"
            assert row["physical_hand_direction"] == row["ball_direction"]
        else:
            assert row["congruency"] == "incongruent"
            assert row["physical_hand_direction"] != row["ball_direction"]


def test_no_nan_in_pre_eeg(loader):
    """Pre-movement EEG for valid trials must contain no NaN or Inf."""
    t1 = loader.load_trial("sub-01", "run-01_L1", 0)
    assert not np.isnan(t1.eeg_pre).any(), "NaN found in pre-movement EEG!"
    assert not np.isinf(t1.eeg_pre).any(), "Inf found in pre-movement EEG!"


def test_multisubject_scalp_harmonization(loader):
    """28-channel execution runs (e.g. sub-15) must harmonize to 26 scalp channels via eeg_me_scalp."""
    t15 = loader.load_trial("sub-15", "run-01_L1", 0)
    assert t15.eeg_me.shape[1] == 28
    assert t15.eeg_me_scalp.shape[1] == 26
    assert t15.eeg_pre.shape[1] == 26


def test_dropped_trials_flagged_properly(manifest_df):
    """Verify dropped trials in sub-08:
    0-based indices [7, 8, 27] correspond to 1-based trial numbers [8, 9, 28].
    Must be explicitly flagged with alignment_status != 'aligned' and valid == False.
    """
    sub08_dropped = manifest_df[
        (manifest_df["participant_id"] == "sub-08") & 
        (manifest_df["alignment_status"] == "dropped_in_execution")
    ]
    dropped_0based = sub08_dropped["trial_index"].tolist()
    dropped_1based = sub08_dropped["trial_number"].tolist()
    
    assert dropped_0based == [7, 8, 27], f"Expected 0-based indices [7, 8, 27], got {dropped_0based}"
    assert dropped_1based == [8, 9, 28], f"Expected 1-based trial numbers [8, 9, 28], got {dropped_1based}"
    
    for _, row in sub08_dropped.iterrows():
        assert not row["valid"]
        assert row["coverage"] == 0.0
        assert not row["execution_eeg_available"]
