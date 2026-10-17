"""Automated integration tests for NeuroAction Phase 6 Interactive Digital Twin.

Verifies:
1. FastAPI application routing and schema validation.
2. Participant discovery and trial availability across all 5 benchmark subjects.
3. Causal kinematic replay timeframes, endpoint coordinates, and conflict timeseries.
4. EEG signal extraction and non-causal zero-phase filter disclosure.
5. Linear Discriminant Analysis spatial weights and motor channel coefficients.
6. Scientific cohort metric consistency matching Phase 4/5 reports.
"""

import pytest
from fastapi.testclient import TestClient

from web.backend.main import app
from web.backend.schemas import (
    HealthResponse,
    ParticipantListResponse,
    TrialDataResponse,
    EEGSignalResponse,
    CohortResultsResponse,
    LDACoefficientsResponse,
)


@pytest.fixture(scope="module")
def client():
    """Create test client for Phase 6 FastAPI backend."""
    return TestClient(app)


def test_health_check(client):
    """Test health check route returns OK with data source tag."""
    resp = client.get("/api/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["service"] == "neuroaction-backend"
    assert data["data_source"] in ("saved_result", "recorded_replay")


def test_list_participants(client):
    """Test participant listing contains all 5 audited benchmark participants."""
    resp = client.get("/api/participants")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total_participants"] == 5
    expected_subs = ["sub-01", "sub-02", "sub-03", "sub-08", "sub-15"]
    assert sorted(data["participants"]) == expected_subs
    # Check valid trial counts
    details = {d["participant_id"]: d["valid_trials"] for d in data["details"]}
    assert details["sub-01"] == 40
    assert details["sub-02"] == 52
    assert details["sub-03"] == 60
    assert details["sub-08"] == 57
    assert details["sub-15"] == 60


def test_list_trials_for_participant(client):
    """Test listing trials for sub-01 returns 40 valid trials."""
    resp = client.get("/api/trials?participant_id=sub-01")
    assert resp.status_code == 200
    data = resp.json()
    trials = data["trials"]
    assert len(trials) == 40
    assert trials[0]["trial_id"] == "sub-01_run-01_L1_tr00"
    assert trials[0]["congruency"] == "congruent"


def test_trial_data_replay(client):
    """Test retrieving full kinematic and multimodal trial replay."""
    resp = client.get("/api/trial-data?trial_id=sub-01_run-01_L1_tr00")
    assert resp.status_code == 200
    data = resp.json()
    assert data["trial_id"] == "sub-01_run-01_L1_tr00"
    assert len(data["times_ms"]) > 0
    assert len(data["hand_x"]) == len(data["times_ms"])
    assert len(data["hand_y"]) == len(data["times_ms"])
    assert len(data["hand_z"]) == len(data["times_ms"])
    assert data["predicted_endpoint"] is not None
    assert len(data["predicted_endpoint"]) == 3
    assert data["neural"]["predicted_goal"] in ("left", "right")
    assert data["data_source"] == "recorded_replay"
    assert len(data["conflict_timeseries"]) == len(data["times_ms"])


def test_eeg_signal_endpoint(client):
    """Test retrieving real scalp EEG signal for a motor channel."""
    resp = client.get("/api/eeg-signal?trial_id=sub-01_run-01_L1_tr00&channel=C3")
    assert resp.status_code == 200
    data = resp.json()
    assert data["channel"] == "C3"
    assert len(data["amplitude_uv"]) == 200
    assert len(data["times_ms"]) == 200
    assert "filtfilt" in data["processing"].lower()
    assert data["srate_hz"] == 200.0


def test_cohort_metrics_consistency(client):
    """Verify cohort metrics match Phase 4 and Phase 5 reports exactly."""
    resp = client.get("/api/results/cohort")
    assert resp.status_code == 200
    data = resp.json()
    assert data["ridge_endpoint_rmse_mm"] == 47.25
    assert data["fair_cv_rmse_mm"] == 85.65
    assert data["persistence_rmse_mm"] == 177.51
    assert data["ridge_vs_cv_reduction_pct"] == 44.8
    assert data["phase4_balanced_accuracy_pct"] == 66.46
    assert data["phase4_auroc"] == 0.6834
    assert data["phase5_cond_disc_balanced_accuracy_pct"] == 88.35
    assert data["phase5_auroc"] == 0.9395
    assert "NOT biological" in data["phase5_note"]
    assert data["n_valid_trials"] == 269
    assert data["n_participants"] == 5


def test_lda_coefficients(client):
    """Verify LDA spatial decision filter coefficients for motor channels."""
    resp = client.get("/api/lda-coefficients?participant=sub-01")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["channels"]) == 9
    assert "C3" in data["channels"]
    assert "C4" in data["channels"]
    assert len(data["coefficients"]) == 9
