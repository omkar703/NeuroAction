"""Results and explainability service loading evaluation metrics and model coefficients."""

from pathlib import Path
from typing import Any, Dict, List, Optional
import pickle
import numpy as np

from src.features.pre_eeg import MOTOR_FOCUSED_CHANNELS
from web.backend.schemas import (
    CohortResultsResponse,
    FeaturesInfoResponse,
    LDACoefficientsResponse,
    ModelInfoResponse,
    ParticipantResultsResponse,
)


# Canonical 26 scalp channels from OpenNeuro ds006840
ALL_26_CHANNELS = [
    "FP1", "FP2", "F7", "F3", "FZ", "F4", "F8",
    "FC3", "FCZ", "FC4", "T7", "C3", "CZ", "C4", "T8",
    "CP3", "CPZ", "CP4", "P7", "P3", "PZ", "P4", "P8",
    "O1", "OZ", "O2"
]

# Audited participant benchmark table from Phase 5 prospective validation report (Section 6)
PARTICIPANT_BENCHMARKS: Dict[str, Dict[str, Any]] = {
    "sub-01": {
        "valid_trials": 40,
        "persistence_rmse_mm": 181.48,
        "fair_cv_rmse_mm": 109.30,
        "ridge_forecaster_rmse_mm": 41.78,
        "ridge_reduction_vs_fair_cv_pct": 61.8,
        "neural_balanced_accuracy_pct": 72.50,
        "condition_balanced_accuracy_pct": 90.00,
        "policy_balanced_accuracy_pct": 95.00,
        "policy_coverage_pct": 70.00,
    },
    "sub-02": {
        "valid_trials": 52,
        "persistence_rmse_mm": 178.36,
        "fair_cv_rmse_mm": 110.30,
        "ridge_forecaster_rmse_mm": 59.90,
        "ridge_reduction_vs_fair_cv_pct": 45.7,
        "neural_balanced_accuracy_pct": 73.08,
        "condition_balanced_accuracy_pct": 92.15,
        "policy_balanced_accuracy_pct": 98.08,
        "policy_coverage_pct": 88.46,
    },
    "sub-03": {
        "valid_trials": 60,
        "persistence_rmse_mm": 176.29,
        "fair_cv_rmse_mm": 38.08,
        "ridge_forecaster_rmse_mm": 28.71,
        "ridge_reduction_vs_fair_cv_pct": 24.6,
        "neural_balanced_accuracy_pct": 86.67,
        "condition_balanced_accuracy_pct": 100.00,
        "policy_balanced_accuracy_pct": 100.00,
        "policy_coverage_pct": 76.67,
    },
    "sub-08": {
        "valid_trials": 57,
        "persistence_rmse_mm": 178.34,
        "fair_cv_rmse_mm": 75.14,
        "ridge_forecaster_rmse_mm": 25.20,
        "ridge_reduction_vs_fair_cv_pct": 66.5,
        "neural_balanced_accuracy_pct": 53.57,
        "condition_balanced_accuracy_pct": 92.86,
        "policy_balanced_accuracy_pct": 97.92,
        "policy_coverage_pct": 92.98,
    },
    "sub-15": {
        "valid_trials": 60,
        "persistence_rmse_mm": 174.49,
        "fair_cv_rmse_mm": 86.97,
        "ridge_forecaster_rmse_mm": 65.63,
        "ridge_reduction_vs_fair_cv_pct": 24.5,
        "neural_balanced_accuracy_pct": 56.67,
        "condition_balanced_accuracy_pct": 68.33,
        "policy_balanced_accuracy_pct": 55.26,
        "policy_coverage_pct": 80.00,
    },
}


class ResultsService:
    """Service providing audited benchmark metrics, explainability, and model coefficients."""

    def __init__(self, models_dir: Path = Path("models")):
        self.models_dir = models_dir

    def get_cohort_results(self) -> CohortResultsResponse:
        """Return verified cohort benchmark metrics from Phase 4 and Phase 5 reports."""
        return CohortResultsResponse(
            ridge_endpoint_rmse_mm=47.25,
            fair_cv_rmse_mm=85.65,
            persistence_rmse_mm=177.51,
            ridge_vs_cv_reduction_pct=44.8,
            phase4_balanced_accuracy_pct=66.46,
            phase4_auroc=0.6834,
            phase5_cond_disc_balanced_accuracy_pct=88.35,
            phase5_auroc=0.9395,
            phase5_note=(
                "Measures experimental condition discrimination (congruent vs. incongruent), "
                "NOT biological intention-action discordance"
            ),
            n_valid_trials=269,
            n_participants=5,
            data_source="saved_result",
        )

    def get_participant_results(self, sub_id: str) -> ParticipantResultsResponse:
        """Return audited participant-level metrics from Phase 5 LOSO evaluation."""
        norm_sub = sub_id.strip().lower()
        if not norm_sub.startswith("sub-") and norm_sub.startswith("sub"):
            norm_sub = f"sub-{norm_sub[3:]}"

        if norm_sub not in PARTICIPANT_BENCHMARKS:
            available = list(PARTICIPANT_BENCHMARKS.keys())
            raise ValueError(
                f"Participant '{sub_id}' not found in benchmark records. Available participants: {available}"
            )

        data = PARTICIPANT_BENCHMARKS[norm_sub]
        return ParticipantResultsResponse(
            participant_id=norm_sub,
            valid_trials=data["valid_trials"],
            persistence_rmse_mm=data["persistence_rmse_mm"],
            fair_cv_rmse_mm=data["fair_cv_rmse_mm"],
            ridge_forecaster_rmse_mm=data["ridge_forecaster_rmse_mm"],
            ridge_reduction_vs_fair_cv_pct=data["ridge_reduction_vs_fair_cv_pct"],
            neural_balanced_accuracy_pct=data["neural_balanced_accuracy_pct"],
            condition_balanced_accuracy_pct=data["condition_balanced_accuracy_pct"],
            policy_balanced_accuracy_pct=data["policy_balanced_accuracy_pct"],
            policy_coverage_pct=data["policy_coverage_pct"],
            data_source="saved_result",
        )

    def get_lda_coefficients(self, participant_id: Optional[str] = None) -> LDACoefficientsResponse:
        """Extract spatial filter coefficients (L2 magnitude across 9 feature groups) for motor channels."""
        phase2_dir = self.models_dir / "phase_2"

        if participant_id and participant_id.lower() not in ("cohort", "all", "none"):
            norm_sub = participant_id.replace("-", "").lower()
            model_file = phase2_dir / f"lda_goal_decoder_held_out_{norm_sub}.pkl"
            target_pid = participant_id
            model_name = f"lda_goal_decoder_held_out_{norm_sub}"
        else:
            model_file = phase2_dir / "lda_goal_decoder.pkl"
            target_pid = "cohort"
            model_name = "lda_goal_decoder_cohort"

        if not model_file.exists():
            raise FileNotFoundError(f"Model checkpoint not found at {model_file}")

        with open(model_file, "rb") as f:
            decoder = pickle.load(f)

        clf = decoder.pipe.named_steps["clf"]
        coef = clf.coef_[0]  # shape (234,)

        # Compute signed L2 norm for each motor-focused channel across its 9 feature dimensions
        coefficients = []
        n_all = len(ALL_26_CHANNELS)
        for ch in MOTOR_FOCUSED_CHANNELS:
            idx = ALL_26_CHANNELS.index(ch)
            ch_feat_indices = [g * n_all + idx for g in range(9)]
            ch_weights = coef[ch_feat_indices]
            sign = 1.0 if float(np.sum(ch_weights)) >= 0 else -1.0
            signed_norm = float(sign * np.linalg.norm(ch_weights))
            coefficients.append(round(signed_norm, 4))

        interpretation = (
            "Linear Discriminant Analysis spatial filter weights mapped over the primary motor and premotor cortex "
            "(FC3/FCz/FC4, C3/Cz/C4, CP3/CPz/CP4). Central sensorimotor channels reflect readiness potential "
            "lateralization and mu/beta desynchronization (ERD) preceding reaching execution."
        )

        return LDACoefficientsResponse(
            participant_id=target_pid,
            model_name=model_name,
            channels=MOTOR_FOCUSED_CHANNELS,
            coefficients=coefficients,
            interpretation=interpretation,
            data_source="saved_result",
        )

    def get_model_info(self) -> ModelInfoResponse:
        """Return structural and mathematical specification of all models in the digital twin."""
        return ModelInfoResponse(
            neural_model={
                "name": "Linear Discriminant Analysis with Shrinkage (LDADecoder)",
                "features": "Time-domain summary statistics (mean, var, rms, slope, min, max) + mu/beta bandpower (234 features)",
                "input_window": "[-1000, 0] ms relative to movement onset",
                "sampling_rate_hz": 200.0,
                "targets": ["left", "right"],
                "balanced_accuracy_pct": 68.34,
                "auroc": 0.6834,
            },
            kinematic_model={
                "name": "Regularized Ridge Regression Forecaster",
                "input_prefix_ms": 200.0,
                "input_samples": 20,
                "sampling_rate_hz": 100.0,
                "predicted_outputs": "Terminal 3D endpoint [x, y, z] mm and multi-horizon paths [100, 200, 300, 500 ms]",
                "endpoint_rmse_mm": 47.25,
                "reduction_vs_fair_cv_pct": 44.8,
            },
            fusion_and_conflict_engine={
                "name": "Multimodal Fusion and Conflict Assessment Engine",
                "policy_states": [
                    "ALIGNED",
                    "POTENTIAL_DISAGREEMENT",
                    "UNCERTAIN",
                    "UNAVAILABLE",
                    "INVALID_INPUT",
                ],
                "distance_threshold_mm": 50.0,
                "directional_threshold_deg": 45.0,
                "confidence_uncertainty_threshold": 0.55,
                "distance_reference_mm": 100.0,
            },
            validation_regime="Strict Leave-One-Subject-Out (LOSO) Cross-Validation with Zero Subject Leakage",
            coordinate_reference={
                "frame": "leap_motion_cartesian",
                "units": "mm",
                "target_left_mm": [-188.0, 150.0, 53.0],
                "target_right_mm": [188.0, 149.0, 51.0],
            },
            data_source="saved_result",
        )

    def get_features_info(self) -> FeaturesInfoResponse:
        """Return details of the neural and kinematic feature extraction pipelines."""
        return FeaturesInfoResponse(
            motor_channels=MOTOR_FOCUSED_CHANNELS,
            all_channels=ALL_26_CHANNELS,
            total_channels=len(ALL_26_CHANNELS),
            eeg_sampling_rate_hz=200.0,
            eeg_window_ms=[-1000.0, 0.0],
            time_domain_features=["mean", "variance", "rms", "slope", "min", "max"],
            frequency_band_features=["mu_power_8_13Hz", "beta_power_13_30Hz", "mu_ratio"],
            total_features_per_channel=9,
            total_neural_features=len(ALL_26_CHANNELS) * 9,
            kinematic_sampling_rate_hz=100.0,
            kinematic_prefix_ms=200.0,
            kinematic_prefix_samples=20,
            coordinate_frame="leap_motion_cartesian",
            coordinate_units="mm",
            targets={
                "target_left": [-188.0, 150.0, 53.0],
                "target_right": [188.0, 149.0, 51.0],
            },
            data_source="saved_result",
        )


# Global singleton instance
_results_service: Optional[ResultsService] = None


def get_results_service() -> ResultsService:
    """Dependency injection provider for ResultsService."""
    global _results_service
    if _results_service is None:
        _results_service = ResultsService()
    return _results_service
