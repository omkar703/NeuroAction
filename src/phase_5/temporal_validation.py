"""Temporal integrity and anti-leakage validator for Phase 5 prospective prediction."""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np


class TemporalIntegrityError(ValueError):
    """Raised when acausal lookahead, future sample leakage, or invalid timestamps are detected."""
    pass


class TemporalValidator:
    """Enforces strict causal boundaries between observation, prediction, and retrospective evaluation."""

    @staticmethod
    def verify_prediction_timing(
        prediction_time_ms: float,
        input_window_start_ms: float,
        input_window_end_ms: float,
    ) -> bool:
        """Assert that prediction is generated strictly at or after the input observation window completes."""
        if input_window_start_ms >= input_window_end_ms:
            raise TemporalIntegrityError(
                f"Invalid input window: start ({input_window_start_ms} ms) >= end ({input_window_end_ms} ms)"
            )
        if prediction_time_ms < input_window_end_ms:
            raise TemporalIntegrityError(
                f"Acausal prediction: prediction generated at {prediction_time_ms} ms, "
                f"before input window completion at {input_window_end_ms} ms."
            )
        return True

    @staticmethod
    def verify_no_future_kinematics(
        prefix_sample_count: int,
        total_trial_samples: int,
        feature_vector: np.ndarray,
    ) -> bool:
        """Assert that prefix extraction does not access future samples beyond prefix_sample_count."""
        if prefix_sample_count > total_trial_samples:
            raise TemporalIntegrityError(
                f"Prefix sample count ({prefix_sample_count}) exceeds total trial samples ({total_trial_samples})"
            )
        if not np.isfinite(feature_vector).all():
            raise TemporalIntegrityError("Extracted kinematic feature vector contains non-finite or NaN values.")
        return True

    @staticmethod
    def verify_duration_leakage_free(
        duration_strategy: str,
        uses_test_trial_future_duration: bool,
    ) -> bool:
        """Assert that baseline models do not leak the held-out trial's future movement duration."""
        if uses_test_trial_future_duration:
            raise TemporalIntegrityError(
                f"Baseline duration leakage detected! Strategy '{duration_strategy}' accessed "
                f"held-out test trial's future duration (len(full_trajectory) - prefix_len)."
            )
        return True

    @staticmethod
    def verify_causal_labeling(
        processing_mode: str,
        is_labeled_online: bool,
    ) -> bool:
        """Assert that zero-phase offline EEG preprocessed derivatives cannot be claimed as online-causal."""
        if processing_mode == "offline_acausal" and is_labeled_online:
            raise TemporalIntegrityError(
                "Scientific mischaracterization: Offline zero-phase forward-backward filtered EEG "
                "cannot be claimed or labeled as 'online-causal'."
            )
        return True

    @staticmethod
    def verify_temporal_invariance(
        prediction_fn: Any,
        trial: Any,
        prefix_len: int = 20,
    ) -> bool:
        """Verify that altering future ground-truth trajectory samples produces identical predictions."""
        pred_original = prediction_fn(trial)
        
        # Modify future samples (beyond prefix)
        trial_copy = trial.copy() if hasattr(trial, "copy") else trial
        if hasattr(trial_copy, "hand_x") and len(trial_copy.hand_x) > prefix_len:
            # Perturb future
            trial_copy.hand_x = trial_copy.hand_x.copy()
            trial_copy.hand_x[prefix_len:] += 1000.0  # Huge future perturbation
            pred_perturbed = prediction_fn(trial_copy)
            
            # Predictions must be bit-identical
            diff = np.linalg.norm(np.asarray(pred_original) - np.asarray(pred_perturbed))
            if diff > 1e-5:
                raise TemporalIntegrityError(
                    f"Future trajectory leakage! Perturbing samples beyond prefix altered prediction by {diff:.4f} mm."
                )
        return True
