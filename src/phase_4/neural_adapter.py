"""Adapter for Phase 2 Pre-movement EEG Neural Intention Decoders."""

from pathlib import Path
from typing import Dict, Optional, Union
import pickle
import numpy as np

from src.data.schema import CanonicalTrial
from src.features.pre_eeg import extract_trial_features
from src.models.decoders import LDADecoder, LogisticRegressionDecoder
from src.phase_4.schemas import NeuralIntentionResult


class NeuralIntentionAdapter:
    """Production runtime adapter wrapping trained Phase 2 decoders."""

    def __init__(
        self,
        model: Optional[Union[LDADecoder, LogisticRegressionDecoder]] = None,
        model_name: str = "lda_goal_decoder",
        is_calibrated: bool = False,
        is_held_out: bool = False,
        held_out_participant: str = "",
    ):
        self.model = model
        self.model_name = model_name
        self.is_calibrated = is_calibrated
        self.is_held_out = is_held_out
        self.held_out_participant = held_out_participant

    @classmethod
    def load(
        cls,
        model_dir: Path = Path("models/phase_2"),
        filename: str = "lda_goal_decoder.pkl",
    ) -> "NeuralIntentionAdapter":
        """Load trained Phase 2 model artifact from disk."""
        model_path = model_dir / filename
        if not model_path.exists():
            raise FileNotFoundError(
                f"Phase 2 model checkpoint not found at {model_path}. "
                f"Ensure models are trained or present in models/phase_2/."
            )
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        model_name = filename.replace(".pkl", "")
        return cls(
            model=model,
            model_name=model_name,
            is_calibrated=False,
            is_held_out=False,
            held_out_participant="",
        )

    @classmethod
    def load_held_out(
        cls,
        participant_id: str = "sub-01",
        model_dir: Path = Path("models/phase_2"),
    ) -> "NeuralIntentionAdapter":
        """Load honest held-out model trained strictly without the target participant."""
        norm_sub = participant_id.replace("-", "")
        held_out_filename = f"lda_goal_decoder_held_out_{norm_sub}.pkl"
        held_out_path = model_dir / held_out_filename
        if held_out_path.exists():
            with open(held_out_path, "rb") as f:
                model = pickle.load(f)
            return cls(
                model=model,
                model_name=held_out_filename.replace(".pkl", ""),
                is_calibrated=False,
                is_held_out=True,
                held_out_participant=participant_id,
            )
        # Fallback to standard cohort model if held-out artifact is not present
        return cls.load(model_dir=model_dir, filename="lda_goal_decoder.pkl")

    def predict_trial(
        self,
        trial: CanonicalTrial,
        current_time_ms: float = 0.0,
    ) -> NeuralIntentionResult:
        """Run neural intention inference on a CanonicalTrial's pre-movement EEG.
        
        Args:
            trial: Multimodal CanonicalTrial with pre-movement EEG
            current_time_ms: Replay clock time (prediction only stamped valid if clock >= 0.0)
        Returns:
            NeuralIntentionResult typed schema
        """
        if self.model is None:
            return NeuralIntentionResult(
                participant_id=trial.participant_id,
                trial_id=trial.trial_id,
                model_name=self.model_name,
                prediction_time_ms=0.0,
                window_start_ms=-1000.0,
                window_end_ms=0.0,
                predicted_goal="unknown",
                probability_left=0.5,
                probability_right=0.5,
                confidence=0.5,
                status="UNAVAILABLE",
                error_message="Phase 2 model is not initialized or loaded.",
                is_temporally_valid=False,
            )

        if trial.eeg_pre is None or len(trial.eeg_pre) < 200:
            return NeuralIntentionResult(
                participant_id=trial.participant_id,
                trial_id=trial.trial_id,
                model_name=self.model_name,
                prediction_time_ms=0.0,
                window_start_ms=-1000.0,
                window_end_ms=0.0,
                predicted_goal="unknown",
                probability_left=0.5,
                probability_right=0.5,
                confidence=0.5,
                status="INVALID_INPUT",
                error_message="Pre-movement EEG is missing or shorter than 200 samples (-1000 to 0 ms).",
                is_temporally_valid=False,
            )

        if np.isnan(trial.eeg_pre).any():
            return NeuralIntentionResult(
                participant_id=trial.participant_id,
                trial_id=trial.trial_id,
                model_name=self.model_name,
                prediction_time_ms=0.0,
                window_start_ms=-1000.0,
                window_end_ms=0.0,
                predicted_goal="unknown",
                probability_left=0.5,
                probability_right=0.5,
                confidence=0.5,
                status="INVALID_INPUT",
                error_message="Pre-movement EEG contains NaN values.",
                is_temporally_valid=False,
            )

        try:
            # Extract combined time-domain and spectral bandpower features
            seg = trial.eeg_pre[:200, :]  # shape: (200, n_channels)
            srate = float(trial.srate_pre) if trial.srate_pre else 200.0
            feat_1d = extract_trial_features(seg, feature_type="combined", srate=srate)
            X = feat_1d.reshape(1, -1).astype(np.float32)

            probs = self.model.predict_proba(X)[0]
            pred_idx = int(self.model.predict(X)[0])

            p_left = float(probs[0])
            p_right = float(probs[1])
            pred_goal = "right" if pred_idx == 1 else "left"
            conf = max(p_left, p_right)

            # Temporal validity check: prediction becomes causally available at t = 0.0 ms
            is_valid_clock = current_time_ms >= 0.0

            return NeuralIntentionResult(
                participant_id=trial.participant_id,
                trial_id=trial.trial_id,
                model_name=self.model_name,
                prediction_time_ms=0.0,
                window_start_ms=-1000.0,
                window_end_ms=0.0,
                predicted_goal=pred_goal,
                probability_left=p_left,
                probability_right=p_right,
                confidence=conf,
                score_type="posterior_probability",
                is_calibrated=self.is_calibrated,
                status="AVAILABLE",
                error_message=None,
                is_temporally_valid=is_valid_clock,
                processing_mode="offline_acausal",
            )
        except Exception as exc:
            return NeuralIntentionResult(
                participant_id=trial.participant_id,
                trial_id=trial.trial_id,
                model_name=self.model_name,
                prediction_time_ms=0.0,
                window_start_ms=-1000.0,
                window_end_ms=0.0,
                predicted_goal="unknown",
                probability_left=0.5,
                probability_right=0.5,
                confidence=0.5,
                status="ERROR",
                error_message=f"Neural feature extraction or prediction error: {str(exc)}",
                is_temporally_valid=False,
            )
