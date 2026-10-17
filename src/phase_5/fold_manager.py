"""Leave-One-Subject-Out (LOSO) fold coordinator enforcing strict participant separation."""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np

from src.phase_4.neural_adapter import NeuralIntentionAdapter
from src.phase_4.kinematic_adapter import KinematicTwinAdapter
from src.trajectory.segmentation import TrajectorySegmenter
from src.phase_5.config import Phase5Config
from src.phase_5.target_reference import TargetReferenceManager, TargetReference
from src.phase_5.provenance import ProvenanceTracker


class LOSOFoldCoordinator:
    """Manages strictly held-out models, scalers, target proxies, and duration estimators per fold."""

    def __init__(self, config: Optional[Phase5Config] = None):
        self.config = config or Phase5Config()
        self.segmenter = TrajectorySegmenter()
        self.target_manager = TargetReferenceManager(default_tolerance_mm=self.config.target_tolerance_mm)
        self._samples_cache: Optional[List[Any]] = None

    def load_cohort_samples(self, mode: str = "kinematic_onset") -> List[Any]:
        """Load and cache valid segmented samples across the cohort."""
        if self._samples_cache is None:
            samples, _ = self.segmenter.load_cohort(
                mode=mode,
                speed_threshold_mm_s=self.config.speed_threshold_mm_s,
            )
            self._samples_cache = samples
        return self._samples_cache

    def get_subject_folds(self) -> List[str]:
        """Return sorted unique participant identifiers."""
        samples = self.load_cohort_samples()
        return sorted(list(set(s.participant_id for s in samples)))

    def get_fold_context(self, test_sub: str) -> Dict[str, Any]:
        """Construct isolated training context and held-out adapters for a test subject.
        
        CRITICAL ANTI-LEAKAGE ENFORCEMENT:
        1. Training samples strictly exclude test_sub.
        2. Neural adapter loaded is lda_goal_decoder_held_out_{test_sub}.pkl.
        3. Kinematic adapter loaded is kinematic_forecaster_held_out_{test_sub}.pkl.
        4. Target references estimated exclusively from training_samples.
        5. Fair baseline remaining duration estimated exclusively from training_samples.
        """
        samples = self.load_cohort_samples()
        train_samples = [s for s in samples if s.participant_id != test_sub]
        test_samples = [s for s in samples if s.participant_id == test_sub]

        # Verify zero leakage
        ProvenanceTracker.verify_provenance_cleanliness(
            training_subjects=[s.participant_id for s in train_samples],
            test_subject=test_sub,
        )

        # 1. Held-Out Neural Adapter
        neural_adapter = NeuralIntentionAdapter.load_held_out(
            participant_id=test_sub,
            model_dir=self.config.model_dir_phase_2,
        )

        # 2. Held-Out Kinematic Adapter
        kinematic_adapter = KinematicTwinAdapter.load_held_out(
            participant_id=test_sub,
            model_dir=self.config.model_dir_phase_3,
        )

        # 3. Fold-Specific Target References (centroid proxies)
        target_refs = self.target_manager.get_fold_target_references(
            training_samples=train_samples,
            held_out_subject=test_sub,
            tolerance_mm=self.config.target_tolerance_mm,
        )

        # 4. Fair Training-Duration Estimator (seconds)
        train_rem_durs = [(s.movement_duration_ms - self.config.prefix_duration_ms) / 1000.0 for s in train_samples]
        dt_train_mean_s = float(np.mean(train_rem_durs))

        return {
            "test_subject": test_sub,
            "training_subjects": sorted(list(set(s.participant_id for s in train_samples))),
            "train_samples": train_samples,
            "test_samples": test_samples,
            "neural_adapter": neural_adapter,
            "kinematic_adapter": kinematic_adapter,
            "target_references": target_refs,
            "dt_train_mean_s": dt_train_mean_s,
        }
