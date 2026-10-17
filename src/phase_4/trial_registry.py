"""Trial registry, data validation, and multi-subject trial cache for Phase 4."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd

from src.data.openneuro_client import OpenNeuroClient
from src.data.schema import CanonicalTrial
from src.data.trial_loader import TrialLoader
from src.phase_4.coordinate_mapping import get_target_coordinates
from src.phase_4.schemas import TrialMetadata


class TrialRegistry:
    """Manages trial discovery, validity filtering, and metadata extraction."""

    def __init__(
        self,
        manifest_path: Path = Path("data/manifests/iackd_manifest.csv"),
        loader: Optional[TrialLoader] = None,
    ):
        self.manifest_path = manifest_path
        self.loader = loader or TrialLoader()
        self._manifest_df: Optional[pd.DataFrame] = None
        self._trial_cache: Dict[str, CanonicalTrial] = {}
        self._metadata_cache: Dict[str, TrialMetadata] = {}

    @property
    def manifest(self) -> pd.DataFrame:
        """Lazy load and cache manifest DataFrame."""
        if self._manifest_df is None:
            if not self.manifest_path.exists():
                raise FileNotFoundError(f"Manifest not found at {self.manifest_path}")
            self._manifest_df = pd.read_csv(self.manifest_path)
        return self._manifest_df

    def list_participants(self) -> List[str]:
        """List sorted unique participant IDs."""
        valid_df = self.manifest[self.manifest["valid"]]
        return sorted(list(valid_df["participant_id"].unique()))

    def list_runs(self, participant_id: str) -> List[str]:
        """List sorted unique run IDs for a given participant."""
        sub_df = self.manifest[self.manifest["participant_id"] == participant_id]
        return sorted(list(sub_df["run_id"].unique()))

    def list_trials(
        self,
        participant_id: Optional[str] = None,
        valid_only: bool = True,
    ) -> List[str]:
        """List trial IDs matching search criteria."""
        df = self.manifest
        if valid_only:
            df = df[df["valid"]]
        if participant_id:
            df = df[df["participant_id"] == participant_id]
        return list(df["trial_id"].values)

    def validate_trial(self, trial: CanonicalTrial) -> Tuple[bool, Optional[str]]:
        """Validate trial data integrity for multimodal digital twin processing."""
        if not trial.valid:
            return False, "Marked invalid in dataset manifest."
        if trial.eeg_pre is None or len(trial.eeg_pre) < 200:
            return False, "Pre-movement EEG is missing or has fewer than 200 samples."
        if trial.hand_x is None or len(trial.hand_x) < 30:
            return False, "Kinematics stream is missing or has fewer than 30 samples."
        if np.isnan(trial.hand_x[:20]).any():
            return False, "Initial kinematic prefix contains NaN values."
        return True, None

    def get_trial(self, trial_id: str) -> Optional[CanonicalTrial]:
        """Load CanonicalTrial by trial_id, utilizing in-memory cache."""
        if trial_id in self._trial_cache:
            return self._trial_cache[trial_id]

        rows = self.manifest[self.manifest["trial_id"] == trial_id]
        if len(rows) == 0:
            return None

        row = rows.iloc[0]
        sub = row["participant_id"]
        run = row["run_id"]

        # Load run trials and cache them
        run_trials = self.loader.load_run(sub, run)
        for t in run_trials:
            self._trial_cache[t.trial_id] = t

        return self._trial_cache.get(trial_id)

    def get_metadata(self, trial_id: str) -> Optional[TrialMetadata]:
        """Get or build typed TrialMetadata for a trial."""
        if trial_id in self._metadata_cache:
            return self._metadata_cache[trial_id]

        trial = self.get_trial(trial_id)
        if trial is None:
            return None

        is_valid, reason = self.validate_trial(trial)
        target_coords = get_target_coordinates(trial.target_goal, allow_unmapped=True)

        meta = TrialMetadata(
            participant_id=trial.participant_id,
            run_id=trial.run_id,
            trial_id=trial.trial_id,
            trial_index=trial.trial_index,
            label_code=trial.label_code,
            congruency=trial.congruency,
            target_goal=trial.target_goal,
            physical_hand_direction=trial.physical_hand_direction,
            ball_color=trial.ball_color,
            valid=is_valid,
            exclusion_reason=reason,
            go_cue_time_ms=0.0,
            kinematic_onset_time_ms=None,
            movement_duration_ms=float(trial.times_me[-1] - trial.times_me[0]) if trial.times_me is not None and len(trial.times_me) > 0 else None,
            target_coordinates=target_coords,
            coordinate_units="mm",
            coordinate_frame="leap_motion_cartesian",
            eeg_sample_range_ms=(-1000.0, 0.0),
            kinematic_sample_count=len(trial.hand_x) if trial.hand_x is not None else 0,
        )
        self._metadata_cache[trial_id] = meta
        return meta

    def get_qc_summary(self) -> Dict[str, Any]:
        """Generate high-level cohort quality control summary."""
        total_indexed = len(self.manifest)
        valid_count = int(self.manifest["valid"].sum())
        return {
            "total_indexed_trials": total_indexed,
            "valid_trials": valid_count,
            "excluded_trials": total_indexed - valid_count,
            "subjects": self.list_participants(),
            "manifest_path": str(self.manifest_path),
        }
