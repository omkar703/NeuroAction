"""Trial service wrapping Phase 4 TrialRegistry and dataset manifest."""

from typing import Any, Dict, List, Optional
import pandas as pd

from src.data.schema import CanonicalTrial
from src.phase_4.schemas import TrialMetadata
from src.phase_4.trial_registry import TrialRegistry
from web.backend.schemas import ParticipantSummary, TrialSummary


class TrialService:
    """Provides high-level query access to trial metadata and canonical trials."""

    def __init__(self, registry: Optional[TrialRegistry] = None):
        self.registry = registry or TrialRegistry()

    def list_participants(self) -> List[str]:
        """Return sorted list of valid participant IDs."""
        return self.registry.list_participants()

    def get_participants_summary(self) -> List[ParticipantSummary]:
        """Compute aggregated trial statistics for all participants."""
        df = self.registry.manifest
        participants = sorted(list(df["participant_id"].unique()))
        summaries = []

        for pid in participants:
            sub_df = df[df["participant_id"] == pid]
            total_trials = len(sub_df)
            valid_trials = int(sub_df["valid"].sum())
            runs = sorted(list(sub_df["run_id"].unique()))
            summaries.append(
                ParticipantSummary(
                    participant_id=pid,
                    total_trials=total_trials,
                    valid_trials=valid_trials,
                    runs=runs,
                )
            )
        return summaries

    def list_trials(
        self,
        participant_id: Optional[str] = None,
        valid_only: bool = True,
    ) -> List[TrialSummary]:
        """Return filtered trial metadata summaries."""
        df = self.registry.manifest
        if valid_only:
            df = df[df["valid"]]
        if participant_id:
            df = df[df["participant_id"] == participant_id]

        trials: List[TrialSummary] = []
        for _, row in df.iterrows():
            duration_ms = None
            if pd.notna(row.get("execution_samples")):
                duration_ms = float(row["execution_samples"]) * 10.0  # 100 Hz sampling

            exclusion = None if bool(row.get("valid", True)) else "Invalid tracking or insufficient prefix"

            trials.append(
                TrialSummary(
                    trial_id=str(row["trial_id"]),
                    participant_id=str(row["participant_id"]),
                    run_id=str(row["run_id"]),
                    trial_index=int(row["trial_index"]),
                    label_code=int(row["label_code"]),
                    congruency=str(row["congruency"]),
                    target_goal=str(row["target_goal"]),
                    physical_hand_direction=str(row["physical_hand_direction"]),
                    ball_color=str(row["ball_color"]),
                    valid=bool(row["valid"]),
                    movement_duration_ms=duration_ms,
                    exclusion_reason=exclusion,
                )
            )
        return trials

    def get_trial(self, trial_id: str) -> Optional[CanonicalTrial]:
        """Retrieve full CanonicalTrial object from registry cache."""
        return self.registry.get_trial(trial_id)

    def get_metadata(self, trial_id: str) -> Optional[TrialMetadata]:
        """Retrieve TrialMetadata object from registry."""
        return self.registry.get_metadata(trial_id)


# Global singleton instance
_trial_service: Optional[TrialService] = None


def get_trial_service() -> TrialService:
    """Dependency injection provider for TrialService."""
    global _trial_service
    if _trial_service is None:
        _trial_service = TrialService()
    return _trial_service
