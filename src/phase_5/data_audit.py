"""Systematic trial audit assessing measurement availability and task eligibility."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd

from src.data.trial_loader import TrialLoader
from src.phase_5.config import Phase5Config
from src.phase_5.schemas import EligibilityRecord


from src.trajectory.segmentation import TrajectorySegmenter


class DataAuditor:
    """Audits multimodal trials and constructs machine-readable eligibility matrices."""

    def __init__(
        self,
        config: Optional[Phase5Config] = None,
        loader: Optional[TrialLoader] = None,
    ):
        self.config = config or Phase5Config()
        self.loader = loader or TrialLoader()
        self.segmenter = TrajectorySegmenter(loader=self.loader)
        self._manifest_df: Optional[pd.DataFrame] = None

    @property
    def manifest(self) -> pd.DataFrame:
        if self._manifest_df is None:
            if not self.config.manifest_path.exists():
                raise FileNotFoundError(f"Manifest not found at {self.config.manifest_path}")
            self._manifest_df = pd.read_csv(self.config.manifest_path)
        return self._manifest_df

    def audit_all_trials(self) -> List[EligibilityRecord]:
        """Audit all 280 trials indexed in the dataset manifest."""
        records: List[EligibilityRecord] = []
        df = self.manifest

        for _, row in df.iterrows():
            trial_id = str(row["trial_id"])
            sub = str(row["participant_id"])
            run = str(row["run_id"])
            is_valid_manifest = bool(row["valid"])

            reasons: List[str] = []

            # 1. Pre-EEG Availability
            has_pre_eeg = bool(row.get("pre_eeg_available", False))
            if not has_pre_eeg:
                reasons.append("Missing pre-movement EEG segment.")

            # 2. Kinematics Availability & Prefix
            has_kinematics = bool(row.get("kinematics_available", False))
            has_valid_prefix = False
            has_valid_endpoint = False

            if has_kinematics and is_valid_manifest:
                try:
                    trial = self.loader.load_trial(sub, run, int(row["trial_index"]))
                    if trial and trial.valid:
                        sample = self.segmenter.process_trial(trial, mode="kinematic_onset")
                        if sample is not None:
                            if not np.isnan(sample.prefix_positions).any():
                                has_valid_prefix = True
                            if not np.isnan(sample.actual_endpoint).any():
                                has_valid_endpoint = True
                except Exception as e:
                    reasons.append(f"Kinematic load failure: {str(e)}")

            if not has_valid_prefix:
                reasons.append("Invalid or incomplete 200 ms kinematic prefix.")
            if not has_valid_endpoint:
                reasons.append("Invalid or missing actual physical endpoint.")

            # 3. Independent Target Coordinates (Crucial Audit Point)
            # OpenNeuro ds006840 DOES NOT contain independent physical target coordinates
            has_indep_target = False
            has_indep_outcome = False
            is_authoritative = False
            reasons.append("OpenNeuro ds006840 lacks independent 3D physical target coordinates (exploratory proxy centroids only).")

            # 4. Task Eligibility Determinations
            eligible_a = is_valid_manifest and has_pre_eeg
            eligible_b = is_valid_manifest and has_valid_prefix and has_valid_endpoint
            eligible_c = eligible_b  # Eligible for exploratory proxy target forecasting
            eligible_d = eligible_a  # Neural-only exploratory proxy
            eligible_e = eligible_b  # Kinematic-only exploratory proxy
            eligible_f = eligible_a and eligible_b  # Multimodal exploratory proxy

            rec = EligibilityRecord(
                trial_id=trial_id,
                participant_id=sub,
                run_id=run,
                has_valid_pre_eeg=has_pre_eeg,
                has_valid_prefix=has_valid_prefix,
                has_valid_endpoint=has_valid_endpoint,
                has_independent_target_coordinates=has_indep_target,
                has_independent_outcome_label=has_indep_outcome,
                is_authoritative_ground_truth=is_authoritative,
                eligible_task_a_neural=eligible_a,
                eligible_task_b_kinematic_forecast=eligible_b,
                eligible_task_c_target_forecast=eligible_c,
                eligible_task_d_neural_outcome=eligible_d,
                eligible_task_e_kinematic_outcome=eligible_e,
                eligible_task_f_fusion_outcome=eligible_f,
                unsupported_reasons=reasons,
            )
            records.append(rec)

        return records

    def build_eligibility_dataframe(self) -> pd.DataFrame:
        """Construct a tabular eligibility summary DataFrame."""
        records = self.audit_all_trials()
        rows = [r.to_dict() for r in records]
        df = pd.DataFrame(rows)
        return df

    def get_summary_statistics(self) -> Dict[str, Any]:
        """Compute aggregate cohort statistics across tasks."""
        df = self.build_eligibility_dataframe()
        return {
            "total_indexed_trials": len(df),
            "valid_pre_eeg_trials": int(df["has_valid_pre_eeg"].sum()),
            "valid_kinematic_prefix_trials": int(df["has_valid_prefix"].sum()),
            "valid_endpoint_trials": int(df["has_valid_endpoint"].sum()),
            "independent_target_coords_available": int(df["has_independent_target_coordinates"].sum()),
            "authoritative_ground_truth_available": int(df["is_authoritative_ground_truth"].sum()),
            "eligible_task_a_neural": int(df["eligible_task_a_neural"].sum()),
            "eligible_task_b_kinematic_forecast": int(df["eligible_task_b_kinematic_forecast"].sum()),
            "eligible_task_c_exploratory_target": int(df["eligible_task_c_target_forecast"].sum()),
            "eligible_task_f_multimodal_fusion": int(df["eligible_task_f_fusion_outcome"].sum()),
            "per_participant_breakdown": df.groupby("participant_id")["eligible_task_f_fusion_outcome"].sum().to_dict(),
        }
