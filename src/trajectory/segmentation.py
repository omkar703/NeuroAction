"""Data processing, movement onset segmentation, and horizon alignment for kinematics."""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from src.data.schema import CanonicalTrial
from src.data.trial_loader import TrialLoader
from src.trajectory.schema import TrajectorySample


class TrajectorySegmenter:
    """Extracts standardized 200 ms movement prefixes, multi-horizon targets, and final endpoints."""

    HORIZONS_MS = [100, 200, 300, 500]  # ms after prefix end
    PREFIX_SAMPLES = 20                 # 200 ms at 100 Hz
    DT_S = 0.01                         # 10 ms per sample

    def __init__(self, loader: Optional[TrialLoader] = None):
        self.loader = loader or TrialLoader()

    def process_trial(
        self,
        trial: CanonicalTrial,
        mode: str = "event_locked",
        speed_threshold_mm_s: float = 20.0,
    ) -> Optional[TrajectorySample]:
        """Segment a CanonicalTrial into a clean TrajectorySample.
        
        Args:
            trial: CanonicalTrial to segment.
            mode: 'event_locked' (starts at event marker t >= 0) or
                  'kinematic_onset' (starts at physical movement onset where speed >= speed_threshold).
            speed_threshold_mm_s: Velocity threshold in mm/s to declare physical kinematic movement onset.
        Returns:
            TrajectorySample or None if trial fails quality constraints.
        """
        if not trial.valid or trial.hand_x is None or trial.hand_y is None or trial.hand_z is None:
            return None

        times = trial.times_me
        if len(times) < self.PREFIX_SAMPLES + 10:
            return None

        # Event reference is nominal t >= 0
        event_indices = np.where(times >= 0.0)[0]
        if len(event_indices) < self.PREFIX_SAMPLES:
            return None
        idx_event = event_indices[0]

        if mode == "event_locked":
            idx_onset = idx_event
            onset_latency_ms = 0.0
        elif mode == "kinematic_onset":
            # Search forward from idx_event for physical movement initiation
            pos_post = np.stack([
                trial.hand_x[idx_event:],
                trial.hand_y[idx_event:],
                trial.hand_z[idx_event:]
            ], axis=1)
            # Instantaneous speed
            speeds = np.zeros(len(pos_post), dtype=np.float32)
            speeds[1:] = np.linalg.norm(np.diff(pos_post, axis=0), axis=1) / self.DT_S
            
            # Find first index where speed >= speed_threshold_mm_s
            speed_crossings = np.where(speeds >= speed_threshold_mm_s)[0]
            if len(speed_crossings) == 0:
                idx_onset = idx_event
                onset_latency_ms = 0.0
            else:
                idx_onset = idx_event + speed_crossings[0]
                onset_latency_ms = float(times[idx_onset] - times[idx_event])
        else:
            raise ValueError(f"Unknown segmentation mode: {mode}")

        # Check prefix bounds
        idx_prefix_end = idx_onset + self.PREFIX_SAMPLES
        if idx_prefix_end > len(times):
            return None

        # Extract prefix positions: shape (20, 3)
        px = trial.hand_x[idx_onset:idx_prefix_end]
        py = trial.hand_y[idx_onset:idx_prefix_end]
        pz = trial.hand_z[idx_onset:idx_prefix_end]

        # Check for NaNs in prefix
        if np.isnan(px).any() or np.isnan(py).any() or np.isnan(pz).any():
            return None

        prefix_pos = np.stack([px, py, pz], axis=1).astype(np.float32)
        prefix_times = times[idx_onset:idx_prefix_end]

        # Calculate prefix velocities (mm/s) using central difference with forward/backward at edges
        velocities = np.zeros_like(prefix_pos)
        velocities[0] = (prefix_pos[1] - prefix_pos[0]) / self.DT_S
        velocities[-1] = (prefix_pos[-1] - prefix_pos[-2]) / self.DT_S
        velocities[1:-1] = (prefix_pos[2:] - prefix_pos[:-2]) / (2.0 * self.DT_S)

        # Extract actual endpoint (last valid sample before recording/tracking ends)
        valid_indices = np.where(~np.isnan(trial.hand_x))[0]
        if len(valid_indices) == 0:
            return None
        last_idx = valid_indices[-1]
        actual_endpoint = np.array(
            [trial.hand_x[last_idx], trial.hand_y[last_idx], trial.hand_z[last_idx]],
            dtype=np.float32,
        )
        endpoint_time = float(times[last_idx])
        duration_ms = endpoint_time - float(times[idx_onset])

        # Extract horizons (+100, +200, +300, +500 ms relative to prefix end)
        horizon_trajs = {}
        horizon_pts = {}
        for h in self.HORIZONS_MS:
            n_samples_h = int(h / 10)
            end_h_idx = idx_prefix_end + n_samples_h
            if end_h_idx <= len(times):
                hx = trial.hand_x[idx_prefix_end:end_h_idx]
                hy = trial.hand_y[idx_prefix_end:end_h_idx]
                hz = trial.hand_z[idx_prefix_end:end_h_idx]
                if not (np.isnan(hx).any() or np.isnan(hy).any() or np.isnan(hz).any()):
                    horizon_trajs[h] = np.stack([hx, hy, hz], axis=1).astype(np.float32)
                    horizon_pts[h] = horizon_trajs[h][-1]

        # Full trajectory from onset to last valid index
        full_x = trial.hand_x[idx_onset : last_idx + 1]
        full_y = trial.hand_y[idx_onset : last_idx + 1]
        full_z = trial.hand_z[idx_onset : last_idx + 1]
        full_pos = np.stack([full_x, full_y, full_z], axis=1).astype(np.float32)
        full_times = times[idx_onset : last_idx + 1]

        return TrajectorySample(
            participant_id=trial.participant_id,
            run_id=trial.run_id,
            trial_id=trial.trial_id,
            congruency=trial.congruency,
            target_goal=trial.target_goal,
            physical_hand_direction=trial.physical_hand_direction,
            sampling_rate=100.0,
            coordinate_units="mm",
            segmentation_mode=mode,
            onset_latency_ms=onset_latency_ms,
            t_onset_ms=float(times[idx_onset]),
            onset_sample_idx=int(idx_onset),
            movement_duration_ms=float(duration_ms),
            prefix_positions=prefix_pos,
            prefix_velocities=velocities.astype(np.float32),
            prefix_times=prefix_times,
            actual_endpoint=actual_endpoint,
            endpoint_time_ms=endpoint_time,
            horizon_trajectories=horizon_trajs,
            horizon_points=horizon_pts,
            full_trajectory=full_pos,
            full_times=full_times,
        )

    def load_cohort(
        self,
        manifest_path: str = "data/manifests/iackd_manifest.csv",
        subjects: Optional[List[str]] = None,
        mode: str = "event_locked",
        speed_threshold_mm_s: float = 20.0,
    ) -> Tuple[List[TrajectorySample], Dict]:
        """Load and segment all valid trials in the manifest across subjects.
        
        Args:
            manifest_path: Path to dataset manifest CSV.
            subjects: Optional list of subject IDs to process.
            mode: 'event_locked' (starts at event marker t >= 0) or
                  'kinematic_onset' (starts at physical movement onset where speed >= speed_threshold).
            speed_threshold_mm_s: Velocity threshold in mm/s for physical kinematic onset.
        
        Returns:
            samples: List of valid TrajectorySample objects
            qc_summary: Dictionary detailing total, valid, and excluded trials
        """
        manifest_df = pd.read_csv(manifest_path)
        valid_manifest = manifest_df[manifest_df["valid"]]

        target_subs = subjects or sorted(list(valid_manifest["participant_id"].unique()))

        samples: List[TrajectorySample] = []
        excluded_trials: List[Dict] = []

        for sub in target_subs:
            sub_rows = valid_manifest[valid_manifest["participant_id"] == sub]
            runs = sorted(list(sub_rows["run_id"].unique()))
            for r in runs:
                run_trials = self.loader.load_run(sub, r)
                for tr in run_trials:
                    sample = self.process_trial(
                        tr,
                        mode=mode,
                        speed_threshold_mm_s=speed_threshold_mm_s,
                    )
                    if sample is not None:
                        samples.append(sample)
                    else:
                        excluded_trials.append({
                            "trial_id": tr.trial_id,
                            "participant_id": tr.participant_id,
                            "reason": "Missing kinematics or NaN/short prefix",
                        })

        qc_summary = {
            "total_candidate_trials": len(manifest_df),
            "manifest_valid_trials": len(valid_manifest),
            "segmented_samples": len(samples),
            "excluded_count": len(excluded_trials),
            "subjects_evaluated": target_subs,
            "horizons_ms": self.HORIZONS_MS,
            "observation_window_ms": 200,
            "segmentation_mode": mode,
            "speed_threshold_mm_s": speed_threshold_mm_s,
        }
        return samples, qc_summary
