from pathlib import Path
from typing import Dict, Generator, List, Optional, Tuple
import h5py
import numpy as np

from src.data.openneuro_client import OpenNeuroClient
from src.data.hdf5_reader import IACKDReader
from src.data.schema import CanonicalTrial


class TrialLoader:
    """High-level, memory-efficient loader for multimodal IACKD trials."""

    def __init__(self, client: Optional[OpenNeuroClient] = None):
        self.client = client or OpenNeuroClient()

    @staticmethod
    def derive_semantics(label_code: int) -> Tuple[str, str, str, str]:
        """Derive ground-truth experimental semantics from verified label code:
        
        0 -> red-left:     color=red,    ball=left,  hand=left,  congruent
        1 -> red-right:    color=red,    ball=right, hand=right, congruent
        2 -> yellow-left:  color=yellow, ball=left,  hand=right, incongruent
        3 -> yellow-right: color=yellow, ball=right, hand=left,  incongruent
        """
        color = "red" if label_code in (0, 1) else "yellow"
        ball_dir = "left" if label_code in (0, 2) else "right"
        hand_dir = "left" if label_code in (0, 3) else "right"
        congruency = "congruent" if label_code in (0, 1) else "incongruent"
        return color, ball_dir, hand_dir, congruency

    def load_run(
        self, subject: str, run_name: str
    ) -> List[CanonicalTrial]:
        """Load all valid multimodal trials for a given subject and run."""
        pre_path, me_path = self.client.get_run_files(subject, run_name)
        
        pre_meta = IACKDReader.read_pre_movement_metadata(pre_path)
        channels = pre_meta["channels"]
        srate_pre = pre_meta["srate"]
        
        trials: List[CanonicalTrial] = []
        
        with h5py.File(pre_path, "r") as fp, h5py.File(me_path, "r") as fm:
            out = fm["OUT"]
            n_me = out["EEG"].shape[0]
            
            for j in range(n_me):
                seg_idx = int(fm[out["eeg_seg_idx"][j, 0]][0, 0])
                pre_idx = seg_idx - 1  # convert 1-based MATLAB index to 0-based Python
                
                # Pre-movement data
                seg_ref = fp["segs_1s"][0, pre_idx]
                eeg_pre = np.array(fp[seg_ref][:], dtype=np.float32)
                time_pre_ref = fp["times_1s"][0, pre_idx]
                times_pre = np.array(fp[time_pre_ref][:].flatten(), dtype=np.float64)
                lbl_code = int(fp["label_code"][0, pre_idx])
                
                # Movement-execution data
                eeg_ref = out["EEG"][j, 0]
                eeg_me = np.array(fm[eeg_ref][:], dtype=np.float32)
                t_ref = out["t_ms"][j, 0]
                times_me = np.array(fm[t_ref][:].flatten(), dtype=np.float64)
                
                x_mm = np.array(fm[out["x_mm"][j, 0]][:].flatten(), dtype=np.float32)
                y_mm = np.array(fm[out["y_mm"][j, 0]][:].flatten(), dtype=np.float32)
                z_mm = np.array(fm[out["z_mm"][j, 0]][:].flatten(), dtype=np.float32)
                ball_x = np.array(fm[out["ball_x_pix"][j, 0]][:].flatten(), dtype=np.float32)
                
                cov = float(fm[out["cov"][j, 0]][0, 0])
                start_res = float(fm[out["start_res_ms"][j, 0]][0, 0])
                
                # Semantics
                color, ball_dir, hand_dir, congruency = self.derive_semantics(lbl_code)
                
                # Determine channels_me
                if eeg_me.shape[1] == 28:
                    # Channels with M1 and M2 at indices 18, 19
                    channels_me = channels[:18] + ["M1", "M2"] + channels[18:]
                else:
                    channels_me = channels

                trial_id = f"{subject}_{run_name}_tr{pre_idx:02d}"
                trial = CanonicalTrial(
                    participant_id=subject,
                    run_id=run_name,
                    trial_id=trial_id,
                    trial_index=pre_idx,           # 0-based Python array index
                    trial_number=pre_idx + 1,       # 1-based experimental trial number
                    eeg_seg_idx=seg_idx,            # Original MATLAB 1-based reference
                    label_code=lbl_code,
                    target_goal=ball_dir,           # intended abstract reaching goal
                    physical_hand_direction=hand_dir, # physical hand movement
                    ball_direction=ball_dir,        # screen ball movement
                    ball_color=color,               # color cue
                    congruency=congruency,          # congruent / incongruent
                    eeg_pre=eeg_pre,
                    times_pre=times_pre,
                    srate_pre=srate_pre,
                    channels=channels,
                    eeg_me=eeg_me,
                    times_me=times_me,
                    srate_me=100.0,
                    channels_me=channels_me,
                    hand_x=x_mm,
                    hand_y=y_mm,
                    hand_z=z_mm,
                    ball_x_pix=ball_x,
                    valid=True,
                    coverage=cov,
                    start_res_ms=start_res,
                    is_multimodal=True,
                )
                trials.append(trial)
                
        return trials

    def load_trial(
        self, subject: str, run_name: str, trial_idx: int
    ) -> CanonicalTrial:
        """Load a specific trial by subject, run name, and 0-based trial index (0, 1, ...)."""
        trials = self.load_run(subject, run_name)
        for t in trials:
            if t.trial_index == trial_idx:
                return t
        raise ValueError(
            f"Trial index {trial_idx} not found in {subject} {run_name}. "
            f"Available 0-based trial indices: {[t.trial_index for t in trials]}"
        )

    def load_trial_by_number(
        self, subject: str, run_name: str, trial_number: int
    ) -> CanonicalTrial:
        """Load a specific trial by subject, run name, and 1-based experimental trial number (1, 2, ...)."""
        return self.load_trial(subject, run_name, trial_number - 1)

    def iterate_trials(
        self,
        subjects: Optional[List[str]] = None,
        runs: Optional[List[str]] = None,
    ) -> Generator[CanonicalTrial, None, None]:
        """Lazy generator yielding multimodal trials across subjects and runs."""
        target_subs = subjects or self.client.list_subjects()
        for sub in target_subs:
            sub_files = self.client.list_subject_files(sub, "pre-movement")
            # Extract run names
            sub_runs = []
            for f in sub_files:
                fname = f["filename"]
                if fname.endswith(".mat"):
                    # e.g. sub-01_task-ihc_run-01_pre_eeg_L1.mat -> run-01_L1
                    parts = fname.split("_")
                    run_tag = [p for p in parts if p.startswith("run-")][0]
                    hand_tag = parts[-1].replace(".mat", "")
                    run_key = f"{run_tag}_{hand_tag}"
                    if runs is None or run_key in runs or run_tag in runs:
                        if run_key not in sub_runs:
                            sub_runs.append(run_key)
            
            for r in sub_runs:
                run_trials = self.load_run(sub, r)
                for t in run_trials:
                    yield t
