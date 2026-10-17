from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
import h5py

from src.data.openneuro_client import OpenNeuroClient
from src.data.hdf5_reader import IACKDReader
from src.data.trial_loader import TrialLoader


class ManifestBuilder:
    """Builds machine-readable manifests indexing trials, file paths, conditions, and quality metrics."""

    def __init__(self, client: Optional[OpenNeuroClient] = None, output_dir: Optional[Path] = None):
        self.client = client or OpenNeuroClient()
        self.loader = TrialLoader(self.client)
        self.output_dir = Path(output_dir or "data/manifests")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def build_cached_manifest(self) -> pd.DataFrame:
        """Scan data/cache for all available subjects and runs and compile a complete trial manifest."""
        cache_dir = self.client.cache_dir
        rows: List[Dict] = []
        
        # Discover cached subjects
        for sub_dir in sorted(cache_dir.iterdir()):
            if not sub_dir.is_dir() or not sub_dir.name.startswith("sub-"):
                continue
            sub = sub_dir.name
            
            # Find all pre-movement mat files in this subject's cache
            pre_files = sorted(list(sub_dir.glob("*_pre_eeg_*.mat")))
            for pre_path in pre_files:
                parts = pre_path.name.replace(".mat", "").split("_")
                run_tag = next(p for p in parts if p.startswith("run-"))
                hand_tag = parts[-1]
                run_name = f"{run_tag}_{hand_tag}"
                
                # Matching me file
                me_name = pre_path.name.replace("_pre_eeg_", "_eeg_leap_")
                me_path = sub_dir / me_name
                me_available = me_path.exists()
                
                # Read pre-movement metadata
                pre_meta = IACKDReader.read_pre_movement_metadata(pre_path)
                n_pre_trials = pre_meta["n_trials"]
                
                # Check me file if available
                me_trials_dict = {}
                if me_available:
                    with h5py.File(me_path, "r") as fm:
                        out = fm["OUT"]
                        n_me = out["EEG"].shape[0]
                        for j in range(n_me):
                            seg_idx = int(fm[out["eeg_seg_idx"][j, 0]][0, 0])
                            pre_idx = seg_idx - 1
                            cov = float(fm[out["cov"][j, 0]][0, 0])
                            start_res = float(fm[out["start_res_ms"][j, 0]][0, 0])
                            t_len = fm[out["t_ms"][j, 0]].shape[1]
                            me_trials_dict[pre_idx] = {
                                "seg_idx": seg_idx,
                                "cov": cov,
                                "start_res_ms": start_res,
                                "t_samples": t_len,
                            }
                
                for tr_idx in range(n_pre_trials):
                    lbl_code = pre_meta["label_codes"][tr_idx]
                    color, ball_dir, hand_dir, congruency = self.loader.derive_semantics(lbl_code)
                    trial_id = f"{sub}_{run_name}_tr{tr_idx:02d}"
                    
                    in_me = tr_idx in me_trials_dict
                    cov = me_trials_dict[tr_idx]["cov"] if in_me else 0.0
                    start_res = me_trials_dict[tr_idx]["start_res_ms"] if in_me else 0.0
                    me_samples = me_trials_dict[tr_idx]["t_samples"] if in_me else 0
                    
                    rows.append({
                        "trial_id": trial_id,
                        "participant_id": sub,
                        "run_id": run_name,
                        "trial_index": tr_idx,                      # 0-based Python array index
                        "trial_number": tr_idx + 1,                 # 1-based experimental trial number
                        "eeg_seg_idx": me_trials_dict[tr_idx]["seg_idx"] if in_me else tr_idx + 1, # MATLAB 1-based index
                        "label_code": lbl_code,
                        "target_goal": ball_dir,                    # intended abstract reaching goal
                        "physical_hand_direction": hand_dir,        # physical motor execution
                        "ball_direction": ball_dir,                 # screen ball motion
                        "ball_color": color,                        # color cue
                        "congruency": congruency,                   # congruent / incongruent
                        "target_direction": ball_dir,               # alias for target_goal
                        "hand_direction": hand_dir,                 # alias for physical_hand_direction
                        "pre_eeg_available": True,
                        "execution_eeg_available": in_me,
                        "kinematics_available": in_me,
                        "coverage": cov,
                        "start_res_ms": start_res,
                        "valid": pre_meta["keep_mask"][tr_idx] > 0.5 and in_me,
                        "alignment_status": "aligned" if in_me else "dropped_in_execution",
                        "pre_samples": 200,
                        "execution_samples": me_samples,
                        "pre_file_path": str(pre_path),
                        "me_file_path": str(me_path) if me_available else "",
                    })
        
        df = pd.DataFrame(rows)
        manifest_csv = self.output_dir / "iackd_manifest.csv"
        df.to_csv(manifest_csv, index=False)
        print(f"Generated manifest with {len(df)} trials at {manifest_csv}")
        return df
