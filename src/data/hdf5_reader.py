from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import h5py
import numpy as np


class IACKDReader:
    """Low-level HDF5 parser for IACKD pre-movement and movement-execution derivatives."""

    @staticmethod
    def _read_str(f: h5py.File, ref: Any) -> str:
        """Decode a MATLAB character array reference into a clean Python string."""
        raw = f[ref][:]
        chars = [chr(c[0]) for c in raw]
        return "".join(chars).strip("\"' \t\r\n")

    @classmethod
    def read_pre_movement_metadata(cls, pre_path: Path) -> Dict[str, Any]:
        """Extract metadata, channel list, and trial counts from a pre-movement MAT file."""
        with h5py.File(pre_path, "r") as f:
            srate = float(f["meta"]["srate"][()][0, 0])
            chan_refs = f["meta"]["chanlabels"][:]
            channels = [cls._read_str(f, ref) for ref in chan_refs.flatten()]
            
            label_codes = f["label_code"][:].flatten().tolist()
            color_codes = f["color_code"][:].flatten().tolist()
            direct_codes = f["direct_code"][:].flatten().tolist()
            keep_mask = f["keep_mask_1s"][:].flatten().tolist()
            n_trials = len(label_codes)
            
            return {
                "srate": srate,
                "channels": channels,
                "n_trials": n_trials,
                "label_codes": label_codes,
                "color_codes": color_codes,
                "direct_codes": direct_codes,
                "keep_mask": keep_mask,
            }

    @classmethod
    def read_pre_movement_trial(
        cls, pre_path: Path, trial_idx: int
    ) -> Dict[str, Any]:
        """Read a single trial segment and times from pre-movement file."""
        with h5py.File(pre_path, "r") as f:
            seg_ref = f["segs_1s"][0, trial_idx]
            seg_data = np.array(f[seg_ref][:], dtype=np.float32)  # shape: (n_samples, n_channels)
            
            time_ref = f["times_1s"][0, trial_idx]
            times = np.array(f[time_ref][:].flatten(), dtype=np.float64)
            
            lbl_code = int(f["label_code"][0, trial_idx])
            keep = bool(f["keep_mask_1s"][0, trial_idx] > 0.5)
            
            return {
                "eeg_pre": seg_data,
                "times_pre": times,
                "label_code": lbl_code,
                "keep": keep,
            }

    @classmethod
    def read_movement_execution_trial(
        cls, me_path: Path, me_trial_idx: int
    ) -> Dict[str, Any]:
        """Read a single trial from movement-execution file."""
        with h5py.File(me_path, "r") as f:
            out = f["OUT"]
            
            eeg_ref = out["EEG"][me_trial_idx, 0]
            eeg = np.array(f[eeg_ref][:], dtype=np.float32)  # shape: (T, n_channels)
            
            t_ref = out["t_ms"][me_trial_idx, 0]
            t_ms = np.array(f[t_ref][:].flatten(), dtype=np.float64)  # shape: (T,)
            
            x_ref = out["x_mm"][me_trial_idx, 0]
            x_mm = np.array(f[x_ref][:].flatten(), dtype=np.float32)
            
            y_ref = out["y_mm"][me_trial_idx, 0]
            y_mm = np.array(f[y_ref][:].flatten(), dtype=np.float32)
            
            z_ref = out["z_mm"][me_trial_idx, 0]
            z_mm = np.array(f[z_ref][:].flatten(), dtype=np.float32)
            
            bx_ref = out["ball_x_pix"][me_trial_idx, 0]
            ball_x_pix = np.array(f[bx_ref][:].flatten(), dtype=np.float32)
            
            ball_color = cls._read_str(f, out["ball_color"][me_trial_idx, 0])
            move_direct = cls._read_str(f, out["move_direct"][me_trial_idx, 0])
            
            cov = float(f[out["cov"][me_trial_idx, 0]][0, 0])
            start_res = float(f[out["start_res_ms"][me_trial_idx, 0]][0, 0])
            
            seg_idx = int(f[out["eeg_seg_idx"][me_trial_idx, 0]][0, 0])
            lm_trial = int(f[out["lm_trial"][me_trial_idx, 0]][0, 0])
            
            return {
                "eeg_me": eeg,
                "times_me": t_ms,
                "hand_x": x_mm,
                "hand_y": y_mm,
                "hand_z": z_mm,
                "ball_x_pix": ball_x_pix,
                "ball_color": ball_color,
                "move_direct": move_direct,
                "cov": cov,
                "start_res_ms": start_res,
                "eeg_seg_idx": seg_idx,
                "lm_trial": lm_trial,
            }
