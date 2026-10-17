"""Neuroscience analysis of the Bereitschaftspotential / Readiness Potential (RP).

Analyzes the slow negative movement-preceding cortical potential over the
pre-movement epoch [-1000 ms, 0 ms] across sensorimotor electrodes (C3, Cz, C4).
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
from scipy import stats

from src.data.schema import CanonicalTrial


class ReadinessPotentialAnalyzer:
    """Computes and statistically evaluates Readiness Potential slow waveforms."""

    TARGET_CHANNELS = ["C3", "CZ", "C4"]

    def __init__(self, channels: Optional[List[str]] = None):
        self.channels = channels or self.TARGET_CHANNELS

    @staticmethod
    def _bootstrap_ci(data: np.ndarray, n_boot: int = 1000, ci: float = 0.95) -> Tuple[np.ndarray, np.ndarray]:
        """Compute bootstrap confidence interval for mean across axis 0."""
        n = data.shape[0]
        if n < 3:
            mean = np.mean(data, axis=0)
            return mean, mean
        rng = np.random.RandomState(42)
        indices = rng.randint(0, n, (n_boot, n))
        boot_means = np.array([np.mean(data[idx], axis=0) for idx in indices])
        alpha = (1.0 - ci) / 2.0
        low = np.percentile(boot_means, alpha * 100, axis=0)
        high = np.percentile(boot_means, (1.0 - alpha) * 100, axis=0)
        return low, high

    def analyze_trials(self, trials: List[CanonicalTrial]) -> Dict[str, Dict]:
        """Extract waveforms and compute group means, SEM, 95% CIs, and RP slopes.
        
        Returns:
            Dictionary partitioned by condition and direction:
            - 'hand_left', 'hand_right'
            - 'goal_left', 'goal_right'
            - 'congruent', 'incongruent'
        """
        valid_trials = [t for t in trials if t.valid]
        if not valid_trials:
            raise ValueError("No valid trials provided for RP analysis.")

        times = valid_trials[0].times_pre  # shape (200,) in seconds [-1.0, 0.0]
        ch_names = valid_trials[0].channels

        ch_indices = {ch: ch_names.index(ch) for ch in self.channels if ch in ch_names}

        # Collect 3D matrices: (N, time, channels)
        groups = {
            "all": valid_trials,
            "hand_left": [t for t in valid_trials if t.physical_hand_direction == "left"],
            "hand_right": [t for t in valid_trials if t.physical_hand_direction == "right"],
            "goal_left": [t for t in valid_trials if t.target_goal == "left"],
            "goal_right": [t for t in valid_trials if t.target_goal == "right"],
            "congruent": [t for t in valid_trials if t.congruency == "congruent"],
            "incongruent": [t for t in valid_trials if t.congruency == "incongruent"],
        }

        results: Dict[str, Dict] = {"times": times, "channels": self.channels}

        for grp_name, grp_trials in groups.items():
            if not grp_trials:
                continue
            n_trials = len(grp_trials)
            grp_data = np.zeros((n_trials, len(times), len(self.channels)), dtype=np.float32)

            for i, tr in enumerate(grp_trials):
                for ch_i, ch in enumerate(self.channels):
                    if ch in ch_indices:
                        grp_data[i, :, ch_i] = tr.eeg_pre[:, ch_indices[ch]]

            mean_wave = np.mean(grp_data, axis=0)  # (time, n_ch)
            sem_wave = stats.sem(grp_data, axis=0)  # (time, n_ch)
            ci_low, ci_high = self._bootstrap_ci(grp_data)

            # Compute late RP slope (-500 ms to 0 ms)
            mask_late = times >= -0.500
            t_late = times[mask_late]
            slopes = {}
            for ch_i, ch in enumerate(self.channels):
                y_late = mean_wave[mask_late, ch_i]
                res = stats.linregress(t_late, y_late)
                slopes[ch] = {
                    "slope_uV_per_s": float(res.slope),
                    "rvalue": float(res.rvalue),
                    "pvalue": float(res.pvalue),
                }

            results[grp_name] = {
                "n_trials": n_trials,
                "mean": mean_wave,
                "sem": sem_wave,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "slopes": slopes,
            }

        # Statistical comparison between left and right hand at movement onset (t=0)
        cz_idx = self.channels.index("CZ") if "CZ" in self.channels else 0
        hl_trials = groups["hand_left"]
        hr_trials = groups["hand_right"]
        if hl_trials and hr_trials:
            hl_onset = np.array([t.eeg_pre[-1, ch_indices["CZ"]] for t in hl_trials])
            hr_onset = np.array([t.eeg_pre[-1, ch_indices["CZ"]] for t in hr_trials])
            t_stat, p_val = stats.ttest_ind(hl_onset, hr_onset, equal_var=False)
            results["stat_hand_diff_cz_onset"] = {"t_stat": float(t_stat), "p_val": float(p_val)}

        return results
