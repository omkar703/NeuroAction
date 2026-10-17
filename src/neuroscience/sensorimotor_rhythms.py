"""Neuroscience analysis of Sensorimotor Rhythms (SMR): mu (8-13 Hz) and beta (13-30 Hz).

Computes Event-Related Desynchronization (ERD) / Event-Related Synchronization (ERS),
contralateral vs ipsilateral lateralization, and time-frequency power dynamics
over the pre-movement epoch [-1000 ms, 0 ms].
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
from scipy import signal

from src.data.schema import CanonicalTrial


class SensorimotorRhythmAnalyzer:
    """Computes mu and beta band ERD/ERS and lateralization dynamics."""

    MU_BAND = (8.0, 13.0)
    BETA_BAND = (13.0, 30.0)
    TARGET_CHANNELS = ["C3", "CZ", "C4"]

    def __init__(self, srate: float = 200.0):
        self.srate = srate

    def _bandpass_power_envelope(
        self, data: np.ndarray, band: Tuple[float, float]
    ) -> np.ndarray:
        """Filter signal in band and compute instantaneous power envelope via analytic signal (Hilbert).
        
        Args:
            data: shape (n_samples, n_channels) or (n_trials, n_samples, n_channels)
            band: (low_freq, high_freq)
        Returns:
            envelope power squared: same shape as data
        """
        nyq = self.srate / 2.0
        b, a = signal.butter(3, [band[0] / nyq, band[1] / nyq], btype="bandpass")
        # Apply forward-backward filter along axis 0 or axis 1
        filtered = signal.filtfilt(b, a, data, axis=1 if data.ndim == 3 else 0)
        # Hilbert envelope
        analytic = signal.hilbert(filtered, axis=1 if data.ndim == 3 else 0)
        power = np.abs(analytic) ** 2
        return power

    def compute_erd(
        self,
        power: np.ndarray,
        times: np.ndarray,
        baseline_window: Tuple[float, float] = (-1.0, -0.8),
    ) -> np.ndarray:
        """Compute ERD% relative to baseline window: ERD% = (P(t) - P_base) / P_base * 100.
        
        Args:
            power: (n_trials, n_samples, n_channels)
            times: (n_samples,) in seconds
            baseline_window: (start_sec, end_sec)
        Returns:
            erd: (n_trials, n_samples, n_channels) in percent
        """
        mask_base = (times >= baseline_window[0]) & (times <= baseline_window[1])
        base_power = np.mean(power[:, mask_base, :], axis=1, keepdims=True)  # (n_trials, 1, n_channels)
        # Avoid division by zero
        base_power = np.maximum(base_power, 1e-10)
        erd = ((power - base_power) / base_power) * 100.0
        return erd

    def analyze_trials(self, trials: List[CanonicalTrial]) -> Dict[str, Dict]:
        """Compute mu and beta ERD and lateralization across conditions."""
        valid_trials = [t for t in trials if t.valid]
        if not valid_trials:
            raise ValueError("No valid trials provided for SMR analysis.")

        times = valid_trials[0].times_pre
        ch_names = valid_trials[0].channels
        ch_indices = {ch: ch_names.index(ch) for ch in self.TARGET_CHANNELS if ch in ch_names}

        n_trials = len(valid_trials)
        n_samples = len(times)
        data = np.zeros((n_trials, n_samples, len(self.TARGET_CHANNELS)), dtype=np.float32)

        for i, tr in enumerate(valid_trials):
            for c_i, ch in enumerate(self.TARGET_CHANNELS):
                if ch in ch_indices:
                    data[i, :, c_i] = tr.eeg_pre[:, ch_indices[ch]]

        # Compute power envelopes
        mu_power = self._bandpass_power_envelope(data, self.MU_BAND)
        beta_power = self._bandpass_power_envelope(data, self.BETA_BAND)

        mu_erd = self.compute_erd(mu_power, times)
        beta_erd = self.compute_erd(beta_power, times)

        # Lateralization index: (C3 - C4) / (C3 + C4)
        c3_idx = self.TARGET_CHANNELS.index("C3")
        c4_idx = self.TARGET_CHANNELS.index("C4")
        mu_li = (mu_power[:, :, c3_idx] - mu_power[:, :, c4_idx]) / np.maximum(
            mu_power[:, :, c3_idx] + mu_power[:, :, c4_idx], 1e-10
        )
        beta_li = (beta_power[:, :, c3_idx] - beta_power[:, :, c4_idx]) / np.maximum(
            beta_power[:, :, c3_idx] + beta_power[:, :, c4_idx], 1e-10
        )

        # Condition splits
        cond_masks = {
            "all": np.ones(n_trials, dtype=bool),
            "hand_left": np.array([t.physical_hand_direction == "left" for t in valid_trials]),
            "hand_right": np.array([t.physical_hand_direction == "right" for t in valid_trials]),
            "congruent": np.array([t.congruency == "congruent" for t in valid_trials]),
            "incongruent": np.array([t.congruency == "incongruent" for t in valid_trials]),
        }

        results: Dict[str, Dict] = {
            "times": times,
            "channels": self.TARGET_CHANNELS,
        }

        for cond_name, mask in cond_masks.items():
            if not np.any(mask):
                continue
            results[cond_name] = {
                "n_trials": int(np.sum(mask)),
                "mu_erd_mean": np.mean(mu_erd[mask], axis=0),  # (n_samples, n_ch)
                "mu_erd_sem": np.std(mu_erd[mask], axis=0) / np.sqrt(np.sum(mask)),
                "beta_erd_mean": np.mean(beta_erd[mask], axis=0),
                "beta_erd_sem": np.std(beta_erd[mask], axis=0) / np.sqrt(np.sum(mask)),
                "mu_li_mean": np.mean(mu_li[mask], axis=0),  # (n_samples,)
                "beta_li_mean": np.mean(beta_li[mask], axis=0),
            }

        # Time-frequency spectrogram across 4-35 Hz for Cz
        cz_idx = self.TARGET_CHANNELS.index("CZ")
        freqs = np.linspace(4, 35, 32)
        # Average STFT across all trials for Cz
        f, t_spec, zxx = signal.spectrogram(
            data[:, :, cz_idx],
            fs=self.srate,
            nperseg=40,
            noverlap=30,
            scaling="spectrum",
            axis=1,
        )
        # zxx shape: (n_trials, n_freqs, n_time_bins)
        # Shift time to match [-1.0, 0.0]
        t_shifted = t_spec - 1.0
        results["spectrogram"] = {
            "freqs": f,
            "times": t_shifted,
            "mean_power_cz": np.mean(zxx, axis=0),
        }

        return results
