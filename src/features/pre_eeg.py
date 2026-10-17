"""Feature extraction and spatial filtering for pre-movement EEG signals.

Implements temporal sliding-window slicing, time-domain metrics, frequency bandpowers,
motor-channel subselection, and strict leakage-free Common Spatial Pattern (CSP) filtering.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from scipy import signal, linalg
from sklearn.base import BaseEstimator, TransformerMixin

from src.data.schema import CanonicalTrial


MOTOR_FOCUSED_CHANNELS = ["FC3", "FCZ", "FC4", "C3", "CZ", "C4", "CP3", "CPZ", "CP4"]


def get_sliding_windows(
    total_samples: int = 200,
    window_samples: int = 40,
    stride_samples: int = 10,
    srate: float = 200.0,
) -> List[Dict]:
    """Generate temporal sliding windows over [-1000, 0] ms.
    
    Args:
        total_samples: 200 samples (1000 ms at 200 Hz)
        window_samples: 40 samples (200 ms)
        stride_samples: 10 samples (50 ms)
        srate: 200 Hz
    Returns:
        List of dicts:
        [{'start_idx': 0, 'end_idx': 40, 'start_ms': -1000.0, 'end_ms': -800.0, 'center_ms': -900.0, 'window_id': 0}, ...]
    """
    windows = []
    idx = 0
    w_id = 0
    while idx + window_samples <= total_samples:
        end_idx = idx + window_samples
        start_ms = (idx / srate - 1.0) * 1000.0
        end_ms = (end_idx / srate - 1.0) * 1000.0
        center_ms = (start_ms + end_ms) / 2.0
        windows.append({
            "window_id": w_id,
            "start_idx": idx,
            "end_idx": end_idx,
            "start_ms": float(start_ms),
            "end_ms": float(end_ms),
            "center_ms": float(center_ms),
        })
        idx += stride_samples
        w_id += 1
    return windows


def extract_time_domain_features(data: np.ndarray) -> np.ndarray:
    """Extract time-domain summary statistics from an EEG window.
    
    Args:
        data: shape (n_samples, n_channels)
    Returns:
        1D array of features (n_channels * 6): [mean, variance, rms, slope, min, max]
    """
    n_samples, n_channels = data.shape
    means = np.mean(data, axis=0)
    vars_ = np.var(data, axis=0)
    rms = np.sqrt(np.mean(data ** 2, axis=0))
    t = np.arange(n_samples)
    t_center = t - np.mean(t)
    denom = np.sum(t_center ** 2)
    slopes = np.sum(t_center[:, None] * (data - means[None, :]), axis=0) / (denom + 1e-12)
    mins = np.min(data, axis=0)
    maxs = np.max(data, axis=0)
    return np.concatenate([means, vars_, rms, slopes, mins, maxs], axis=0)


def extract_bandpower_features(
    data: np.ndarray, srate: float = 200.0
) -> np.ndarray:
    """Extract mu (8-13 Hz) and beta (13-30 Hz) bandpowers and ratio.
    
    Args:
        data: shape (n_samples, n_channels)
        srate: sampling rate in Hz
    Returns:
        1D array of features (n_channels * 3): [mu_power, beta_power, mu_ratio]
    """
    n_samples, n_channels = data.shape
    # FFT spectrum
    freqs = np.fft.rfftfreq(n_samples, d=1.0 / srate)
    fft_vals = np.abs(np.fft.rfft(data, axis=0)) ** 2 / n_samples

    mu_mask = (freqs >= 8.0) & (freqs <= 13.0)
    beta_mask = (freqs >= 13.0) & (freqs <= 30.0)

    mu_power = np.mean(fft_vals[mu_mask, :], axis=0) if np.any(mu_mask) else np.zeros(n_channels)
    beta_power = np.mean(fft_vals[beta_mask, :], axis=0) if np.any(beta_mask) else np.zeros(n_channels)
    total_power = mu_power + beta_power + 1e-10
    mu_ratio = mu_power / total_power

    return np.concatenate([mu_power, beta_power, mu_ratio], axis=0)


def extract_trial_features(
    eeg_segment: np.ndarray,
    feature_type: str = "combined",
    srate: float = 200.0,
) -> np.ndarray:
    """Extract specified feature vector from a 2D EEG segment (n_samples, n_channels)."""
    if feature_type == "time_domain":
        return extract_time_domain_features(eeg_segment)
    elif feature_type == "bandpower":
        return extract_bandpower_features(eeg_segment, srate=srate)
    elif feature_type == "combined":
        td = extract_time_domain_features(eeg_segment)
        bp = extract_bandpower_features(eeg_segment, srate=srate)
        return np.concatenate([td, bp], axis=0)
    else:
        raise ValueError(f"Unknown feature_type: {feature_type}")


class CommonSpatialPattern(BaseEstimator, TransformerMixin):
    """Common Spatial Pattern (CSP) for two-class multi-channel EEG.
    
    Strictly estimates spatial projection filters from the training fold only.
    Input format: (n_trials, n_samples, n_channels)
    Output format: (n_trials, 2 * n_components) log-variance features.
    """

    def __init__(self, n_components: int = 3, reg: float = 1e-4):
        self.n_components = n_components
        self.reg = reg
        self.filters_: Optional[np.ndarray] = None
        self.patterns_: Optional[np.ndarray] = None

    def _compute_covariance(self, X: np.ndarray) -> np.ndarray:
        """Compute average normalized covariance across trials."""
        n_trials, n_samples, n_channels = X.shape
        covs = np.zeros((n_trials, n_channels, n_channels))
        for i in range(n_trials):
            xi = X[i] - np.mean(X[i], axis=0, keepdims=True)
            cov_i = np.dot(xi.T, xi)
            trace = np.trace(cov_i)
            covs[i] = cov_i / (trace if trace > 0 else 1.0)
        return np.mean(covs, axis=0)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "CommonSpatialPattern":
        """Fit spatial filters on training data X and binary labels y (0 and 1)."""
        classes = np.unique(y)
        if len(classes) != 2:
            raise ValueError(f"CSP requires exactly 2 classes, got {len(classes)}.")

        X0 = X[y == classes[0]]
        X1 = X[y == classes[1]]

        cov0 = self._compute_covariance(X0)
        cov1 = self._compute_covariance(X1)

        # Regularize covariances to ensure positive definiteness
        n_ch = cov0.shape[0]
        cov0 += self.reg * np.eye(n_ch)
        cov1 += self.reg * np.eye(n_ch)
        cov_composite = cov0 + cov1

        # Solve generalized eigenvalue problem: cov0 * w = lambda * (cov0 + cov1) * w
        eigenvals, eigenvecs = linalg.eigh(cov0, cov_composite)

        # Sort eigenvectors by eigenvalue in descending order
        sort_idx = np.argsort(eigenvals)[::-1]
        eigenvecs = eigenvecs[:, sort_idx]

        # Select top m and bottom m filters
        m = min(self.n_components, n_ch // 2)
        top_filters = eigenvecs[:, :m]
        bottom_filters = eigenvecs[:, -m:]
        self.filters_ = np.concatenate([top_filters, bottom_filters], axis=1)  # (n_channels, 2*m)

        # Spatial patterns are pseudo-inverse of filters
        self.patterns_ = linalg.pinv(self.filters_.T)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Project signals onto spatial filters and extract log-variance features."""
        if self.filters_ is None:
            raise ValueError("CSP must be fitted before transforming data.")

        n_trials, n_samples, n_channels = X.shape
        n_filters = self.filters_.shape[1]
        features = np.zeros((n_trials, n_filters), dtype=np.float32)

        for i in range(n_trials):
            # Project signal: (n_samples, n_filters)
            projected = np.dot(X[i], self.filters_)
            # Variance per filter
            var = np.var(projected, axis=0)
            # Log transform
            features[i] = np.log(var + 1e-10)

        return features
