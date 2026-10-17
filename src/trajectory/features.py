"""Feature engineering for kinematic trajectory forecasting."""

from typing import Dict, List, Tuple
import numpy as np
from src.trajectory.schema import TrajectorySample


def extract_kinematic_features(sample: TrajectorySample) -> np.ndarray:
    """Extract a rich, standardized feature vector from the 200 ms observed prefix (20 samples @ 100 Hz).
    
    Features extracted strictly from observed prefix:
    1. Start position p(0) [3]
    2. End-of-prefix position p(t_obs) [3]
    3. Total prefix displacement p(t_obs) - p(0) [3]
    4. Mean velocity over prefix [3]
    5. Final instantaneous velocity [3]
    6. Least-squares linear velocity slope [3]
    7. Approximate acceleration [3]
    8. Integrated arc length (path length) [1]
    9. Path straightness ratio [1]
    10. Per-axis summary statistics (mean, std, min, max) [12]
    11. Relative sequence p(i) - p(0) flattened [60]
    
    Total dimension: 95 features.
    """
    pos = sample.prefix_positions  # shape (20, 3)
    vel = sample.prefix_velocities # shape (20, 3)

    p0 = pos[0]
    p_end = pos[-1]
    disp = p_end - p0

    dt_total = 0.19  # 19 time steps * 10 ms
    v_mean = disp / dt_total
    v_final = vel[-1]
    v_start = vel[0]

    # Least squares linear velocity per axis: p(t) = p0 + v * t
    t_steps = np.arange(20) * 0.01  # seconds
    t_center = t_steps - np.mean(t_steps)
    denom = np.sum(t_center ** 2) + 1e-12
    v_ls = np.sum(t_center[:, None] * (pos - np.mean(pos, axis=0, keepdims=True)), axis=0) / denom

    acc_mean = (v_final - v_start) / dt_total

    # Step-by-step distances
    diffs = np.diff(pos, axis=0)
    step_dists = np.linalg.norm(diffs, axis=1)
    path_len = np.sum(step_dists)
    disp_norm = np.linalg.norm(disp)
    straightness = disp_norm / (path_len + 1e-8)

    # Per axis summary stats
    mean_pos = np.mean(pos, axis=0)
    std_pos = np.std(pos, axis=0)
    min_pos = np.min(pos, axis=0)
    max_pos = np.max(pos, axis=0)

    # Relative coordinates sequence
    rel_seq = (pos - p0[None, :]).flatten()

    features = np.concatenate([
        p0,
        p_end,
        disp,
        v_mean,
        v_final,
        v_ls,
        acc_mean,
        np.array([path_len, straightness], dtype=np.float32),
        mean_pos,
        std_pos,
        min_pos,
        max_pos,
        rel_seq,
    ], axis=0).astype(np.float32)

    return features
