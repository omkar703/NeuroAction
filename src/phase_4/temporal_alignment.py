"""Temporal alignment, clock synchronization, and causal availability verification for Phase 4."""

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np


@dataclass
class TrialTimeline:
    """Explicitly synchronized timeline of events for an experimental trial."""

    participant_id: str
    trial_id: str
    eeg_start_ms: float = -1000.0
    eeg_end_ms: float = 0.0
    go_cue_ms: float = 0.0
    kinematic_onset_ms: float = 0.0
    prefix_duration_ms: float = 200.0
    prefix_end_ms: float = 200.0
    movement_end_ms: float = 1500.0
    protocol: str = "kinematic_onset"  # "kinematic_onset" or "event_locked"

    @property
    def neural_available_ms(self) -> float:
        """Timestamp at which pre-movement neural intention prediction becomes available."""
        return self.go_cue_ms

    @property
    def forecast_available_ms(self) -> float:
        """Timestamp at which kinematic trajectory forecast becomes causally available."""
        if self.protocol == "event_locked":
            return self.go_cue_ms + self.prefix_duration_ms
        else:
            return self.kinematic_onset_ms + self.prefix_duration_ms

    def is_neural_available(self, replay_time_ms: float) -> bool:
        """True if the replay clock has reached or passed neural prediction availability."""
        return replay_time_ms >= self.neural_available_ms

    def is_forecast_available(self, replay_time_ms: float) -> bool:
        """True if the replay clock has reached or passed kinematic forecast availability."""
        return replay_time_ms >= self.forecast_available_ms


def construct_trial_timeline(
    participant_id: str,
    trial_id: str,
    kinematic_times: np.ndarray,
    speed_series: Optional[np.ndarray] = None,
    protocol: str = "kinematic_onset",
    speed_threshold_mm_s: float = 20.0,
    prefix_duration_ms: float = 200.0,
) -> TrialTimeline:
    """Construct an aligned timeline from experimental timestamps and velocity profile."""
    # Find visual go-cue index: nominal t >= 0
    event_indices = np.where(kinematic_times >= 0.0)[0]
    idx_cue = event_indices[0] if len(event_indices) > 0 else 0
    go_cue_ms = float(kinematic_times[idx_cue])

    # Find kinematic onset
    if protocol == "kinematic_onset" and speed_series is not None:
        cue_post_speeds = speed_series[idx_cue:]
        crossings = np.where(cue_post_speeds >= speed_threshold_mm_s)[0]
        if len(crossings) > 0:
            idx_onset = idx_cue + crossings[0]
            kinematic_onset_ms = float(kinematic_times[idx_onset])
        else:
            idx_onset = idx_cue
            kinematic_onset_ms = go_cue_ms
    else:
        idx_onset = idx_cue
        kinematic_onset_ms = go_cue_ms

    prefix_end_ms = (
        kinematic_onset_ms + prefix_duration_ms
        if protocol == "kinematic_onset"
        else go_cue_ms + prefix_duration_ms
    )
    movement_end_ms = float(kinematic_times[-1])

    return TrialTimeline(
        participant_id=participant_id,
        trial_id=trial_id,
        eeg_start_ms=-1000.0,
        eeg_end_ms=0.0,
        go_cue_ms=go_cue_ms,
        kinematic_onset_ms=kinematic_onset_ms,
        prefix_duration_ms=prefix_duration_ms,
        prefix_end_ms=prefix_end_ms,
        movement_end_ms=movement_end_ms,
        protocol=protocol,
    )


def verify_temporal_integrity(
    prediction_declared_time_ms: float,
    data_max_timestamp_ms: float,
    tolerance_ms: float = 1e-3,
) -> Tuple[bool, Optional[str]]:
    """Check that input data does not contain samples from after the declared prediction time.
    
    Returns:
        (is_valid, violation_explanation)
    """
    if data_max_timestamp_ms > prediction_declared_time_ms + tolerance_ms:
        return (
            False,
            f"Temporal leakage detected: input data extends to {data_max_timestamp_ms:.1f} ms, "
            f"which exceeds declared prediction timestamp {prediction_declared_time_ms:.1f} ms.",
        )
    return True, None
