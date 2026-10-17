"""Signal service for accessing pre-movement EEG channels from .mat derivatives."""

from typing import Optional
import numpy as np

from web.backend.schemas import EEGSignalResponse
from web.backend.services.trial_service import TrialService, get_trial_service


class SignalService:
    """Extracts calibrated EEG timeseries from canonical trials."""

    def __init__(self, trial_service: Optional[TrialService] = None):
        self.trial_service = trial_service or get_trial_service()

    def get_eeg_signal(
        self,
        trial_id: str,
        channel: str = "CZ",
    ) -> EEGSignalResponse:
        """Extract pre-movement EEG epoch (-1000 to 0 ms) for a specific scalp channel."""
        trial = self.trial_service.get_trial(trial_id)
        if trial is None:
            raise ValueError(f"Trial '{trial_id}' not found in registry.")

        if trial.eeg_pre is None or len(trial.eeg_pre) == 0:
            raise ValueError(f"Pre-movement EEG data not available for trial '{trial_id}'.")

        channels = trial.channels
        norm_ch = channel.strip().upper()
        ch_upper_list = [c.upper() for c in channels]

        if norm_ch not in ch_upper_list:
            raise ValueError(
                f"Channel '{channel}' is not valid. Available channels: {channels}"
            )

        ch_idx = ch_upper_list.index(norm_ch)
        actual_ch_name = channels[ch_idx]

        # Ensure times are in milliseconds
        times_raw = trial.times_pre
        if times_raw is not None and len(times_raw) > 0:
            if np.abs(times_raw[0]) <= 2.0:
                times_ms = [float(t * 1000.0) for t in times_raw]
            else:
                times_ms = [float(t) for t in times_raw]
        else:
            # 200 samples @ 200 Hz -> [-1000, 0] ms
            times_ms = [float(i * 5.0 - 1000.0) for i in range(200)]

        amplitude_uv = [float(v) for v in trial.eeg_pre[:, ch_idx]]
        srate = float(trial.srate_pre) if trial.srate_pre is not None else 200.0

        return EEGSignalResponse(
            channel=actual_ch_name,
            available_channels=channels,
            times_ms=times_ms,
            amplitude_uv=amplitude_uv,
            processing="filtfilt_offline_zero_phase",
            srate_hz=srate,
            trial_id=trial_id,
            data_source="recorded_replay",
        )


# Global singleton instance
_signal_service: Optional[SignalService] = None


def get_signal_service() -> SignalService:
    """Dependency injection provider for SignalService."""
    global _signal_service
    if _signal_service is None:
        _signal_service = SignalService()
    return _signal_service
