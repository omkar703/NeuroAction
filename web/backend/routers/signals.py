"""API router for raw and preprocessed EEG signal retrieval."""

from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Query

from web.backend.schemas import EEGSignalResponse
from web.backend.services.signal_service import SignalService, get_signal_service

router = APIRouter()


@router.get("/eeg-signal", response_model=EEGSignalResponse)
def get_eeg_signal(
    trial_id: str = Query(..., description="Canonical trial ID (e.g. 'sub-01_run-01_L1_tr00')"),
    channel: Optional[str] = Query("CZ", description="Target EEG channel name (e.g. 'C3', 'CZ', 'C4')"),
    signal_service: SignalService = Depends(get_signal_service),
) -> EEGSignalResponse:
    """Retrieve actual pre-movement EEG epoch (-1000 to 0 ms) for a specific electrode channel."""
    try:
        ch = channel or "CZ"
        return signal_service.get_eeg_signal(trial_id=trial_id, channel=ch)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to load EEG signal: {str(exc)}")
