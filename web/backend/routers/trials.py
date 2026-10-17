"""API router for participant and trial discovery."""

from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Query

from web.backend.schemas import ParticipantListResponse, TrialListResponse
from web.backend.services.trial_service import TrialService, get_trial_service

router = APIRouter()


@router.get("/participants", response_model=ParticipantListResponse)
def list_participants(
    trial_service: TrialService = Depends(get_trial_service),
) -> ParticipantListResponse:
    """Return list of sorted unique participant IDs and cohort summary."""
    try:
        participants = trial_service.list_participants()
        details = trial_service.get_participants_summary()
        return ParticipantListResponse(
            participants=participants,
            details=details,
            total_participants=len(participants),
            data_source="recorded_replay",
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to list participants: {str(exc)}")


@router.get("/trials", response_model=TrialListResponse)
def list_trials(
    participant_id: Optional[str] = Query(None, description="Optional filter by participant ID (e.g. 'sub-01')"),
    valid_only: bool = Query(True, description="Filter for valid multimodal trials only"),
    trial_service: TrialService = Depends(get_trial_service),
) -> TrialListResponse:
    """Return available trials matching the specified criteria."""
    try:
        trials = trial_service.list_trials(participant_id=participant_id, valid_only=valid_only)
        return TrialListResponse(
            trials=trials,
            total_trials=len(trials),
            data_source="recorded_replay",
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to list trials: {str(exc)}")
