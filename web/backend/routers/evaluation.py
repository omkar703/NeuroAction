"""API router for audited scientific benchmark and evaluation results."""

from fastapi import APIRouter, Depends, HTTPException, Path

from web.backend.schemas import (
    CohortResultsResponse,
    ParticipantResultsResponse,
)
from web.backend.services.results_service import (
    ResultsService,
    get_results_service,
)

router = APIRouter()


@router.get("/results/cohort", response_model=CohortResultsResponse)
def get_cohort_results(
    results_service: ResultsService = Depends(get_results_service),
) -> CohortResultsResponse:
    """Retrieve verified cohort benchmark metrics from Phase 4 and Phase 5 reports."""
    try:
        return results_service.get_cohort_results()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to load cohort results: {str(exc)}")


@router.get("/results/participant/{sub_id}", response_model=ParticipantResultsResponse)
def get_participant_results(
    sub_id: str = Path(..., description="Participant ID (e.g. 'sub-01', 'sub-02')"),
    results_service: ResultsService = Depends(get_results_service),
) -> ParticipantResultsResponse:
    """Retrieve audited prospective evaluation metrics for an individual participant under LOSO isolation."""
    try:
        return results_service.get_participant_results(sub_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to load participant results: {str(exc)}")
