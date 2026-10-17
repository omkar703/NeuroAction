"""API router for model architecture, feature descriptions, and explainability coefficients."""

from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Query

from web.backend.schemas import (
    FeaturesInfoResponse,
    LDACoefficientsResponse,
    ModelInfoResponse,
)
from web.backend.services.results_service import (
    ResultsService,
    get_results_service,
)

router = APIRouter()


@router.get("/model-info", response_model=ModelInfoResponse)
def get_model_info(
    results_service: ResultsService = Depends(get_results_service),
) -> ModelInfoResponse:
    """Retrieve architecture, parameters, and metadata for models incorporated in the digital twin."""
    try:
        return results_service.get_model_info()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to load model info: {str(exc)}")


@router.get("/features", response_model=FeaturesInfoResponse)
def get_features_info(
    results_service: ResultsService = Depends(get_results_service),
) -> FeaturesInfoResponse:
    """Retrieve specifications of the multi-channel neural and 3D kinematic feature extraction pipelines."""
    try:
        return results_service.get_features_info()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to load feature info: {str(exc)}")


@router.get("/lda-coefficients", response_model=LDACoefficientsResponse)
def get_lda_coefficients(
    participant_id: Optional[str] = Query(None, description="Optional participant ID (e.g. 'sub-01') or 'cohort'"),
    participant: Optional[str] = Query(None, description="Alias for participant_id"),
    results_service: ResultsService = Depends(get_results_service),
) -> LDACoefficientsResponse:
    """Retrieve Linear Discriminant Analysis spatial filter weights mapped across motor-cortex electrode channels."""
    try:
        pid = participant or participant_id
        return results_service.get_lda_coefficients(participant_id=pid)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to extract LDA coefficients: {str(exc)}")

