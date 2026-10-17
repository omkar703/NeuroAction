"""API router for causal trial simulation and live inference execution."""

from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Query

from web.backend.schemas import (
    InferenceRequest,
    LiveInferenceResponse,
    TrialDataResponse,
)
from web.backend.services.inference_service import (
    InferenceService,
    get_inference_service,
)

router = APIRouter()


@router.get("/trial-data", response_model=TrialDataResponse)
def get_trial_data(
    trial_id: str = Query(..., description="Canonical trial identifier (e.g. 'sub-01_run-01_L1_tr00')"),
    protocol: Optional[str] = Query("kinematic_onset", description="Forecasting protocol ('kinematic_onset' or 'event_locked')"),
    inference_service: InferenceService = Depends(get_inference_service),
) -> TrialDataResponse:
    """Retrieve complete 3D kinematic trajectories, neural intention decodings, and conflict timeseries for replay."""
    try:
        return inference_service.get_trial_data(trial_id=trial_id, protocol=protocol)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(exc)}")


@router.post("/run-inference", response_model=LiveInferenceResponse)
def run_inference(
    request: InferenceRequest,
    inference_service: InferenceService = Depends(get_inference_service),
) -> LiveInferenceResponse:
    """Execute live prospective multimodal inference on a specified trial."""
    try:
        protocol = request.protocol or "kinematic_onset"
        return inference_service.run_live_inference(
            trial_id=request.trial_id, protocol=protocol
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Live inference execution failed: {str(exc)}")
