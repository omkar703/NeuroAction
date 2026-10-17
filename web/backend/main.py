import os
import sys
from pathlib import Path

# Ensure backend directory and project root are in Python search path
BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parent.parent

os.chdir(PROJECT_ROOT)

for p in [str(PROJECT_ROOT), str(BACKEND_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

try:
    from web.backend.routers import (
        evaluation,
        explainability,
        inference,
        signals,
        trials,
    )
    from web.backend.schemas import HealthResponse
except ImportError:
    from routers import (
        evaluation,
        explainability,
        inference,
        signals,
        trials,
    )
    from schemas import HealthResponse

# Initialize FastAPI application
app = FastAPI(
    title="NeuroAction Phase 6 Digital Twin API",
    version="1.0.0",
    description=(
        "Production REST API for the NeuroAction Phase 6 Interactive Multimodal Digital Twin. "
        "Integrates pre-movement scalp EEG neural intention decoding with 3D kinematic trajectory forecasting "
        "and causal conflict detection on OpenNeuro dataset ds006840."
    ),
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS configuration supporting Next.js/React frontend on localhost:3000
origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:8765",
    "http://127.0.0.1:8765",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register modular API routers
app.include_router(trials.router, prefix="/api", tags=["trials"])
app.include_router(inference.router, prefix="/api", tags=["inference"])
app.include_router(signals.router, prefix="/api", tags=["signals"])
app.include_router(evaluation.router, prefix="/api", tags=["evaluation"])
app.include_router(explainability.router, prefix="/api", tags=["explainability"])


@app.get("/api/health", response_model=HealthResponse, tags=["health"])
def health_check() -> HealthResponse:
    """Service health check endpoint."""
    return HealthResponse(
        status="ok",
        service="neuroaction-backend",
        version="1.0.0",
        data_source="saved_result",
    )


@app.get("/", tags=["root"])
def root():
    """Root entrypoint with API documentation reference."""
    return {
        "service": "NeuroAction Phase 6 Interactive Digital Twin API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/health",
        "endpoints": {
            "participants": "/api/participants",
            "trials": "/api/trials",
            "trial_data": "/api/trial-data?trial_id={trial_id}",
            "live_inference": "POST /api/run-inference",
            "eeg_signal": "/api/eeg-signal?trial_id={trial_id}&channel={channel}",
            "cohort_results": "/api/results/cohort",
            "participant_results": "/api/results/participant/{sub_id}",
            "model_info": "/api/model-info",
            "features": "/api/features",
            "lda_coefficients": "/api/lda-coefficients",
        },
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8765, reload=True)
