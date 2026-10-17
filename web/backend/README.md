# NeuroAction Phase 6 — Interactive Digital Twin Web Backend

Production REST API for the **NeuroAction Phase 6 Interactive Digital Twin**, bridging the audited neuro-computational and biomechanical machine learning pipelines with an interactive 3D web frontend.

---

## 1. Overview and Architecture

The backend wraps existing Python scientific pipelines from Phases 2, 3, 4, and 5 without fabricating any synthetic model outputs:
- **Phase 2 (Neural Intention):** Standardized Linear Discriminant Analysis (`LDADecoder`) decoding reaching intention from pre-movement scalp EEG epochs ($-1000$ to $0\text{ ms}$).
- **Phase 3 (Kinematic Forecasting):** Regularized Ridge Regression (`KinematicForecaster`) extrapolating the first $200\text{ ms}$ of reaching movement into continuous 3D paths and terminal hand endpoints ($47.25\text{ mm}$ RMSE).
- **Phase 4 (Causal Replay Engine):** Real-time multimodal state synchronization (`ReplayEngine`) enforcing causal boundaries and evaluating conflict states at every discrete timestep.
- **Phase 5 (Prospective Validation):** Leave-One-Subject-Out (LOSO) cross-validation benchmarks and active abstention decision policy metrics across 269 valid trials.

---

## 2. Directory Structure

```
web/backend/
├── __init__.py
├── main.py                     # FastAPI application entrypoint with CORS
├── schemas.py                  # Pydantic typed data contracts for all API responses
├── requirements.txt            # Backend Python dependencies
├── README.md                   # Setup and API documentation
├── routers/
│   ├── __init__.py
│   ├── trials.py               # /api/participants, /api/trials
│   ├── inference.py            # /api/trial-data, /api/run-inference
│   ├── signals.py              # /api/eeg-signal
│   ├── evaluation.py           # /api/results/cohort, /api/results/participant/{sub_id}
│   └── explainability.py       # /api/model-info, /api/features, /api/lda-coefficients
└── services/
    ├── __init__.py
    ├── trial_service.py        # Wraps TrialRegistry and dataset manifest
    ├── inference_service.py    # Wraps ReplayEngine, runs causal simulation & live inference
    ├── signal_service.py       # Loads calibrated EEG timeseries from .mat derivatives
    └── results_service.py      # Serves audited benchmark matrices and LDA coefficients
```

---

## 3. How to Start the Backend

### Prerequisites
- Python 3.10+ in project virtual environment: `.venv/bin/python`
- Always set `PYTHONPATH` to the project root directory.

### Launch Command (from `web/backend/`)
```bash
cd /run/media/op/DATA/Omkar/CODE-111/GitHub/Brain_Movement/web/backend
PYTHONPATH=/run/media/op/DATA/Omkar/CODE-111/GitHub/Brain_Movement /run/media/op/DATA/Omkar/CODE-111/GitHub/Brain_Movement/.venv/bin/uvicorn main:app --port 8765 --reload
```

### Launch Command (from Project Root)
```bash
cd /run/media/op/DATA/Omkar/CODE-111/GitHub/Brain_Movement
PYTHONPATH=. .venv/bin/uvicorn web.backend.main:app --port 8765 --reload
```

Once running:
- **Interactive Swagger UI:** [http://localhost:8765/docs](http://localhost:8765/docs)
- **ReDoc:** [http://localhost:8765/redoc](http://localhost:8765/redoc)
- **Health Check:** [http://localhost:8765/api/health](http://localhost:8765/api/health)

---

## 4. API Endpoints Reference

All endpoints return a mandatory `data_source` field: `"recorded_replay" | "live_inference" | "saved_result" | "illustrative"`.

| Method | Endpoint | Description | Data Source |
| :--- | :--- | :--- | :--- |
| `GET` | `/api/health` | Service health status | `saved_result` |
| `GET` | `/api/participants` | List of 5 benchmark participants (`sub-01` to `sub-15`) and trial counts | `recorded_replay` |
| `GET` | `/api/trials` | Filtered list of trials (`participant_id`, `valid_only`) | `recorded_replay` |
| `GET` | `/api/trial-data` | Complete 3D trajectories, forecast endpoint, neural decoding, and conflict timeseries | `recorded_replay` |
| `POST` | `/api/run-inference` | Executes live prospective prediction on a trial and returns outcome assessment | `live_inference` |
| `GET` | `/api/eeg-signal` | Calibrated $-1000$ to $0\text{ ms}$ EEG epoch for an electrode channel (e.g. `CZ`, `C3`) | `recorded_replay` |
| `GET` | `/api/results/cohort` | Verified benchmark metrics across 269 trials ($47.25\text{ mm}$ RMSE, $88.35\%$ bal acc) | `saved_result` |
| `GET` | `/api/results/participant/{sub_id}` | Audited LOSO prospective evaluation metrics for a specific participant | `saved_result` |
| `GET` | `/api/model-info` | Architecture, parameters, and horizons for models in the digital twin | `saved_result` |
| `GET` | `/api/features` | Feature pipeline specifications (26 channels, 9 features/channel = 234 features) | `saved_result` |
| `GET` | `/api/lda-coefficients` | LDA spatial filter weights mapped across motor electrode channels | `saved_result` |

---

## 5. Scientific Validation Notes

- **Condition Discrimination vs Biological Ground Truth:** Phase 5 metrics measure discrimination between experimental task conditions (`congruent` vs `incongruent`), **not** internal biological intention-action discordance. The OpenNeuro `ds006840` dataset contains no independent measure of internal participant intention or muscle co-contraction.
- **Temporal Causal Boundary:** Pre-movement EEG decodings are locked to go-cue/onset ($t = 0\text{ ms}$). Trajectory forecasts strictly ingest $t \le 200\text{ ms}$ kinematic prefixes ($20$ samples at $100\text{ Hz}$).
