"""Pydantic schemas and data contracts for NeuroAction Phase 6 Web Backend."""

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field


DataSourceType = Literal["recorded_replay", "live_inference", "saved_result", "illustrative"]


class ParticipantSummary(BaseModel):
    """Participant metadata and trial summary."""
    participant_id: str = Field(..., description="Canonical participant identifier (e.g. 'sub-01')")
    total_trials: int = Field(..., description="Total trials indexed in dataset manifest")
    valid_trials: int = Field(..., description="Number of validated multimodal trials")
    runs: List[str] = Field(default_factory=list, description="List of recorded run IDs")


class ParticipantListResponse(BaseModel):
    """Response model for /api/participants."""
    participants: List[str] = Field(..., description="List of sorted unique participant IDs")
    details: List[ParticipantSummary] = Field(..., description="Detailed summary per participant")
    total_participants: int = Field(..., description="Total count of participants in cohort")
    data_source: DataSourceType = Field("recorded_replay", description="Data provenance flag")


class TrialSummary(BaseModel):
    """Metadata summary for a single canonical trial."""
    trial_id: str
    participant_id: str
    run_id: str
    trial_index: int
    label_code: int
    congruency: str
    target_goal: str
    physical_hand_direction: str
    ball_color: str
    valid: bool
    movement_duration_ms: Optional[float] = None
    exclusion_reason: Optional[str] = None


class TrialListResponse(BaseModel):
    """Response model for /api/trials."""
    trials: List[TrialSummary] = Field(..., description="List of trial summaries")
    total_trials: int = Field(..., description="Total trials returned")
    data_source: DataSourceType = Field("recorded_replay", description="Data provenance flag")


class NeuralSummary(BaseModel):
    """Pre-movement EEG neural intention decoder prediction summary."""
    predicted_goal: str = Field(..., description="'left' or 'right'")
    probability_left: float
    probability_right: float
    confidence: float = Field(..., description="Max posterior probability")
    prediction_time_ms: float = Field(0.0, description="Visual go-cue timestamp")
    model_name: str
    status: str = Field("AVAILABLE", description="Model execution status")


class ConflictStep(BaseModel):
    """Single-timestamp conflict assessment in a trial timeseries."""
    time_ms: float = Field(..., description="Timestamp in ms relative to movement execution start")
    state: str = Field(..., description="Conflict state e.g. ALIGNED, POTENTIAL_DISAGREEMENT, UNCERTAIN, UNAVAILABLE")
    dist_mm: Optional[float] = Field(None, description="Euclidean distance from forecast endpoint to target centroid in mm")
    norm_dist: Optional[float] = Field(None, description="Normalized distance score [0, 1]")
    explanation: str = Field("", description="Auditable diagnostic explanation")


class TrialDataResponse(BaseModel):
    """Complete trial kinematic and neural data for 3D visualization and replay."""
    trial_id: str
    participant_id: str
    congruency: str
    target_goal: str
    physical_hand_direction: str
    ball_color: str
    times_ms: List[float] = Field(..., description="Array of kinematic timestamps in ms")
    hand_x: List[float] = Field(..., description="Observed hand X trajectory in mm")
    hand_y: List[float] = Field(..., description="Observed hand Y trajectory in mm")
    hand_z: List[float] = Field(..., description="Observed hand Z trajectory in mm")
    predicted_endpoint: List[float] = Field(..., description="Forecasted 3D hand endpoint [x, y, z] mm")
    forecast_generation_time_ms: float = Field(..., description="Clock timestamp at which forecast is causally available")
    horizon_path: Optional[List[List[float]]] = Field(None, description="Predicted 3D trajectory path points [x, y, z] mm")
    neural: NeuralSummary = Field(..., description="Neural intention classification output")
    conflict_timeseries: List[ConflictStep] = Field(..., description="Conflict state evaluated at each timestep")
    target_coordinates: Optional[List[float]] = Field(None, description="Physical/centroid target coordinates [x, y, z] mm")
    coordinate_units: str = Field("mm", description="Spatial measurement units")
    coordinate_frame: str = Field("leap_motion_cartesian", description="Coordinate reference frame")
    data_source: DataSourceType = Field("recorded_replay", description="Data provenance flag")


class InferenceRequest(BaseModel):
    """Request payload for /api/run-inference."""
    trial_id: str = Field(..., description="Target trial ID to run live inference on")
    protocol: Optional[str] = Field("kinematic_onset", description="Forecast protocol: 'kinematic_onset' or 'event_locked'")


class ConflictAssessmentSummary(BaseModel):
    """Instantaneous conflict assessment summary."""
    conflict_state: str
    endpoint_target_distance_mm: Optional[float] = None
    directional_disagreement_deg: Optional[float] = None
    normalized_distance_score: Optional[float] = None
    neural_goal_agreement: Optional[bool] = None
    motor_forecast_agreement: Optional[bool] = None
    retrospective_endpoint_error_mm: Optional[float] = None
    explanation: str = ""
    is_conflict: bool = False


class ForecastSummary(BaseModel):
    """Kinematic trajectory forecast summary."""
    model_name: str
    forecast_protocol: str
    forecast_generation_time_ms: float
    predicted_endpoint: List[float]
    actual_endpoint: Optional[List[float]] = None
    endpoint_error_mm: Optional[float] = None
    horizon_path: Optional[List[List[float]]] = None
    status: str = "AVAILABLE"


class LiveInferenceResponse(BaseModel):
    """Live inference output response."""
    trial_id: str
    participant_id: str
    protocol: str
    neural: NeuralSummary
    trajectory_forecast: ForecastSummary
    conflict_assessment: ConflictAssessmentSummary
    data_source: DataSourceType = Field("live_inference", description="Data provenance flag")


class EEGSignalResponse(BaseModel):
    """Response model for /api/eeg-signal containing actual pre-movement EEG waveform."""
    channel: str = Field(..., description="Selected EEG channel name (e.g. 'CZ')")
    available_channels: List[str] = Field(..., description="List of all 26 available scalp channels")
    times_ms: List[float] = Field(..., description="Pre-movement time array from -1000 to 0 ms")
    amplitude_uv: List[float] = Field(..., description="Actual pre-movement EEG microvolt readings")
    processing: str = Field("filtfilt_offline_zero_phase", description="Signal preprocessing descriptor")
    srate_hz: float = Field(200.0, description="EEG sampling rate in Hz")
    trial_id: str = Field(..., description="Associated trial ID")
    data_source: DataSourceType = Field("recorded_replay", description="Data provenance flag")


class CohortResultsResponse(BaseModel):
    """Verified scientific benchmark results for the 5-participant cohort."""
    ridge_endpoint_rmse_mm: float = Field(47.25, description="3D endpoint RMSE of Regularized Ridge Forecaster")
    fair_cv_rmse_mm: float = Field(85.65, description="3D endpoint RMSE of Fair Constant-Velocity baseline")
    persistence_rmse_mm: float = Field(177.51, description="3D endpoint RMSE of Static Persistence baseline")
    ridge_vs_cv_reduction_pct: float = Field(44.8, description="Percentage error reduction of Ridge vs Fair Constant-Velocity")
    phase4_balanced_accuracy_pct: float = Field(66.46, description="Phase 4 intention-action alignment balanced accuracy")
    phase4_auroc: float = Field(0.6834, description="Phase 4 AUROC")
    phase5_cond_disc_balanced_accuracy_pct: float = Field(88.35, description="Phase 5 condition discrimination balanced accuracy")
    phase5_auroc: float = Field(0.9395, description="Phase 5 continuous discordance AUROC")
    phase5_note: str = Field(
        "Measures experimental condition discrimination (congruent vs. incongruent), NOT biological intention-action discordance",
        description="Mandatory scientific disclaimer on task condition vs biological ground truth"
    )
    n_valid_trials: int = Field(269, description="Total valid multimodal trials in benchmark cohort")
    n_participants: int = Field(5, description="Total participants in benchmark cohort")
    data_source: DataSourceType = Field("saved_result", description="Data provenance flag")


class ParticipantResultsResponse(BaseModel):
    """Audited participant-level metrics from Phase 5 prospective validation."""
    participant_id: str
    valid_trials: int
    persistence_rmse_mm: float
    fair_cv_rmse_mm: float
    ridge_forecaster_rmse_mm: float
    ridge_reduction_vs_fair_cv_pct: float
    neural_balanced_accuracy_pct: float
    condition_balanced_accuracy_pct: float
    policy_balanced_accuracy_pct: float
    policy_coverage_pct: float
    data_source: DataSourceType = Field("saved_result", description="Data provenance flag")


class LDACoefficientsResponse(BaseModel):
    """Response model for /api/lda-coefficients."""
    participant_id: str = Field(..., description="Participant ID or 'cohort'")
    model_name: str
    channels: List[str] = Field(..., description="Motor-focused channels evaluated")
    coefficients: List[float] = Field(..., description="LDA weight magnitude (L2 norm across features) per channel")
    interpretation: str = Field(..., description="Scientific interpretation of motor channel weights")
    data_source: DataSourceType = Field("saved_result", description="Data provenance flag")


class ModelInfoResponse(BaseModel):
    """Architectural and mathematical metadata for models in the twin."""
    neural_model: Dict[str, Any]
    kinematic_model: Dict[str, Any]
    fusion_and_conflict_engine: Dict[str, Any]
    validation_regime: str
    coordinate_reference: Dict[str, Any]
    data_source: DataSourceType = Field("saved_result", description="Data provenance flag")


class FeaturesInfoResponse(BaseModel):
    """Feature engineering pipeline metadata."""
    motor_channels: List[str]
    all_channels: List[str]
    total_channels: int
    eeg_sampling_rate_hz: float
    eeg_window_ms: List[float]
    time_domain_features: List[str]
    frequency_band_features: List[str]
    total_features_per_channel: int
    total_neural_features: int
    kinematic_sampling_rate_hz: float
    kinematic_prefix_ms: float
    kinematic_prefix_samples: int
    coordinate_frame: str
    coordinate_units: str
    targets: Dict[str, List[float]]
    data_source: DataSourceType = Field("saved_result", description="Data provenance flag")


class HealthResponse(BaseModel):
    """Service health response."""
    status: str = "ok"
    service: str = "neuroaction-backend"
    version: str = "1.0.0"
    data_source: DataSourceType = Field("saved_result", description="Data provenance flag")
