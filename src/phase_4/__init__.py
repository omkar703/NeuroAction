"""NeuroAction Phase 4 — Neural Intention-Action Digital Twin.

Multimodal fusion, continuous trajectory forecasting, intention-action conflict detection,
and deterministic trial replay engine.
"""

from src.phase_4.config import DigitalTwinConfig
from src.phase_4.schemas import (
    TrialMetadata,
    NeuralIntentionResult,
    KinematicObservation,
    TrajectoryForecast,
    ConflictAssessment,
    FusionState,
    ReplayFrame,
)

__all__ = [
    "DigitalTwinConfig",
    "TrialMetadata",
    "NeuralIntentionResult",
    "KinematicObservation",
    "TrajectoryForecast",
    "ConflictAssessment",
    "FusionState",
    "ReplayFrame",
]
