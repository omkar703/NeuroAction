"""NeuroAction Phase 5: Independent Scientific Validation and Prospective Evaluation."""

from src.phase_5.config import Phase5Config
from src.phase_5.schemas import (
    TargetReference,
    OutcomeLabel,
    EligibilityRecord,
    Phase5Prediction,
    Phase5EvaluationRow,
    TaskMetricsSummary,
)
from src.phase_5.data_audit import DataAuditor
from src.phase_5.target_reference import TargetReferenceManager
from src.phase_5.outcome_labels import OutcomeLabelEngine
from src.phase_5.temporal_validation import TemporalValidator
from src.phase_5.provenance import ProvenanceTracker
from src.phase_5.fold_manager import LOSOFoldCoordinator
from src.phase_5.prediction_policy import PredictionPolicy, DecisionPolicyState
from src.phase_5.fusion_model import MultimodalOutcomeFusionModel
from src.phase_5.statistical_analysis import StatisticalAnalyzer
from src.phase_5.calibration import ProbabilityCalibrator
from src.phase_5.robustness import RobustnessAnalyzer
from src.phase_5.external_validation import ExternalValidationManager
from src.phase_5.evaluation import Phase5Evaluator

__all__ = [
    "Phase5Config",
    "TargetReference",
    "OutcomeLabel",
    "EligibilityRecord",
    "Phase5Prediction",
    "Phase5EvaluationRow",
    "TaskMetricsSummary",
    "DataAuditor",
    "TargetReferenceManager",
    "OutcomeLabelEngine",
    "TemporalValidator",
    "ProvenanceTracker",
    "LOSOFoldCoordinator",
    "PredictionPolicy",
    "DecisionPolicyState",
    "MultimodalOutcomeFusionModel",
    "StatisticalAnalyzer",
    "ProbabilityCalibrator",
    "RobustnessAnalyzer",
    "ExternalValidationManager",
    "Phase5Evaluator",
]
