"""Configurable decision policy managing prediction states and principled abstention."""

from typing import Dict, Any, Optional
import numpy as np


class DecisionPolicyState:
    """Standardized decision states."""
    PREDICTED_SUCCESS = "PREDICTED_SUCCESS"
    PREDICTED_FAILURE = "PREDICTED_FAILURE"
    UNCERTAIN = "UNCERTAIN"
    UNAVAILABLE = "UNAVAILABLE"
    INVALID_INPUT = "INVALID_INPUT"


class PredictionPolicy:
    """Evaluates geometric and confidence boundaries to make definite predictions or abstain."""

    def __init__(
        self,
        tolerance_mm: float = 50.0,
        abstention_margin_mm: float = 10.0,
        min_confidence: float = 0.55,
    ):
        self.tolerance_mm = tolerance_mm
        self.abstention_margin_mm = abstention_margin_mm
        self.min_confidence = min_confidence

    def evaluate_target_outcome(
        self,
        distance_to_target_mm: float,
        confidence_score: float = 1.0,
        is_input_valid: bool = True,
        has_target_reference: bool = True,
    ) -> str:
        """Determine decision policy state from predicted target distance and model confidence."""
        if not is_input_valid:
            return DecisionPolicyState.INVALID_INPUT
        if not has_target_reference or np.isnan(distance_to_target_mm):
            return DecisionPolicyState.UNAVAILABLE

        # Low confidence triggers abstention
        if confidence_score < self.min_confidence:
            return DecisionPolicyState.UNCERTAIN

        # Borderline distance around target tolerance boundary triggers abstention
        lower_bound = self.tolerance_mm - self.abstention_margin_mm
        upper_bound = self.tolerance_mm + self.abstention_margin_mm

        if lower_bound < distance_to_target_mm <= upper_bound:
            return DecisionPolicyState.UNCERTAIN
        elif distance_to_target_mm <= lower_bound:
            return DecisionPolicyState.PREDICTED_SUCCESS
        else:
            return DecisionPolicyState.PREDICTED_FAILURE

    @staticmethod
    def is_coverage_decision(state: str) -> bool:
        """True if the system took a definitive binary stance rather than abstaining."""
        return state in {
            DecisionPolicyState.PREDICTED_SUCCESS,
            DecisionPolicyState.PREDICTED_FAILURE,
        }

    @staticmethod
    def is_abstention(state: str) -> bool:
        """True if the system abstained from making a definitive decision."""
        return state in {
            DecisionPolicyState.UNCERTAIN,
            DecisionPolicyState.UNAVAILABLE,
            DecisionPolicyState.INVALID_INPUT,
        }
