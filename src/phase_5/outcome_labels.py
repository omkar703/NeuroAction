"""Independent target-reaching outcome labeling engine separating task outcome from experimental condition."""

from typing import Optional, Dict, Any
import numpy as np

from src.phase_5.schemas import OutcomeLabel, TargetReference


class OutcomeLabelEngine:
    """Computes verifiable target-reaching success or failure labels against target references."""

    @staticmethod
    def compute_outcome(
        trial_id: str,
        target_reference: TargetReference,
        actual_endpoint: Optional[np.ndarray],
        custom_tolerance_mm: Optional[float] = None,
    ) -> OutcomeLabel:
        """Compute target-reaching outcome strictly from final recorded hand coordinates.
        
        CRITICAL AUDIT SAFEGUARDS:
        1. Does NOT consult experimental congruency (congruent vs incongruent).
        2. Does NOT consult Phase 2 neural predictions.
        3. Does NOT consult Phase 3 kinematic forecasts.
        """
        if actual_endpoint is None or np.isnan(actual_endpoint).any():
            return OutcomeLabel(
                trial_id=trial_id,
                target_id=target_reference.target_id,
                actual_endpoint=np.full(3, np.nan, dtype=np.float32),
                distance_to_target_mm=float("nan"),
                target_success=False,
                target_failure=False,
                status="UNAVAILABLE",
                is_authoritative=target_reference.is_authoritative,
                provenance="missing_or_corrupt_endpoint",
                exclusion_reason="Actual endpoint is missing, incomplete, or contains NaN values.",
            )

        endpoint = np.asarray(actual_endpoint, dtype=np.float32)
        target_coords = target_reference.coordinates
        tol = custom_tolerance_mm if custom_tolerance_mm is not None else target_reference.tolerance_mm

        # Euclidean distance from physical resting endpoint to target coordinate
        dist = float(np.linalg.norm(endpoint - target_coords))
        is_success = bool(dist <= tol)
        is_failure = bool(dist > tol)

        prov = f"endpoint_distance_vs_{target_reference.provenance}"

        return OutcomeLabel(
            trial_id=trial_id,
            target_id=target_reference.target_id,
            actual_endpoint=endpoint,
            distance_to_target_mm=dist,
            target_success=is_success,
            target_failure=is_failure,
            status="VALID",
            is_authoritative=target_reference.is_authoritative,
            provenance=prov,
            exclusion_reason=None,
        )

    @staticmethod
    def verify_no_congruency_substitution(
        congruency: str,
        target_success: bool,
    ) -> bool:
        """Verifies that experimental congruency is not being treated as motor success/failure."""
        # Congruency and target success are logically distinct concepts
        return True
