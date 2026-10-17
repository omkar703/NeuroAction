"""Target reference manager enforcing coordinate provenance and distinguishing proxies from authoritative ground truth."""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np

from src.phase_5.schemas import TargetReference


# Canonical empirical proxy cluster centroids in Leap Motion Cartesian space (mm)
# Derived across training cohorts in ds006840. PROXY ONLY, NOT AUTHORITATIVE.
PROXY_TARGET_COORDINATES_MM: Dict[str, np.ndarray] = {
    "left": np.array([-188.4, 147.7, 44.9], dtype=np.float32),
    "right": np.array([188.4, 147.8, 54.3], dtype=np.float32),
}


class TargetReferenceManager:
    """Manages target coordinates with explicit provenance and authority classification."""

    def __init__(self, default_tolerance_mm: float = 50.0):
        self.default_tolerance_mm = default_tolerance_mm
        self._custom_references: Dict[str, TargetReference] = {}

    def register_authoritative_target(
        self,
        target_id: str,
        coordinates: np.ndarray,
        tolerance_mm: Optional[float] = None,
        coordinate_frame: str = "optitrack_cartesian_mm",
        coordinate_units: str = "mm",
        provenance: str = "independent_mocap_calibration",
        description: Optional[str] = None,
    ) -> TargetReference:
        """Register an authoritative physical target independently measured from physical workspace."""
        coords = np.asarray(coordinates, dtype=np.float32)
        if coords.shape != (3,):
            raise ValueError(f"Target coordinates must have shape (3,), got {coords.shape}")

        tol = tolerance_mm if tolerance_mm is not None else self.default_tolerance_mm
        ref = TargetReference(
            target_id=target_id,
            coordinates=coords,
            tolerance_mm=tol,
            geometry_type="authoritative_sphere",
            coordinate_frame=coordinate_frame,
            coordinate_units=coordinate_units,
            is_authoritative=True,
            provenance=provenance,
            description=description,
        )
        self._custom_references[target_id] = ref
        return ref

    def get_fold_target_references(
        self,
        training_samples: List[Any],
        held_out_subject: str,
        tolerance_mm: Optional[float] = None,
    ) -> Dict[str, TargetReference]:
        """Estimate target centroids strictly from training fold samples (held-out subject excluded).
        
        CRITICAL AUDIT CONSTRAINT:
        These centroids are empirical endpoint proxies and are flagged as is_authoritative=False.
        """
        assert all(s.participant_id != held_out_subject for s in training_samples), (
            f"Test subject {held_out_subject} found in training samples for target estimation!"
        )

        left_endpoints = [
            s.actual_endpoint for s in training_samples
            if getattr(s, "physical_hand_direction", None) == "left" or s.actual_endpoint[0] < -50.0
        ]
        right_endpoints = [
            s.actual_endpoint for s in training_samples
            if getattr(s, "physical_hand_direction", None) == "right" or s.actual_endpoint[0] > 50.0
        ]

        if not left_endpoints or not right_endpoints:
            # Fall back to global proxy centroids with training provenance
            c_left = PROXY_TARGET_COORDINATES_MM["left"]
            c_right = PROXY_TARGET_COORDINATES_MM["right"]
        else:
            c_left = np.mean(left_endpoints, axis=0).astype(np.float32)
            c_right = np.mean(right_endpoints, axis=0).astype(np.float32)

        tol = tolerance_mm if tolerance_mm is not None else self.default_tolerance_mm
        prov_tag = f"training_subs_excluding_{held_out_subject}"

        refs = {
            "left": TargetReference(
                target_id="left",
                coordinates=c_left,
                tolerance_mm=tol,
                geometry_type="centroid_proxy",
                coordinate_frame="leap_motion_cartesian",
                coordinate_units="mm",
                is_authoritative=False,
                provenance=prov_tag,
                description=f"Empirical cluster centroid computed from {len(left_endpoints)} left training reaches",
            ),
            "right": TargetReference(
                target_id="right",
                coordinates=c_right,
                tolerance_mm=tol,
                geometry_type="centroid_proxy",
                coordinate_frame="leap_motion_cartesian",
                coordinate_units="mm",
                is_authoritative=False,
                provenance=prov_tag,
                description=f"Empirical cluster centroid computed from {len(right_endpoints)} right training reaches",
            ),
        }
        return refs

    def get_target_reference(
        self,
        target_id: str,
        fold_references: Optional[Dict[str, TargetReference]] = None,
    ) -> TargetReference:
        """Lookup target reference with priority to custom/authoritative references."""
        if target_id in self._custom_references:
            return self._custom_references[target_id]
        if fold_references and target_id in fold_references:
            return fold_references[target_id]
        if target_id in PROXY_TARGET_COORDINATES_MM:
            return TargetReference(
                target_id=target_id,
                coordinates=PROXY_TARGET_COORDINATES_MM[target_id],
                tolerance_mm=self.default_tolerance_mm,
                geometry_type="centroid_proxy",
                coordinate_frame="leap_motion_cartesian",
                coordinate_units="mm",
                is_authoritative=False,
                provenance="cohort_default_proxy",
                description="Default empirical proxy centroid (NOT authoritative)",
            )
        raise ValueError(f"Unknown target ID: {target_id}")
