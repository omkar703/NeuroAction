"""Coordinate frame validation, unit conversions, and spatial target mapping for Phase 4."""

from typing import Dict, Optional, Tuple
import numpy as np


# Empirical Leap Motion Cartesian Target Centroid Proxies (in millimeters)
# Estimated from reaching endpoints across OpenNeuro ds006840 (not independent physical markers)
TARGET_COORDINATES_MM: Dict[str, np.ndarray] = {
    "left": np.array([-188.0, 150.0, 53.0], dtype=np.float32),
    "right": np.array([187.5, 149.0, 51.0], dtype=np.float32),
}

STANDARD_COORDINATE_FRAME = "leap_motion_cartesian"
STANDARD_UNITS = "mm"


def estimate_fold_target_centroids(
    train_samples: list,
) -> Dict[str, np.ndarray]:
    """Estimate fold-specific target centroid proxies strictly from training participants' congruent reaches.
    
    Prevents cross-participant target-mapping leakage under LOSO cross-validation.
    """
    left_pts = [
        s.actual_endpoint
        for s in train_samples
        if s.congruency == "congruent" and s.target_goal == "left" and not np.isnan(s.actual_endpoint).any()
    ]
    right_pts = [
        s.actual_endpoint
        for s in train_samples
        if s.congruency == "congruent" and s.target_goal == "right" and not np.isnan(s.actual_endpoint).any()
    ]

    c_left = (
        np.mean(left_pts, axis=0).astype(np.float32)
        if len(left_pts) > 0
        else TARGET_COORDINATES_MM["left"].copy()
    )
    c_right = (
        np.mean(right_pts, axis=0).astype(np.float32)
        if len(right_pts) > 0
        else TARGET_COORDINATES_MM["right"].copy()
    )

    return {
        "left": c_left,
        "right": c_right,
    }


def get_target_coordinates(
    goal_label: Optional[str],
    units: str = STANDARD_UNITS,
    allow_unmapped: bool = False,
    fold_centroids: Optional[Dict[str, np.ndarray]] = None,
) -> Optional[np.ndarray]:
    """Retrieve 3D spatial target coordinates corresponding to an abstract goal label.
    
    Args:
        goal_label: "left" or "right" (case-insensitive)
        units: Target coordinate units (must be "mm" or convertible)
        allow_unmapped: If True, returns None on unknown label instead of raising ValueError
        fold_centroids: Optional fold-specific centroid dictionary (estimated strictly on training fold)
    Returns:
        3D numpy array [x, y, z] in specified units, or None if unmapped
    """
    if goal_label is None:
        if allow_unmapped:
            return None
        raise ValueError("goal_label cannot be None")

    norm_goal = goal_label.strip().lower()
    centroids_map = fold_centroids if fold_centroids is not None else TARGET_COORDINATES_MM

    if norm_goal not in centroids_map:
        if allow_unmapped:
            return None
        raise ValueError(
            f"Target mapping unknown for goal '{goal_label}'. Expected 'left' or 'right'."
        )

    base_coords = centroids_map[norm_goal].copy()
    if units != STANDARD_UNITS:
        base_coords = convert_units(base_coords, from_units=STANDARD_UNITS, to_units=units)
    return base_coords


def convert_units(
    data: np.ndarray,
    from_units: str,
    to_units: str,
) -> np.ndarray:
    """Convert Cartesian coordinates between length units."""
    if from_units == to_units:
        return data.copy()

    scale_to_meters = {
        "mm": 1e-3,
        "cm": 1e-2,
        "m": 1.0,
    }

    if from_units not in scale_to_meters or to_units not in scale_to_meters:
        raise ValueError(f"Unsupported unit conversion from '{from_units}' to '{to_units}'")

    meters = data * scale_to_meters[from_units]
    return meters / scale_to_meters[to_units]


def validate_coordinate_compatibility(
    frame_a: str,
    frame_b: str,
    units_a: str,
    units_b: str,
) -> Tuple[bool, Optional[str]]:
    """Validate that two coordinate representations share identical frames and units.
    
    Returns:
        (is_compatible, reason_if_incompatible)
    """
    if frame_a != frame_b:
        return (
            False,
            f"Coordinate frame mismatch: '{frame_a}' vs '{frame_b}'. Transformations required.",
        )
    if units_a != units_b:
        return (
            False,
            f"Coordinate unit mismatch: '{units_a}' vs '{units_b}'. Conversion required.",
        )
    return True, None


def classify_endpoint_lateral_direction(
    position: np.ndarray,
    lateral_deadzone_mm: float = 20.0,
) -> str:
    """Classify 3D position into lateral direction ('left', 'right', or 'center').
    
    Args:
        position: Array of shape (3,) [x, y, z] in mm
        lateral_deadzone_mm: Tolerance around x=0 to designate as 'center'
    """
    if position is None or len(position) < 1 or np.isnan(position[0]):
        return "unknown"
    x = position[0]
    if x < -lateral_deadzone_mm:
        return "left"
    elif x > lateral_deadzone_mm:
        return "right"
    else:
        return "center"
