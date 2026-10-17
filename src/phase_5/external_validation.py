"""External validation dataset ingestion interface and schema validator."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd


REQUIRED_EXTERNAL_COLUMNS: List[str] = [
    "participant_id",
    "session_id",
    "trial_id",
    "intended_target_id",
    "target_x_mm",
    "target_y_mm",
    "target_z_mm",
    "target_tolerance_mm",
    "coordinate_frame",
    "coordinate_units",
    "actual_endpoint_x_mm",
    "actual_endpoint_y_mm",
    "actual_endpoint_z_mm",
    "target_success",
    "is_synthetic_example",
]

SUPPORTED_COORDINATE_FRAMES: List[str] = [
    "leap_motion_cartesian",
    "optitrack_cartesian_mm",
    "vicon_cartesian_mm",
]


class ExternalValidationIngestionError(ValueError):
    """Raised when an external dataset fails schema or coordinate validation."""
    pass


class ExternalValidationManager:
    """Manages schema validation and ingestion for prospective validation datasets."""

    @classmethod
    def validate_dataset_file(cls, csv_path: Path) -> Tuple[bool, Dict[str, Any]]:
        """Validate external CSV file against canonical Phase 5 schema."""
        if not csv_path.exists():
            return False, {"error": f"File not found at {csv_path}"}

        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            return False, {"error": f"Failed to parse CSV: {str(e)}"}

        # Check required columns
        missing = [col for col in REQUIRED_EXTERNAL_COLUMNS if col not in df.columns]
        if missing:
            return False, {"error": f"Missing required columns: {missing}"}

        # Check coordinate frames
        frames = df["coordinate_frame"].unique()
        unsupported = [f for f in frames if f not in SUPPORTED_COORDINATE_FRAMES]
        if unsupported:
            return False, {"error": f"Unsupported coordinate frames: {unsupported}. Supported: {SUPPORTED_COORDINATE_FRAMES}"}

        # Check units
        units = df["coordinate_units"].unique()
        if not all(u == "mm" for u in units):
            return False, {"error": f"Only 'mm' coordinate units currently supported, found: {units}"}

        # Check finite coordinates
        coord_cols = [
            "target_x_mm", "target_y_mm", "target_z_mm",
            "actual_endpoint_x_mm", "actual_endpoint_y_mm", "actual_endpoint_z_mm",
        ]
        for col in coord_cols:
            if not np.isfinite(df[col]).all():
                return False, {"error": f"Column {col} contains NaN or non-finite values."}

        # Check binary target success
        success_vals = set(df["target_success"].unique())
        if not success_vals.issubset({0, 1, True, False}):
            return False, {"error": f"target_success must be binary 0 or 1, found: {success_vals}"}

        is_synthetic = bool(df["is_synthetic_example"].all())

        return True, {
            "total_rows": len(df),
            "unique_participants": list(df["participant_id"].unique()),
            "is_synthetic_example": is_synthetic,
            "success_rate": float(df["target_success"].mean()),
        }

    @classmethod
    def generate_synthetic_template(cls, output_path: Path) -> Path:
        """Generate an explicitly labeled synthetic example CSV file for interface verification."""
        records = [
            {
                "participant_id": "sub-ext01",
                "session_id": "ses-01",
                "trial_id": "sub-ext01_ses-01_tr00",
                "intended_target_id": "left_target",
                "target_x_mm": -190.0,
                "target_y_mm": 150.0,
                "target_z_mm": 50.0,
                "target_tolerance_mm": 40.0,
                "coordinate_frame": "optitrack_cartesian_mm",
                "coordinate_units": "mm",
                "actual_endpoint_x_mm": -185.2,
                "actual_endpoint_y_mm": 148.1,
                "actual_endpoint_z_mm": 52.4,
                "target_success": 1,
                "is_synthetic_example": True,
            },
            {
                "participant_id": "sub-ext01",
                "session_id": "ses-01",
                "trial_id": "sub-ext01_ses-01_tr01",
                "intended_target_id": "right_target",
                "target_x_mm": 190.0,
                "target_y_mm": 150.0,
                "target_z_mm": 50.0,
                "target_tolerance_mm": 40.0,
                "coordinate_frame": "optitrack_cartesian_mm",
                "coordinate_units": "mm",
                "actual_endpoint_x_mm": 120.5,  # Missed reach
                "actual_endpoint_y_mm": 135.0,
                "actual_endpoint_z_mm": 45.0,
                "target_success": 0,
                "is_synthetic_example": True,
            },
            {
                "participant_id": "sub-ext02",
                "session_id": "ses-01",
                "trial_id": "sub-ext02_ses-01_tr00",
                "intended_target_id": "right_target",
                "target_x_mm": 190.0,
                "target_y_mm": 150.0,
                "target_z_mm": 50.0,
                "target_tolerance_mm": 40.0,
                "coordinate_frame": "optitrack_cartesian_mm",
                "coordinate_units": "mm",
                "actual_endpoint_x_mm": 188.7,
                "actual_endpoint_y_mm": 149.2,
                "actual_endpoint_z_mm": 51.0,
                "target_success": 1,
                "is_synthetic_example": True,
            },
        ]
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(records)
        df.to_csv(output_path, index=False)
        return output_path
