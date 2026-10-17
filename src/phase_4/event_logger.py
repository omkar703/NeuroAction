"""Structured event logging, reproducibility manifests, and trial export utilities for Phase 4."""

from pathlib import Path
from typing import Dict, List, Optional, Any
import datetime
import hashlib
import json
import os
import platform
import sys
import numpy as np
import pandas as pd

from src.phase_4.config import DigitalTwinConfig
from src.phase_4.schemas import ReplayFrame


def compute_file_sha256(file_path: Path) -> str:
    """Compute hex SHA-256 digest of a local file."""
    if not file_path.exists():
        return "file_not_found"
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


class EventLogger:
    """Logs simulation frames, conflict state transitions, and exports reproducibility manifests."""

    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir or Path("reports/phase_4/demo")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export_event_log_csv(
        self,
        frames: List[ReplayFrame],
        filename: str = "demo_replay_event_log.csv",
    ) -> Path:
        """Export tabular time-series log of replay frames."""
        records = []
        for f in frames:
            s = f.fusion_state
            c = s.conflict_assessment
            k = s.current_kinematics
            n = s.neural_result
            r = {
                "frame_index": f.frame_index,
                "timestamp_ms": f.timestamp_ms,
                "participant_id": s.participant_id,
                "trial_id": s.trial_id,
                "hand_x_mm": float(k.position[0]) if k and k.position is not None else None,
                "hand_y_mm": float(k.position[1]) if k and k.position is not None else None,
                "hand_z_mm": float(k.position[2]) if k and k.position is not None else None,
                "hand_speed_mm_s": float(k.speed) if k and k.speed is not None else None,
                "neural_goal": n.predicted_goal if n else None,
                "neural_confidence": float(n.confidence) if n else None,
                "neural_status": n.status if n else None,
                "conflict_state": c.conflict_state if c else None,
                "endpoint_target_distance_mm": c.endpoint_target_distance_mm if c else None,
                "directional_angle_deg": c.directional_disagreement_deg if c else None,
                "normalized_distance_score": c.normalized_distance_score if c else None,
                "is_conflict": c.is_conflict if c else None,
                "explanation": c.explanation if c else None,
            }
            records.append(r)

        df = pd.DataFrame(records)
        out_path = self.output_dir / filename
        df.to_csv(out_path, index=False)
        return out_path

    def export_trial_summary_json(
        self,
        final_frame: ReplayFrame,
        filename: str = "demo_trial_summary.json",
    ) -> Path:
        """Export comprehensive trial summary dictionary to JSON."""
        summary = {
            "export_timestamp": datetime.datetime.now().isoformat(),
            "trial_summary": final_frame.fusion_state.to_dict(),
        }
        out_path = self.output_dir / filename
        with open(out_path, "w") as fp:
            json.dump(summary, fp, indent=2)
        return out_path

    def export_reproducibility_manifest(
        self,
        config: DigitalTwinConfig,
        filename: str = "reproducibility_manifest.json",
    ) -> Path:
        """Export complete runtime provenance manifest for audits."""
        p2_model_full = config.model_dir_phase_2 / "lda_goal_decoder.pkl"
        p2_model_sub01 = config.model_dir_phase_2 / "lda_goal_decoder_held_out_sub01.pkl"
        p3_model_full = config.model_dir_phase_3 / "kinematic_forecaster.pkl"
        p3_model_sub01 = config.model_dir_phase_3 / "kinematic_forecaster_held_out_sub01.pkl"

        manifest = {
            "generation_timestamp": datetime.datetime.now().isoformat(),
            "python_version": sys.version,
            "platform": platform.platform(),
            "repository": "NeuroAction — Neural Intention-Action Digital Twin",
            "phase": 4,
            "configuration": {
                "distance_reference_mm": config.distance_reference_mm,
                "distance_threshold_mm": config.distance_threshold_mm,
                "directional_threshold_deg": config.directional_threshold_deg,
                "confidence_uncertainty_threshold": config.confidence_uncertainty_threshold,
                "speed_threshold_mm_s": config.speed_threshold_mm_s,
                "prefix_duration_ms": config.prefix_duration_ms,
                "demo_trial_id": config.demo_trial_id,
                "default_protocol": config.default_protocol,
                "target_left_coord_mm": config.target_left_coord,
                "target_right_coord_mm": config.target_right_coord,
            },
            "model_artifacts": {
                "phase_2_lda_full": {
                    "path": str(p2_model_full),
                    "sha256": compute_file_sha256(p2_model_full),
                },
                "phase_2_lda_held_out_sub01": {
                    "path": str(p2_model_sub01),
                    "sha256": compute_file_sha256(p2_model_sub01),
                },
                "phase_3_ridge_full": {
                    "path": str(p3_model_full),
                    "sha256": compute_file_sha256(p3_model_full),
                },
                "phase_3_ridge_held_out_sub01": {
                    "path": str(p3_model_sub01),
                    "sha256": compute_file_sha256(p3_model_sub01),
                },
            },
        }

        out_path = self.output_dir / filename
        with open(out_path, "w") as fp:
            json.dump(manifest, fp, indent=2)
        return out_path
