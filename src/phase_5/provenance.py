"""Cryptographic artifact provenance and execution metadata tracker."""

import hashlib
from pathlib import Path
import platform
import sys
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd


class ProvenanceTracker:
    """Tracks model checksums, training splits, and runtime environment specifications."""

    @staticmethod
    def compute_file_hash(file_path: Path) -> str:
        """Compute SHA-256 hash of a persisted file."""
        if not file_path.exists():
            return "FILE_NOT_FOUND"
        hasher = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    @classmethod
    def get_system_provenance(cls, model_paths: Optional[Dict[str, Path]] = None) -> Dict[str, Any]:
        """Collect complete system and artifact provenance manifest."""
        hashes = {}
        if model_paths:
            for name, path in model_paths.items():
                hashes[name] = {
                    "path": str(path),
                    "exists": path.exists(),
                    "sha256": cls.compute_file_hash(path) if path.exists() else None,
                }

        return {
            "platform": platform.platform(),
            "python_version": sys.version,
            "numpy_version": np.__version__,
            "pandas_version": pd.__version__,
            "model_hashes": hashes,
        }

    @staticmethod
    def verify_provenance_cleanliness(
        training_subjects: list,
        test_subject: str,
    ) -> bool:
        """Verify that training set contains zero instances of test subject."""
        if test_subject in training_subjects:
            raise ValueError(f"Data leakage detected! Test subject {test_subject} present in training set: {training_subjects}")
        return True
