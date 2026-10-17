import json
import os
import requests
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class OpenNeuroClient:
    """Client for querying and downloading IACKD files from OpenNeuro S3 storage."""

    DEFAULT_CATALOG_PATH = Path("data/manifests/openneuro_remote_catalog.json")
    DEFAULT_CACHE_DIR = Path("data/cache")

    def __init__(
        self,
        catalog_path: Optional[Path] = None,
        cache_dir: Optional[Path] = None,
    ):
        self.catalog_path = Path(catalog_path or self.DEFAULT_CATALOG_PATH)
        self.cache_dir = Path(cache_dir or self.DEFAULT_CACHE_DIR)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        if not self.catalog_path.exists():
            raise FileNotFoundError(
                f"Catalog file {self.catalog_path} not found. "
                "Ensure openneuro_remote_catalog.json is generated."
            )
        with open(self.catalog_path, "r", encoding="utf-8") as f:
            self.catalog = json.load(f)

    def list_subjects(self) -> List[str]:
        """Return sorted list of all 15 subjects in the dataset."""
        return sorted(list(self.catalog.get("pre-movement", {}).keys()))

    def list_subject_files(self, subject: str, branch: str = "pre-movement") -> List[Dict]:
        """Return list of files for a subject and branch ('pre-movement' or 'movement-execution')."""
        return self.catalog.get(branch, {}).get(subject, {}).get("ihc", [])

    def download_file(self, url: str, target_path: Path, force: bool = False) -> Path:
        """Download a file from OpenNeuro S3 if not already cached."""
        target_path.parent.mkdir(parents=True, exist_ok=True)
        if target_path.exists() and not force:
            return target_path

        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()

        tmp_path = target_path.with_suffix(target_path.suffix + ".tmp")
        with open(tmp_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
        tmp_path.rename(target_path)
        return target_path

    def get_run_files(
        self, subject: str, run_name: str, force: bool = False
    ) -> Tuple[Path, Path]:
        """Ensure both pre-movement and movement-execution files exist locally for a given subject and run.
        
        Example run_name: 'run-01_L1', 'run-01_pre_eeg_L1', or 'run-01_eeg_leap_L1'
        Returns:
            Tuple[Path, Path]: (local_pre_movement_path, local_movement_execution_path)
        """
        # Match files in pre-movement
        pre_files = self.list_subject_files(subject, "pre-movement")
        pre_entry = None
        
        # Tokenize run_name by '_' or '-'
        clean_run = run_name.replace(".mat", "")
        tokens = [t for t in clean_run.split("_") if t and t not in ("pre", "eeg", "task-ihc")]
        
        for item in pre_files:
            fname = item["filename"]
            if not fname.endswith(".mat"):
                continue
            if clean_run in fname:
                pre_entry = item
                break
            if all(t in fname for t in tokens):
                pre_entry = item
                break

        if not pre_entry:
            raise ValueError(f"Could not find pre-movement file for {subject} with run {run_name}")

        # Match files in movement-execution
        me_files = self.list_subject_files(subject, "movement-execution")
        # Extract run key, e.g. 'run-01_pre_eeg_L1' -> 'run-01' and 'L1'
        parts = pre_entry["filename"].replace(".mat", "").split("_")
        run_tag = next(p for p in parts if p.startswith("run-"))
        hand_tag = parts[-1]  # e.g. 'L1'
        
        me_entry = None
        for item in me_files:
            fname = item["filename"]
            if not fname.endswith(".mat"):
                continue
            if run_tag in fname and hand_tag in fname:
                me_entry = item
                break

        if not me_entry:
            raise ValueError(f"Could not find movement-execution file for {subject} matching {run_tag}_{hand_tag}")

        pre_local = self.cache_dir / subject / pre_entry["filename"]
        me_local = self.cache_dir / subject / me_entry["filename"]

        self.download_file(pre_entry["url"], pre_local, force=force)
        self.download_file(me_entry["url"], me_local, force=force)

        return pre_local, me_local
