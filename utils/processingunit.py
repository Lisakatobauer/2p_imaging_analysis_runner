import hashlib
import json
import os
from pathlib import Path

import numpy as np
from datetime import datetime
from typing import Dict, Any, Optional, List
import shutil


class ProcessingUnit:
    """A mother class which handles multiple runs of a certain analysis process using hashing."""

    def __init__(self, processing_path: str):
        """
        Initialize the ProcessingUnit.

        Args:
            processing_path: Base path where processed data will be stored
        """
        self.processing_path = processing_path
        self.run_hash = None
        self.hash_dir = None
        self.current_ops = None
        self.best_run_hash = None
        self._init_hash_directories()

    def _init_hash_directories(self) -> None:
        """Initialize necessary directories for hash tracking."""
        os.makedirs(self.processing_path, exist_ok=True)
        os.makedirs(self._get_hash_root(), exist_ok=True)

    def _get_hash_root(self) -> str:
        """Get the root directory for storing run hashes."""
        return os.path.join(self.processing_path, 'run_hashes')

    def generate_run_hash(self, config) -> str:
        """
        Generate a unique hash for the current run configuration.

        Args:
            config: Object containing all relevant configuration parameters

        Returns:
            str: MD5 hash of the configuration
        """
        # Convert to JSON string for consistent hashing
        config_dict = config.to_dict()
        hash_str = json.dumps(config_dict, sort_keys=True, default=self._json_serializer)
        return hashlib.md5(hash_str.encode()).hexdigest()

    @staticmethod
    def _json_serializer(obj) -> Any:
        """Custom JSON serializer for non-standard types."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if hasattr(obj, '__dict__'):
            return obj.__dict__
        if isinstance(obj, Path):
            return str(obj)
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    def setup_run(self, run_params: Dict) -> str:
        """
        Set up a new processing run with hash tracking.

        Args:
            run_params: Dictionary of parameters for this run

        Returns:
            str: The generated run hash
        """
        # Generate and store the run hash
        self.run_hash = self.generate_run_hash(run_params)
        self.hash_dir = os.path.join(self._get_hash_root(), self.run_hash)
        os.makedirs(self.hash_dir, exist_ok=True)

        # Save metadata
        self.save_metadata(run_params)
        return self.run_hash

    def save_metadata(self, run_params: Dict) -> None:
        """Save metadata about the current run."""
        metadata = {
            "hash": self.run_hash,
            "timestamp": datetime.now().isoformat(),
            "parameters": run_params,
            "ops": getattr(self, 'current_ops', None)
        }

        with open(os.path.join(self.hash_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)

    def check_existing_run(self, run_hash: str = None) -> bool:
        """
        Check if a run with the given hash (or current run) already exists.

        Args:
            run_hash: Optional specific hash to check (defaults to current run)

        Returns:
            bool: True if run exists, False otherwise
        """
        hash_to_check = run_hash or self.run_hash
        return hash_to_check is not None and os.path.exists(
            os.path.join(self._get_hash_root(), hash_to_check)
        )

    def get_run_folder(self, run_hash: str = None) -> str:
        """
        Get the processing path for a specific run.

        Args:
            run_hash: Optional specific hash (defaults to current run)

        Returns:
            str: Full path to the run folder
        """
        run_hash = run_hash or self.run_hash
        return os.path.join(self.processing_path, run_hash)

    def mark_as_best_run(self, run_hash: str = None) -> None:
        """
        Mark a specific run as the "best" reference run.

        Args:
            run_hash: Optional specific hash (defaults to current run)
        """
        run_hash = run_hash or self.run_hash
        best_run_file = os.path.join(self.processing_path, 'best_run_hash.npy')
        np.save(best_run_file, run_hash)
        self.best_run_hash = run_hash

    def get_best_run(self) -> Optional[str]:
        """Get the hash of the best run if one exists."""
        best_run_file = os.path.join(self.processing_path, 'best_run_hash.npy')
        return np.load(best_run_file).item() if os.path.exists(best_run_file) else None

    def get_all_runs(self) -> List[str]:
        """Get a list of all run hashes."""
        hash_root = self._get_hash_root()
        if not os.path.exists(hash_root):
            return []
        return [d for d in os.listdir(hash_root)
                if os.path.isdir(os.path.join(hash_root, d))]

    def get_run_metadata(self, run_hash: str) -> Optional[Dict]:
        """Get metadata for a specific run."""
        metadata_file = os.path.join(self._get_hash_root(), run_hash, "metadata.json")
        if os.path.exists(metadata_file):
            with open(metadata_file, "r") as f:
                return json.load(f)
        return None

    def find_matching_run(self, config: Dict) -> Optional[str]:
        """
        Find an existing run that matches the given configuration.

        Args:
            config: Configuration parameters to match

        Returns:
            Optional[str]: Matching run hash if found, None otherwise
        """
        target_hash = self.generate_run_hash(config)
        for run_hash in self.get_all_runs():
            if run_hash == target_hash:
                return run_hash
        return None

    def copy_best_run_outputs(self, destination: str) -> None:
        """
        Copy outputs from the best run to a destination folder.

        Args:
            destination: Path to copy outputs to
        """
        best_run = self.get_best_run()
        if best_run:
            src = self.get_run_folder(best_run)
            shutil.copytree(src, destination, dirs_exist_ok=True)