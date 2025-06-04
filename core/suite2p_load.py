from pathlib import Path
from typing import Dict, List, Union
import numpy as np


class Suite2pDataLoader:
    """A class to load and manage Suite2p processed data with caching and coordinate transformations."""

    def __init__(
            self,
            suite2ppath_processed: str,
            suite2ppath_raw: str,
            imagingfiles: List[str],
            exptype: str,
            expnum: Union[int, str],
            animalnum: Union[int, str],
            number_planes: int,
            experiments: Dict,
            framerate: float,
            date: str,
            experiment_lengths: Dict
    ):
        """
        Initialize the Suite2p data loader.

        Args:
            suite2ppath_processed: Path to processed Suite2p data
            suite2ppath_raw: Path to raw imaging data
            imagingfiles: List of imaging files
            exptype: Experiment type identifier
            expnum: Experiment number
            animalnum: Animal identifier
            number_planes: Number of imaging planes
            experiments: Dictionary of experiment metadata
            framerate: Imaging frame rate (Hz)
            date: Experiment date
            experiment_lengths: Dictionary of experiment lengths
        """
        self.suite2ppath_processed = Path(suite2ppath_processed)
        self.suite2ppath_raw = Path(suite2ppath_raw)
        self.imagingfiles = imagingfiles
        self.exptype = exptype
        self.expnum = expnum
        self.animalnum = animalnum
        self.number_planes = number_planes
        self.experiments = experiments
        self.framerate = framerate
        self.date = date
        self.experiment_lengths = experiment_lengths

        # State tracking
        self._suite2p_cache = None
        self.isloaded = False
        self.isprocessed = False
        self.run_hash = None

        # Ensure directories exist
        self._ensure_directories()

    def _ensure_directories(self) -> None:
        """Ensure required directories exist."""
        self.suite2ppath_processed.mkdir(parents=True, exist_ok=True)

    def _load_plane_data(self, plane_n: int) -> List:
        """
        Load data for a specific plane.

        Args:
            plane_n: Plane index to load

        Returns:
            List containing [stat, iscell, F, ops, spks] data
        """
        files_to_load = ["stat.npy", "iscell.npy", "F.npy", "ops.npy", "spks.npy"]
        plane_path = self.suite2ppath_processed / f'plane{plane_n}'

        if not plane_path.exists():
            raise FileNotFoundError(f"Plane directory not found: {plane_path}")

        loaded_data = []
        for file in files_to_load:
            file_path = plane_path / file
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")

            data = np.load(file_path, allow_pickle=True)
            # ops.npy needs .item() called on it
            if file == "ops.npy":
                data = data.item()
            loaded_data.append(data)

        return loaded_data

    @property
    def load_or_process_data(self) -> Dict[str, List]:
        """
        Load existing Suite2p data or process if not available.

        Returns:
            Dictionary mapping plane numbers to loaded data
        """
        if self._suite2p_cache is not None:
            return self._suite2p_cache

        loaded_data = {}
        try:
            print(f"Attempting to load existing data for hash: {self.run_hash}")
            for plane_n in range(self.number_planes):
                loaded_data[str(plane_n)] = self._load_plane_data(plane_n)
            self.isloaded = True
            print("Successfully loaded existing Suite2p data")

            try:
                print("Attempting to load processed suite2p data...")
                for plane_n in range(self.number_planes):
                    loaded_data[str(plane_n)] = self._load_plane_data(plane_n)
                self.isloaded = True
                print("Successfully loaded processed data")
            except Exception as e:
                print(f"Failed to load processed suite2p data: {e}")
        except Exception as e:
            print(f"Failed to load existing suite2p data: {e}")

        self._suite2p_cache = loaded_data
        return loaded_data

    @property
    def suite2p_data(self) -> Dict[str, List]:
        """Property to access Suite2p data with lazy loading."""
        if self._suite2p_cache is None:
            self._suite2p_cache = self.load_or_process_data
        return self._suite2p_cache

    def get_fluorescence_traces(self, plane_n: int = 0) -> np.ndarray:
        """Get fluorescence traces for specified plane, filtered by cell IDs."""
        cell_ids = self.get_cell_ids(plane_n)
        return self.suite2p_data[str(plane_n)][2][cell_ids, :]

    def get_spike_traces(self, plane_n: int = 0) -> np.ndarray:
        """Get spike traces for specified plane, filtered by cell IDs."""
        cell_ids = self.get_cell_ids(plane_n)
        return self.suite2p_data[str(plane_n)][4][cell_ids, :]

    def get_cell_ids(self, plane_n: int = 0) -> np.ndarray:
        """Get valid cell IDs for a plane."""
        iscell_data = self.suite2p_data[str(plane_n)][1]
        ftraces = self.suite2p_data[str(plane_n)][2]

        # Get ROI indices where fluorescence trace has variation
        roi_indices = [i for i, trace in enumerate(ftraces) if len(set(trace)) > 1]
        cell_ids = iscell_data[:, 0].copy()
        cell_ids[roi_indices] = 0  # Mark non-cell ROIs
        return np.nonzero(cell_ids)[0]

    def get_statistics(self, plane_n: int = 0) -> np.ndarray:
        """Get statistics for specified plane."""
        return self.suite2p_data[str(plane_n)][0]

    def get_ops(self, plane_n: int = 0) -> Dict:
        """Get ops configuration for specified plane."""
        return self.suite2p_data[str(plane_n)][3]

