import os
import numpy as np
from scipy import stats
from scipy.ndimage import uniform_filter1d
from sklearn import preprocessing
from typing import Dict, Optional, Union
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Suite2pTraces:
    """Processes and manages Suite2p fluorescence traces for multi-plane imaging data."""

    TRACE_TYPES = {
        'dff': 'dff_traces.npy',
        'zscore': 'zscore_traces.npy',
        'dff_smooth': 'dff_smooth_traces.npy',
        'zscore_smooth': 'zscore_smooth_traces.npy'
    }

    def __init__(
            self, traces, framerate
    ):
        """
        Initialize the Suite2p traces processor.

        Args:
            animalnum: Animal identifier number
            exptype: Experiment type identifier
            expnum: Experiment number
            suite2p: Suite2p data object
            number_planes: Number of imaging planes
            framerate: Imaging frame rate (Hz)
            force_processing: If True, reprocess even if cached files exist
        """

        # Initialize data structures
        self.ftracescells = traces
        self.processed_data = {t: {} for t in self.TRACE_TYPES}
        self.load_status = {plane: False for plane in range(number_planes)}
        self.process_status = {plane: False for plane in range(number_planes)}

        # Set up paths
        self.base_path = processedfilepath(self.animalnum, self.exptype, self.expnum)
        Path(self.base_path).mkdir(parents=True, exist_ok=True)

    def get_trace_path(self, plane: int, trace_type: str) -> Path:
        """Get path for a specific trace type and plane."""
        if trace_type not in self.TRACE_TYPES:
            raise ValueError(f"Invalid trace type. Must be one of: {list(self.TRACE_TYPES.keys())}")

        plane_dir = Path(self.base_path) / 'suite2p' / f'plane{plane}'
        plane_dir.mkdir(parents=True, exist_ok=True)
        return plane_dir / self.TRACE_TYPES[trace_type]

    def save_traces(self, plane: int) -> None:
        """Save processed traces for a specific plane."""
        logger.info(f"Saving processed traces for plane {plane}")
        try:
            for trace_type in self.TRACE_TYPES:
                if self.processed_data[trace_type].get(plane) is not None:
                    np.save(self.get_trace_path(plane, trace_type),
                            self.processed_data[trace_type][plane],
                            allow_pickle=True)
        except Exception as e:
            logger.error(f"Error saving traces for plane {plane}: {e}")
            raise

    def process_all_planes(self) -> Dict[int, np.ndarray]:
        """Process traces for all planes."""
        logger.info("Processing traces for all planes...")
        results = {}
        for plane in range(self.number_planes):
            results[plane] = self.process_plane(plane)
        return results

    def process_plane(self, plane: int) -> np.ndarray:
        """Process traces for a specific plane."""
        logger.info(f"Processing traces for plane {plane}...")

        # Check if processing is needed
        needs_processing = (
                self.force_processing or
                not all(self.get_trace_path(plane, t).exists() for t in self.TRACE_TYPES)
        )

        if needs_processing or not self.process_status[plane]:
            try:
                # Process all trace types
                self.processed_data['dff'][plane] = self._process_traces(
                    plane, dff=True)
                self.processed_data['zscore'][plane] = self._process_traces(
                    plane, zscore=True)
                self.processed_data['dff_smooth'][plane] = self._process_traces(
                    plane, dff=True, smooth=True)
                self.processed_data['zscore_smooth'][plane] = self._process_traces(
                    plane, zscore=True, smooth=True)

                self.save_traces(plane)
                self.process_status[plane] = True
                self.load_status[plane] = True
            except Exception as e:
                logger.error(f"Error processing plane {plane}: {e}")
                raise

        elif not self.load_status[plane]:
            self.load_traces(plane)

        return self.processed_data['dff'][plane]

    def _process_traces(
            self,
            plane: int,
            dff: bool = False,
            smooth: bool = False,
            normalize: bool = False,
            zscore: bool = False
    ) -> np.ndarray:
        """Process fluorescence traces with specified transformations."""
        logger.debug(f"Processing traces for plane {plane} with dff={dff}, "
                     f"smooth={smooth}, normalize={normalize}, zscore={zscore}")

        traces = self.ftracescells[plane].astype(np.float32)
        processed = np.empty_like(traces)

        # TODO maybe implement processors?

        for cell in range(traces.shape[0]):
            trace = traces[cell, :]

            # Apply processing steps in optimized order
            trace = self._detrend(trace, window_size=500)

            if normalize:
                trace = preprocessing.normalize(trace.reshape(1, -1)).flatten()

            if dff:
                trace = self._calculate_dff(trace)

            if zscore:
                trace = stats.zscore(trace, nan_policy='raise')

            if smooth:
                trace = self._smooth_trace(trace)

            processed[cell, :] = trace

        return processed

    @staticmethod
    def _detrend(data: np.ndarray, window_size: int) -> np.ndarray:
        """Remove slow trends from the data."""
        edge_mean = np.mean(data[:window_size])
        detrended = np.full_like(data, edge_mean, dtype=np.float32)
        moving_avg = np.convolve(data, np.ones(window_size) / window_size, mode='valid')
        detrended[window_size - 1:] = data[window_size - 1:] - moving_avg
        detrended[:window_size] = data[:window_size] - edge_mean
        return detrended

    @staticmethod
    def _calculate_dff(trace: np.ndarray) -> np.ndarray:
        """Calculate ΔF/F with baseline as lowest 5% of values."""
        f0 = np.mean(trace[trace <= np.percentile(trace, 5)])
        f0 = max(f0, 1.0)  # Ensure we don't divide by very small numbers
        return (trace - f0) / f0

    def _smooth_trace(self, trace: np.ndarray) -> np.ndarray:
        """Apply temporal smoothing to the trace."""
        window = int(self.framerate * 5)  # 5-second window
        return uniform_filter1d(trace, size=window)

    def get_processed_traces(
            self,
            plane: Optional[int] = None,
            trace_type: str = 'dff'
    ) -> Union[np.ndarray, Dict[int, np.ndarray]]:
        """
        Get processed traces for a specific plane or all planes.

        Args:
            plane: Plane index or None for all planes
            trace_type: Type of trace to return ('dff', 'zscore', etc.)

        Returns:
            Requested traces for specified plane(s)
        """
        if trace_type not in self.TRACE_TYPES:
            raise ValueError(f"Invalid trace type. Must be one of: {list(self.TRACE_TYPES.keys())}")

        if plane is not None:
            if not self.process_status.get(plane, False):
                self.process_plane(plane)
            return self.processed_data[trace_type][plane]
        else:
            return {p: self.get_processed_traces(p, trace_type)
                    for p in range(self.number_planes)}

    def load_traces(self, plane: int) -> None:
        """Load processed traces for a specific plane."""
        logger.info(f"Loading processed traces for plane {plane}")
        try:
            for trace_type in self.TRACE_TYPES:
                trace_path = self.get_trace_path(plane, trace_type)
                if trace_path.exists():
                    self.processed_data[trace_type][plane] = np.load(trace_path, allow_pickle=True)

            self.load_status[plane] = True
            self.process_status[plane] = True
        except Exception as e:
            logger.error(f"Error loading traces for plane {plane}: {e}")
            raise
