import os

import numpy as np
from scipy import stats
from scipy.ndimage import uniform_filter1d
from sklearn import preprocessing
from pathlib import Path
from typing import Dict, Optional, Union


class Suite2pTraces:
    """
    Runs processing of suite2p outputted fluorescent traces.
    """
    TRACE_TYPES = ['dff', 'zscore', 'dff_smooth', 'zscore_smooth']

    def __init__(self, config, fishnum):

        """
        Initialize Suite2p traces processing.

        Args:
            config: instance of config
            fishnum: your fishnum
        """

        self.config = config
        self.fishnum = fishnum
        self.suite2p_ops = config.suite2p_ops
        self.nplanes = self.suite2p_ops['nplanes']
        self.framerate = self.suite2p_ops['framerate']
        self.experiments = list(config.get_fish_config(fishnum)['experiments'][self.config.date].keys())

        self._suite2p = {}

        self.processed_data = {t: {} for t in self.TRACE_TYPES}
        self.load_status = {plane: False for plane in range(self.nplanes)}
        self.process_status = {plane: False for plane in range(self.nplanes)}

    def get_trace_path(self, experiment_number: str, plane: int) -> Path:
        """Returns the path to the raw F.npy trace file."""
        return (Path(self.config.processed_path) / f'Fish_{self.fishnum}' / experiment_number / 'suite2p' /
                f'plane{plane}' / 'F.npy')

    def get_save_path(self, experiment_number: str, plane: int, trace_type: str) -> Path:
        """Returns the path where the processed trace should be saved."""
        plane_dir = (Path(self.config.processed_path) / f'Fish_{self.fishnum}' / experiment_number / 'suite2p'
                     / f'plane{plane}')
        plane_dir.mkdir(parents=True, exist_ok=True)
        return plane_dir / f'{trace_type}_traces.npy'

    def cellid(self, plane_n, experiment_number):
        folder = (self.config.processed_path / f'Fish_{self.fishnum}' /
                  f'{experiment_number}' / 'suite2p' / f'plane{plane_n}')

        iscell = np.load(os.path.join(folder, 'iscell.npy'), allow_pickle=True)
        return iscell

    def _load(self, experiment_n: int, plane_n: int):
        required = ["stat.npy", "iscell.npy", "F.npy", "ops.npy", "spks.npy"]
        takeitem = [False, False, False, True, False]
        folder = (self.config.processed_path / f'Fish_{self.fishnum}' /
                  f'{experiment_n}' / 'suite2p' / f'plane{plane_n}')

        data = []
        for i, fname in enumerate(required):
            fpath = folder / fname
            if not fpath.exists():
                raise FileNotFoundError(f"Missing required file: {fpath}")
            loaded = np.load(fpath, allow_pickle=True)
            if takeitem[i]:
                loaded = loaded.item()
            data.append(loaded)

        self._suite2p[str(plane_n)] = data

    def load_raw_traces(self, experiment_number: str) -> Dict[int, np.ndarray]:
        """Loads raw fluorescence traces (F.npy) for each plane."""
        traces = {}
        for plane in range(self.nplanes):
            path = self.get_trace_path(experiment_number, plane)
            if not path.exists():
                raise FileNotFoundError(f"Trace file not found: {path}")
            traces[plane] = np.load(path, allow_pickle=True)
        return traces

    def process_all(self):
        for exp in self.experiments:
            exp = str(int(exp))
            print(f"\nProcessing experiment {exp} for Fish {self.fishnum}")
            raw_traces = self.load_raw_traces(exp)
            for plane in range(self.nplanes):
                self.process_plane(raw_traces[plane], exp, plane)

    def process_plane(self, traces: np.ndarray, experiment_number: str, plane: int):
        print(f"Processing plane {plane} for experiment {experiment_number}...")

        if self.process_status[plane]:
            return

        dff = self._process_traces(traces, dff=True)
        zscore = self._process_traces(traces, zscore=True)
        dff_smooth = self._process_traces(traces, dff=True, smooth=True)
        zscore_smooth = self._process_traces(traces, zscore=True, smooth=True)

        self.processed_data['dff'][plane] = dff[self.cellid(plane, experiment_number), :]
        self.processed_data['zscore'][plane] = zscore[self.cellid(plane, experiment_number), :]
        self.processed_data['dff_smooth'][plane] = dff_smooth[self.cellid(plane, experiment_number), :]
        self.processed_data['zscore_smooth'][plane] = zscore_smooth[self.cellid(plane, experiment_number), :]

        self.save_traces(experiment_number, plane)

        self.process_status[plane] = True
        self.load_status[plane] = True

    def save_traces(self, experiment_number: str, plane: int):
        print(f"Saving traces for plane {plane} in experiment {experiment_number}")
        for trace_type in self.TRACE_TYPES:
            data = self.processed_data[trace_type].get(plane)
            if data is not None:
                save_path = self.get_save_path(experiment_number, plane, trace_type)
                np.save(save_path, data)

    def _process_traces(self, traces: np.ndarray, dff=False, zscore=False, smooth=False, normalize=False) -> np.ndarray:
        traces = traces.astype(np.float32)
        processed = np.empty_like(traces)

        for i in range(traces.shape[0]):
            trace = traces[i]
            trace = self._detrend(trace)

            if normalize:
                trace = preprocessing.normalize(trace.reshape(1, -1)).flatten()
            if dff:
                trace = self._calculate_dff(trace)
            if zscore:
                trace = stats.zscore(trace, nan_policy='raise')
            if smooth:
                trace = self._smooth_trace(trace)

            processed[i] = trace

        return processed

    @staticmethod
    def _detrend(trace: np.ndarray, window_size: int = 500) -> np.ndarray:
        baseline = np.mean(trace[:window_size])
        moving_avg = np.convolve(trace, np.ones(window_size) / window_size, mode='valid')
        detrended = np.full_like(trace, baseline)
        detrended[window_size - 1:] = trace[window_size - 1:] - moving_avg
        detrended[:window_size] = trace[:window_size] - baseline
        return detrended

    @staticmethod
    def _calculate_dff(trace: np.ndarray) -> np.ndarray:
        f0 = np.mean(trace[trace <= np.percentile(trace, 5)])
        f0 = max(f0, 1.0)
        return (trace - f0) / f0

    def _smooth_trace(self, trace: np.ndarray) -> np.ndarray:
        window = int(self.framerate * 5)
        return uniform_filter1d(trace, size=window)

    def get_processed_traces(self, trace_type: str,
                             plane: Optional[int] = None) -> Union[Dict[int, np.ndarray], np.ndarray]:
        if trace_type not in self.TRACE_TYPES:
            raise ValueError(f"Invalid trace type: {trace_type}")
        if plane is not None:
            return self.processed_data[trace_type].get(plane)
        else:
            return {p: self.processed_data[trace_type].get(p) for p in range(self.nplanes)}
