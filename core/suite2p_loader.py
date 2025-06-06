import warnings
from typing import Dict, Union, List
import numpy as np
from dataclasses import dataclass
from skimage import io as skio


@dataclass
class Suite2pData:
    """Container for Suite2p processed data"""
    traces: List[np.ndarray]  # Fluorescence traces
    coords: List[np.ndarray]  # cell coords, 2D
    tif_average: List[np.ndarray]  # Average projection image [512, 512]
    framerate: float  # Imaging frame rate


class Suite2pLoader:
    """
    loads suite2p pipeline output.
    """
    def __init__(self, config, fishnum, experiment_n):

        """
        Initialize Suite2p loader.

        Args:
            config: instance of config
            fishnum: your fishnum
        """

        self.config = config
        self.fishnum = fishnum

        self.suite2ppath_processed = self.config.processed_path
        self.setup_data = config.suite2p_ops
        self.number_planes = self.setup_data.get('number_planes', 1)
        self.experiment_n = experiment_n

        self._suite2p = {}
        self._optional_data = {}
        self.isloaded = False

        self._ensure_directories()

    def _ensure_directories(self):
        self.suite2ppath_processed.mkdir(parents=True, exist_ok=True)

    def _load(self, plane_n: int):
        required = ["stat.npy", "iscell.npy", "F.npy", "ops.npy", "spks.npy"]
        takeitem = [False, False, False, True, False]
        folder = (self.suite2ppath_processed / f'Fish_{self.fishnum}' /
                  f'{self.experiment_n}' / 'suite2p' / f'plane{plane_n}')

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

    def _ensure_loaded(self):
        if not self._suite2p:
            for plane_n in range(self.number_planes):
                self._load(plane_n)
            self.isloaded = True

    def _load_optional(self, plane_n: int, key: str) -> Union[np.ndarray, None]:
        """
        Lazily load optional files like 'zscore.npy', 'dff.npy', 'smoothed.npy'
        """
        if key not in self._optional_data:
            self._optional_data[key] = {}

        if plane_n not in self._optional_data[key]:
            path = self.suite2ppath_processed / f"plane{plane_n}" / f"{key}.npy"
            if not path.exists():
                warnings.warn(f"Optional file missing: {key}.npy for plane {plane_n}")
                self._optional_data[key][plane_n] = None
            else:
                self._optional_data[key][plane_n] = np.load(path, allow_pickle=True)

        return self._optional_data[key][plane_n]

    @property
    def suite2p_data(self):
        self._ensure_loaded()
        return self._suite2p

    def ftracesrois(self, plane_n=0):
        return self.suite2p_data[str(plane_n)][2]

    def s2p_spks(self, plane_n=0):
        return self.suite2p_data[str(plane_n)][4]

    def s2p_stats(self, plane_n=0):
        return self.suite2p_data[str(plane_n)][0]

    def s2p_ops(self, plane_n=0):
        return self.suite2p_data[str(plane_n)][3]

    def zscore(self, plane_n=0):
        return self._load_optional(plane_n, 'zscore_traces')

    def dff(self, plane_n=0):
        return self._load_optional(plane_n, 'dff_traces')

    def smoothed_dff(self, plane_n=0):
        return self._load_optional(plane_n, 'zscore_smoothed_traces')

    def smoothed_zscore(self, plane_n=0):
        return self._load_optional(plane_n, 'dff_smoothed_traces')

    def cellid(self, plane_n=0):
        rois = self.ftracesrois(plane_n)
        iscell = self.suite2p_data[str(plane_n)][1][:, 0].copy()
        no_var = [i for i, trace in enumerate(rois) if len(set(trace)) == 1]
        iscell[no_var] = 0
        return np.nonzero(iscell)[0]

    def ftracescells(self, plane_n=0):
        return self.ftracesrois(plane_n)[self.cellid(plane_n), :]

    def spkscells(self, plane_n=0):
        return self.s2p_spks(plane_n)[self.cellid(plane_n), :]

    def zscorecells(self, plane_n=0):
        data = self.zscore(plane_n)
        return data[self.cellid(plane_n), :] if data is not None else None

    def dffcells(self, plane_n=0):
        data = self.dff(plane_n)
        return data[self.cellid(plane_n), :] if data is not None else None

    def smoothed_dffcells(self, plane_n=0):
        data = self.smoothed_dff(plane_n)
        return data[self.cellid(plane_n), :] if data is not None else None

    def smoothed_zscorecells(self, plane_n=0):
        data = self.smoothed_zscore(plane_n)
        return data[self.cellid(plane_n), :] if data is not None else None

    def rawtif(self, plane_n=0):
        """Loads the raw merged TIFF for a given plane."""
        tif_path = (
                self.suite2ppath_processed /
                f'Fish_{self.fishnum}' / 'suite2p' / f'plane{plane_n}' /
                'merge_exp001_plane0_rec0_raw.tif'
        )
        if not tif_path.exists():
            raise FileNotFoundError(f"Raw TIFF not found: {tif_path}")
        tif_stack = skio.imread(str(tif_path), plugin='tifffile')
        return tif_stack

    def _tif_mean_path(self, plane_n=0):
        """Returns the path for the cached mean image .npy file."""
        return (
                self.suite2ppath_processed /
                f'Fish_{self.fishnum}' / 'suite2p' / f'plane{plane_n}' /
                'mean_image.npy'
        )

    def tif_mean_image(self, plane_n=0):
        """
        Returns the mean image of the raw TIFF stack.
        Loads from cache if it exists, otherwise computes and saves.
        """
        mean_path = self._tif_mean_path(plane_n)
        if mean_path.exists():
            return np.load(mean_path)

        # Load full TIFF and compute mean
        mean_img = self.rawtif(plane_n).mean(axis=0)
        np.save(mean_path, mean_img)
        return mean_img

    def get_basic_data(self, plane_n=0, transform: str = None) -> Dict[str, Union[np.ndarray, None, Suite2pData]]:
        """
        Returns a dict with 'traces', 'coordinates', and 'suite2p_data' for given plane.

        Args:
            plane_n: Plane index to query
            transform: One of [None, 'raw', 'zscore', 'dff', 'smoothed']

        Returns:
            Dict with keys: 'traces', 'coordinates', 'suite2p_data'
        """
        coords = np.array([s['med'] for s in self.s2p_stats(plane_n)])
        cell_coords = coords[self.cellid(plane_n)]

        trace_map = {
            None: self.ftracescells,
            'raw': self.ftracescells,
            'zscore': self.zscorecells,
            'dff': self.dffcells,
            'smoothed_dff': self.smoothed_dffcells,
            'smoothed_zscore': self.smoothed_zscorecells
        }

        if transform not in trace_map:
            raise ValueError(f"Unknown transform type: {transform}")

        suite2p_data = Suite2pData(
            traces=trace_map[transform](plane_n),
            coords=cell_coords,
            tif_average=self.tif_mean_image(plane_n),
            framerate=self.config.suite2p_ops.get('framerate', 1.0)
        )

        return suite2p_data
