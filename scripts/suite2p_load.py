import os
import shutil

import numpy as np

from suite2p_run.lisa_ops import lisa_ops
from suite2p_run.suite2p_run import Suite2pRun
from Analysis_pipeline.generalinfo import baseprocessedpath, lisabasedatapath


# In Lisa Bauer's standard imaging experiments. Do change.
MICRONS_PER_PIXEL_ZSTACK_XY = 0.273  # Zoom 2x, 20x objective
MICRONS_PER_PIXEL_ZSTACK_Z = 2  # 2um steps in standard zstack acquisition
ZSTACK_TO_IMAGING_RESOLUTION_DIFF = 4  # Imaging is with 512x512, zstack 2048x2048
MICRONS_PER_PIXEL_STANDARD = 0.8303  # From the standard brain in the MapZeBrain atlas


class Suite2p_Getter:

    def __init__(self, suite2ppath_processed, suite2ppath_raw, imagingfiles, exptype, expnum, animalnum, number_planes,
                 experiments, framerate, date, experiment_lengths):
        print('running suite2p wrapper')
        self.suite2ppath_processed = suite2ppath_processed
        self.suite2ppath_raw = suite2ppath_raw
        self.isloaded = False
        self.isprocessed = False
        self._suite2p_cache = None

        self.imagingfiles = imagingfiles
        self.exptype = exptype
        self.animalnum = animalnum
        self.expnum = expnum
        self.number_planes = number_planes
        self.number_imagingfiles = len(imagingfiles)
        self.experiments = experiments
        self.experiment_lengths = experiment_lengths
        self.framerate = framerate
        self.date = date

        self.microns_per_pixel_zstackXY = MICRONS_PER_PIXEL_ZSTACK_XY
        self.microns_per_pixel_imagingXY = self.microns_per_pixel_zstackXY * ZSTACK_TO_IMAGING_RESOLUTION_DIFF
        self.microns_per_pixelzstackZ = MICRONS_PER_PIXEL_ZSTACK_Z
        self.microns_per_pixel_standard = MICRONS_PER_PIXEL_STANDARD

    def _load(self, plane_n):
        toload = ["stat.npy", "iscell.npy", "F.npy", "ops.npy", "spks.npy"]
        takeitem = [False, False, False, True, False]
        # ops.npy loads as something that needs .item() called on it
        filepaths = [os.path.join(self.suite2ppath_processed, f'plane{plane_n}\\{s}') for s in toload]
        assert all([os.path.exists(p) for p in filepaths]), "Paths don't exist"
        items = [np.load(filepath, allow_pickle=True) for filepath in filepaths]
        loaded = [item.item() if takeitem[i] else item for i, item in enumerate(items)]
        return loaded

    @property
    def _suite2p(self):
        if self._suite2p_cache is None:
            self._suite2p_cache = self._perform_suite2p_processing()
        return self._suite2p_cache

    def _perform_suite2p_processing(self):
        # Try loading from a specific hash folder the suite2p

        s = Suite2pRun(
            self.suite2ppath_raw,
            self.suite2ppath_processed,
            self.animalnum,
            self.framerate,
            self.imagingfiles,
            self.number_planes,
            self.experiments,
            self.experiment_lengths
        )

        self.run_hash = s.define_run_hash(
            your_ops=lisa_ops(),
            motion_correction=True,
            filtering=True,
            downsampling=True
        )

        try:
            # Try to load the data for the existing run
            print(f"Run already exists for hash: {self.run_hash}. Attempting to load data...")
            self.loadedsuite2p = {}
            for plane_n in range(self.number_planes):
                key_name = str(plane_n)
                self.loadedsuite2p[key_name] = self._load(plane_n)
            self.isloaded = True
        except AssertionError as e:
            # Handle the case where the file does not exist or fails to load
            print(f"Error loading data for hash: {self.run_hash}. Reason: {e}. Starting Suite2p extraction...")
            s.run_extraction(self.run_hash)
            self.isloaded = False  # Mark as not loaded since extraction is being performed

            try:
                # Try to load the data again for the existing run
                print(f"Run already exists for hash: {self.run_hash}. Attempting to load data...")
                self.loadedsuite2p = {}
                for plane_n in range(self.number_planes):
                    key_name = str(plane_n)
                    self.loadedsuite2p[key_name] = self._load(plane_n)
                self.isloaded = True
            except AssertionError as e:
                print('problem loading suite2p even though it should exist')

        return self.loadedsuite2p

    def ftracesrois(self, plane_n=0):
        return self._suite2p[str(plane_n)][2]

    def s2p_spks(self, plane_n=0):
        return self._suite2p[str(plane_n)][4]

    def s2p_stats(self, plane_n=0):
        return self._suite2p[str(plane_n)][0]

    def s2p_ops(self, plane_n=0):
        return self._suite2p[str(plane_n)][3]

    def cellid(self, plane_n=0):
        roi_indices = [cell for cell, n in enumerate(self._suite2p[str(plane_n)][2]) if len(set(n)) == 1]
        cellids = self._suite2p[str(plane_n)][1][:, 0]
        cellids[roi_indices] = 0
        return np.nonzero(cellids)[0]

    def xyz_coords_atlas(self):
        """ Get the x y z coordinates in the right standard brain atlas format. """

        # TODO finish the implementation

        # Get the right
        suite2p_file = os.path.join(self.suite2ppath_processed, f'Fish_{self.animalnum}\\1')
        z_plane_files = os.path.join(self.suite2ppath_processed, f'Fish_{self.animalnum}\\all\\plane_zs.txt')

        # Load the z data from text file
        f = open(file_path_z[animal_n], 'r')
        z_data = f.read().split()
        f.close()
        original_zs = [int(z_data[plane_n]) for plane_n in range(len(z_data))]

        original_xyzs = []

        # load all the xy data from suite2p
        for plane_n in range(0, len(original_zs)):
            stat_data = np.load(
                f'J:\\_Projects\\Lisa\\processed\\Fish_{animal}\\1\\suite2p\\plane{plane_n}\\stat.npy',
                allow_pickle=True)
            iscell = np.load(
                f'J:\\_Projects\\Lisa\\processed\\Fish_{animal}\\1\\suite2p\\plane{plane_n}\\iscell.npy',
                allow_pickle=True)[:, 0]

            cellids = np.nonzero(iscell)[0]

            original_xys = np.asarray([stat_data[roi_n]['med'] for roi_n in range(len(stat_data))])[cellids]
            n_cells = original_xys.shape[0]

            plane_zs = np.asarray([original_zs[plane_n]] * n_cells)

            original_plane_xyzs = np.vstack((original_xys.T, plane_zs)).T

            original_xyzs.append(original_plane_xyzs)

        original_coords = np.concatenate(original_xyzs, axis=0)

        # transform coords into right orientation
        R_z = np.array([
            [0, 1, 0],
            [-1, 0, 0],
            [0, 0, 1]
        ])
        flipped_coords = np.dot(original_coords, R_z.T)

        # I maybe need to do this?
        # # Determine the maximum y-coordinate, assuming 2nd column is y
        # y_max = np.max(coordinates[:, 1])

        # # Mirror the y-coordinates about the horizontal midline
        # coordinates[:, 1] = y_max - coordinates[:, 1]

        adjusted_coords = np.copy(flipped_coords)
        adjusted_coords[:, 1] = 512 + flipped_coords[:, 1]  # is this correct?

        # This is still in pixel value, it should change to micron (only to x and y)
        adjusted_coords = adjusted_coords.astype(float)
        adjusted_coords[:, 0] *= microns_per_pixel_zstackXY
        adjusted_coords[:, 1] *= microns_per_pixel_zstackXY
        adjusted_coords[:, 2] *= microns_per_pixelzstackZ
        adjusted_coords = adjusted_coords.astype(int)

        return adjusted_coords

    def transformed_coords(self):
        transformed_coords_file = os.path.join(lisabasedatapath, f'{self.animalnum}',
                                               'transformed_coords.npy')
        if os.path.exists(transformed_coords_file):
            np.load(transformed_coords_file)
        else:
            print('you dont have the transformed coords for this fish')
        pass

    def ftracescells(self, plane_n=0):
        return self.ftracesrois(plane_n=plane_n)[self.cellid(plane_n=plane_n), :]

    def spkscells(self, plane_n=0):
        return self.s2p_spks(plane_n=plane_n)[self.cellid(plane_n=plane_n), :]

    @_suite2p.setter
    def _suite2p(self, value):
        self.__suite2p = value
