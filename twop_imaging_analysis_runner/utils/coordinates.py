# coordinate transformations
from typing import Optional

import numpy as np
from twop_imaging_analysis_runner.config.base_config import processed_path

# Constants for imaging parameters
MICRONS_PER_PIXEL_ZSTACK_XY = 0.273  # Zoom 2x, 20x objective
MICRONS_PER_PIXEL_ZSTACK_Z = 2.0  # 2um steps in standard zstack acquisition
ZSTACK_TO_IMAGING_RESOLUTION_DIFF = 4  # Imaging is with 512x512, zstack 2048x2048
MICRONS_PER_PIXEL_STANDARD = 0.8303  # From the standard brain in the MapZeBrain atlas

# Derived imaging parameters
microns_per_pixel_imagingXY = (
        MICRONS_PER_PIXEL_ZSTACK_XY *
        ZSTACK_TO_IMAGING_RESOLUTION_DIFF
)


def get_atlas_coordinates(self) -> np.ndarray:
    """
    Get xyz coordinates transformed to standard brain atlas format.

    Returns:
        Array of coordinates in microns (x, y, z)
    """
    z_file = self.suite2ppath_processed / f'Fish_{self.animalnum}\\all\\plane_zs.txt'
    if not z_file.exists():
        raise FileNotFoundError(f"Z-plane file not found: {z_file}")

    with open(z_file, 'r') as f:
        z_data = f.read().split()
    original_zs = [int(z) for z in z_data]

    all_coords = []
    for plane_n, z in enumerate(original_zs):
        stats = self.get_statistics(plane_n)
        cell_ids = self.get_cell_ids(plane_n)

        # Get median positions of cells
        xy_coords = np.array([stats[roi_n]['med'] for roi_n in cell_ids])
        z_coords = np.full((len(cell_ids), 1), z)
        xyz_coords = np.hstack((xy_coords, z_coords))

        all_coords.append(xyz_coords)

    # Combine coordinates from all planes
    original_coords = np.vstack(all_coords)

    # Apply coordinate transformations
    # Rotation matrix for z-axis rotation
    R_z = np.array([
        [0, 1, 0],
        [-1, 0, 0],
        [0, 0, 1]
    ])

    # Apply rotation and flip y-axis
    transformed = np.dot(original_coords, R_z.T)
    transformed[:, 1] = 512 - transformed[:, 1]  # Flip y-axis

    # Convert to microns
    transformed[:, :2] *= self.MICRONS_PER_PIXEL_ZSTACK_XY
    transformed[:, 2] *= self.MICRONS_PER_PIXEL_ZSTACK_Z

    return transformed.astype(np.int32)


def get_transformed_coordinates(fishnum) -> Optional[np.ndarray]:
    """Load pre-transformed coordinates if they exist."""
    coord_file = processed_path / f'{fishnum}' / 'transformed_coords.npy'
    if coord_file.exists():
        return np.load(coord_file)
    print(f"No transformed coordinates found for animal {fishnum}")
    return None
