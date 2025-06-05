import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy import stats
from matplotlib.collections import PatchCollection
from matplotlib.patches import Circle
from dataclasses import dataclass
from typing import List, Optional
from pathlib import Path


@dataclass
class Suite2pData:
    """Container for Suite2p processed data"""
    stats: List[np.ndarray]  # ROI stats per plane
    F: List[np.ndarray]  # Fluorescence traces
    spks: List[np.ndarray]  # Deconvolved spikes
    tif_average: List[np.ndarray]  # Average projection images
    ops: List[dict]  # Ops dictionaries
    framerate: float  # Imaging frame rate


class Suite2pVisualiser:
    """Visualization toolkit for Suite2p processed data with built-in plotting."""

    def __init__(self, data: Suite2pData, fishnum: int, output_dir: Path = Path(".")):
        """
        Args:
            data: Suite2pData object containing processed data
            fishnum: Animal identifier
            output_dir: Base directory for saving plots
        """
        self.data = data
        self.fishnum = fishnum
        self.output_dir = output_dir / f"Fish_{fishnum}"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_highly_active(
            self,
            plane: int,
            threshold: float = 2.0,
            n_rois: int = 20,
            spacing: float = 3.0,
            time_window: Optional[tuple] = None,
            save_plot: bool = False,
            plot_name: str = "highly_active_traces"
    ) -> plt.Figure:
        """
        Create and display a plot of vertically offset z-scored traces.

        Args:
            plane: Imaging plane index
            threshold: Z-score threshold for activity
            n_rois: Number of top ROIs to display
            spacing: Vertical spacing between traces
            time_window: (start, end) in seconds for zoomed view
            save_plot: Whether to save the figure
            plot_name: Base name for saved file

        Returns:
            matplotlib.figure.Figure
        """
        fig = plt.figure(figsize=(12, 6))
        ax = fig.add_subplot(111)

        # Calculate and sort by activity
        zscores = stats.zscore(self.data.F[plane], axis=1)
        activity = np.mean(zscores > threshold, axis=1)
        top_rois = np.argsort(activity)[-n_rois:][::-1]

        # Plot each trace with offset
        for i, roi in enumerate(top_rois):
            y_offset = i * spacing
            ax.plot(zscores[roi] + y_offset, lw=1,
                    label=f'ROI {roi} (act={activity[roi]:.2f})')

        # Configure plot
        ax.set_yticks(np.arange(n_rois) * spacing)
        ax.set_yticklabels(top_rois)
        ax.set_title(f'Fish {self.fishnum} - Top {n_rois} Active ROIs (Plane {plane})')
        ax.set_xlabel('Frames')
        ax.set_ylabel('ROI ID (Z-score + offset)')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        if time_window:
            ax.set_xlim(time_window[0] * self.data.framerate,
                        time_window[1] * self.data.framerate)

        if save_plot:
            save_path = self.output_dir / f"{plot_name}_plane{plane}.png"
            fig.savefig(save_path, bbox_inches='tight', dpi=300)
            plt.close(fig)
            return None

        return fig

    def plot_location(
            self,
            plane: int,
            cell_ids: List[int] = None,
            radius_um: float = 10,
            alpha: float = 0.7,
            save_plot: bool = False,
            plot_name: str = "cell_locations"
    ) -> plt.Figure:
        """
        Create and display cell locations overlaid on average projection.

        Args:
            plane: Imaging plane index
            cell_ids: List of cell indices to highlight
            radius_um: Circle radius in microns
            alpha: Transparency of cell markers
            save_plot: Whether to save the figure
            plot_name: Base name for saved file

        Returns:
            matplotlib.figure.Figure
        """
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111)

        # Display background
        avg_img = self.data.tif_average[plane]
        ax.imshow(avg_img, cmap='gray',
                  vmax=np.percentile(avg_img, 99))

        # Plot all ROIs
        plane_stats = self.data.stats[plane]
        all_pos = np.array([s['med'] for s in plane_stats])
        ax.scatter(all_pos[:, 1], all_pos[:, 0],
                   s=5, c='cyan', alpha=0.3, label='All ROIs')

        # Highlight specified cells
        if cell_ids:
            microns_per_pixel = self.data.ops[plane]['microns_per_pixel']
            radius_px = radius_um / microns_per_pixel
            circles = [Circle((plane_stats[i]['med'][1], plane_stats[i]['med'][0]), radius_px)
                       for i in cell_ids]
            pc = PatchCollection(circles, facecolors='red',
                                 edgecolors='white', alpha=alpha)
            ax.add_collection(pc)

        ax.set_title(f'Fish {self.fishnum} - Cell Locations (Plane {plane})')
        ax.set_xlabel('X (pixels)')
        ax.set_ylabel('Y (pixels)')
        ax.legend()
        ax.invert_yaxis()

        if save_plot:
            save_path = self.output_dir / f"{plot_name}_plane{plane}.png"
            fig.savefig(save_path, bbox_inches='tight', dpi=300)
            plt.close(fig)
            return None

        return fig

    def plot_heatmap(
            self,
            plane: int,
            cluster: bool = True,
            metric: str = 'zscore',
            save_plot: bool = False,
            plot_name: str = "activity_heatmap"
    ) -> plt.Figure:
        """
        Create and display clustered activity heatmap.

        Args:
            plane: Imaging plane index
            cluster: Whether to cluster ROIs by activity
            metric: 'zscore' or 'dff' for normalization
            save_plot: Whether to save the figure
            plot_name: Base name for saved file

        Returns:
            matplotlib.figure.Figure
        """
        fig = plt.figure(figsize=(12, 6))
        ax = fig.add_subplot(111)

        traces = self.data.F[plane]

        # Normalize traces
        if metric == 'zscore':
            data = stats.zscore(traces, axis=1)
        else:  # dFF
            data = (traces - np.median(traces, axis=1, keepdims=True)) / \
                   np.median(traces, axis=1, keepdims=True)

        # Cluster if requested
        if cluster:
            Z = linkage(data, method='ward')
            order = leaves_list(Z)
            data = data[order]

        # Create heatmap
        sns.heatmap(
            data,
            ax=ax,
            cmap='viridis',
            cbar_kws={'label': f'{metric} activity'},
            xticklabels=False,
            yticklabels=10
        )

        ax.set_title(f'Fish {self.fishnum} - ROI Activity Heatmap (Plane {plane})')
        ax.set_ylabel('ROI Index')

        if save_plot:
            save_path = self.output_dir / f"{plot_name}_plane{plane}.png"
            fig.savefig(save_path, bbox_inches='tight', dpi=300)
            plt.close(fig)
            return None

        return fig
