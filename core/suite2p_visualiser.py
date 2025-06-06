from datetime import date

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy import stats
from typing import Optional
from pathlib import Path

from config.base_config import plot_path


class Suite2pVisualiser:
    """Visualization toolkit for Suite2p processed data with built-in plotting."""

    def __init__(self, data, config, fishnum: int, experiment_n, output_dir: str = plot_path):
        """
        Args:
            data: Suite2pData object containing processed data
            fishnum: Animal identifier
            output_dir: Base directory for saving plots
        """
        self.data = data
        self.fishnum = fishnum
        self.experiment_n = experiment_n
        self.output_dir = (Path(output_dir) /
                           str(date.today().strftime("%Y%m%d")) / f"Fish_{fishnum}" / f'{experiment_n}')
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.suite2p_ops = config.suite2p_ops
        self.framerate = self.suite2p_ops['framerate']

    def plot_highly_active(
            self,
            plane: int = 0,
            threshold: float = 2.0,
            n_cells: int = 20,
            spacing: float = 10.0,
            time_window: Optional[tuple] = None,
            save_plot: bool = False,
            plot_name: str = "highly_active_traces"
    ) -> plt.Figure:
        """
        Create and display a plot of vertically offset z-scored traces.

        Args:
            plane: Imaging plane index
            threshold: Z-score threshold for activity
            n_cells: Number of top cells to display
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
        zscores = stats.zscore(self.data.traces, axis=1)
        activity = np.mean(zscores > threshold, axis=1)
        top_cells = np.argsort(activity)[-n_cells:][::-1]

        # Create time vector in seconds
        n_frames = zscores.shape[1]
        time = np.arange(n_frames) / self.framerate

        # Plot each trace with offset
        for i, cell in enumerate(top_cells):
            y_offset = i * spacing
            ax.plot(time, zscores[cell] + y_offset, lw=1,
                    label=f'Cell {cell}')

        # Configure plot
        ax.set_yticks(np.arange(n_cells) * spacing)
        ax.set_yticklabels(top_cells)
        ax.set_title(f'Fish {self.fishnum} - Top {n_cells} Active Cells (Plane {plane})')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Cell ID (Z-score + offset)')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        if time_window:
            ax.set_xlim(time_window)

        if save_plot:
            save_path = (self.output_dir / f"{plot_name}_plane{plane}.png")
            fig.savefig(save_path, bbox_inches='tight', dpi=300)
            plt.close(fig)
            return None

        return fig

    def plot_location(
            self,
            plane: int = 0,
            save_plot: bool = False,
            plot_name: str = "cell_locations"
    ) -> plt.Figure:
        """
        Create and display cell locations overlaid on average projection.

        Args:
            plane: Imaging plane index
            save_plot: Whether to save the figure
            plot_name: Base name for saved file

        Returns:
            matplotlib.figure.Figure
        """
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111)

        # Display background
        avg_img = self.data.tif_average
        ax.imshow(avg_img, cmap='gray',
                  vmax=np.percentile(avg_img, 99))

        # Plot all cells
        plane_coords = self.data.coords
        ax.scatter(plane_coords[:, 1], plane_coords[:, 0],
                   s=5, c='cyan', alpha=0.3)

        ax.set_title(f'Fish {self.fishnum} - Cell Locations (Plane {plane})')
        ax.set_xlabel('X (pixels)')
        ax.set_ylabel('Y (pixels)')
        ax.invert_yaxis()

        if save_plot:
            save_path = (self.output_dir / f"{plot_name}_plane{plane}.png")
            fig.savefig(save_path, bbox_inches='tight', dpi=300)
            plt.close(fig)
            return None

        return fig

    def plot_heatmap(
            self,
            plane: int = 0,
            cluster: bool = True,
            metric: str = 'zscore',
            save_plot: bool = False,
            plot_name: str = "activity_heatmap"
    ) -> plt.Figure:
        """
        Create and display clustered activity heatmap.

        Args:
            plane: Imaging plane index
            cluster: Whether to cluster Cells by activity
            metric: 'zscore' or 'dff' for normalization
            save_plot: Whether to save the figure
            plot_name: Base name for saved file

        Returns:
            matplotlib.figure.Figure
        """
        fig = plt.figure(figsize=(12, 6))
        ax = fig.add_subplot(111)

        traces = self.data.traces

        # Normalize traces
        if metric == 'zscore':
            data = stats.zscore(traces, axis=1)
        else:  # dFF
            median = np.median(traces, axis=1, keepdims=True)
            median[median == 0] = 1e-3
            data = (traces - median) / median

        # Cluster if requested
        if cluster:
            Z = linkage(data, method='ward')
            order = leaves_list(Z)
            data = data[order]

        # Clip extremes for better visual contrast
        vmin = np.percentile(data, 1)
        vmax = np.percentile(data, 99)

        # Create heatmap
        sns.heatmap(
            data,
            ax=ax,
            cmap='gray',
            cbar_kws={'label': f'{metric} activity'},
            xticklabels=False,
            yticklabels=False,  # We'll set custom ticks below
            vmin=vmin,
            vmax=vmax
        )

        # Custom y-ticks every 100th cell
        n_cells = data.shape[0]
        tick_locs = np.arange(0, n_cells, 100)
        ax.set_yticks(tick_locs)
        ax.set_yticklabels(tick_locs)

        ax.set_title(f'Fish {self.fishnum} - Cell Activity Heatmap (Plane {plane})', fontsize=12)
        ax.set_ylabel('Cell Index')

        if save_plot:
            save_path = self.output_dir / f"{plot_name}_plane{plane}.png"
            fig.savefig(save_path, bbox_inches='tight', dpi=300)
            plt.close(fig)
            return None

        return fig
