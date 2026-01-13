"""
Speed Profile Plots
===================

Generates professional speed profile visualizations with standard deviation bands,
reference lines, and optional acceleration overlays.

Author: Stride Labs
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize, LinearSegmentedColormap
from typing import List, Tuple, Optional, Dict, Any, Union
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class SpeedPlotStyle:
    """Style configuration for speed profile plots."""
    # Figure settings
    figure_size: Tuple[float, float] = (14, 8)
    dpi: int = 150
    background_color: str = "#1a1a2e"
    plot_face_color: str = "#16213e"

    # Speed line settings
    speed_color: str = "#4cc9f0"
    speed_linewidth: float = 2.5
    speed_alpha: float = 1.0

    # Standard deviation band
    std_color: str = "#4cc9f0"
    std_alpha: float = 0.25

    # Reference lines
    belt_speed_color: str = "#f72585"
    belt_speed_style: str = "--"
    belt_speed_width: float = 2.0

    avg_speed_color: str = "#7209b7"
    avg_speed_style: str = ":"
    avg_speed_width: float = 2.0

    # Acceleration overlay
    accel_positive_color: str = "#06d6a0"  # Green for acceleration
    accel_negative_color: str = "#ef476f"  # Red for deceleration
    accel_alpha: float = 0.6

    # Grid settings
    grid_color: str = "#4a4e69"
    grid_alpha: float = 0.3

    # Labels
    title_color: str = "#ffffff"
    title_fontsize: int = 16
    label_color: str = "#e0e0e0"
    label_fontsize: int = 12
    tick_color: str = "#b0b0b0"

    # Markers
    marker_style: str = "o"
    marker_size: int = 8
    marker_color: str = "#f72585"


@dataclass
class SpeedData:
    """Container for speed profile data."""
    time: np.ndarray  # Time points in seconds
    speed: np.ndarray  # Speed values in cm/s
    speed_std: Optional[np.ndarray] = None  # Standard deviation
    acceleration: Optional[np.ndarray] = None  # Acceleration in cm/s^2

    def __post_init__(self):
        """Validate data arrays."""
        assert len(self.time) == len(self.speed), "Time and speed arrays must have same length"
        if self.speed_std is not None:
            assert len(self.speed_std) == len(self.speed), "Speed std must match speed length"
        if self.acceleration is not None:
            # Acceleration might have different length due to differentiation
            pass


class SpeedProfilePlotter:
    """
    Generates professional speed profile visualizations.

    Creates publication-quality plots showing speed over time with
    standard deviation bands, reference lines, and acceleration overlays.

    Attributes:
        style: Plot style configuration
    """

    def __init__(self, style: Optional[SpeedPlotStyle] = None):
        """
        Initialize the speed profile plotter.

        Args:
            style: Plot style configuration
        """
        self.style = style or SpeedPlotStyle()
        plt.style.use('dark_background')

    def calculate_acceleration(
        self,
        speed: np.ndarray,
        time: np.ndarray,
        smoothing_window: int = 5
    ) -> np.ndarray:
        """
        Calculate acceleration from speed data.

        Args:
            speed: Speed values
            time: Time points
            smoothing_window: Window size for smoothing

        Returns:
            Acceleration array
        """
        # Calculate time differences
        dt = np.diff(time)
        dt[dt == 0] = 1e-6  # Avoid division by zero

        # Calculate acceleration
        acceleration = np.diff(speed) / dt

        # Smooth acceleration
        if smoothing_window > 1:
            kernel = np.ones(smoothing_window) / smoothing_window
            acceleration = np.convolve(acceleration, kernel, mode='same')

        return acceleration

    def plot_profile(
        self,
        data: SpeedData,
        belt_speed: Optional[float] = None,
        avg_speed: Optional[float] = None,
        title: str = "Speed Profile",
        ax: Optional[plt.Axes] = None,
        show_legend: bool = True
    ) -> plt.Axes:
        """
        Create a speed profile plot.

        Args:
            data: SpeedData object containing the data
            belt_speed: Optional treadmill belt speed for reference
            avg_speed: Optional average speed for reference
            title: Plot title
            ax: Optional existing axes
            show_legend: Whether to show legend

        Returns:
            Matplotlib axes object
        """
        # Create axes if not provided
        if ax is None:
            fig = plt.figure(
                figsize=self.style.figure_size,
                facecolor=self.style.background_color
            )
            ax = fig.add_subplot(111)

        ax.set_facecolor(self.style.plot_face_color)

        # Plot standard deviation band
        if data.speed_std is not None:
            ax.fill_between(
                data.time,
                data.speed - data.speed_std,
                data.speed + data.speed_std,
                color=self.style.std_color,
                alpha=self.style.std_alpha,
                label='Std. Deviation',
                zorder=1
            )

        # Plot main speed line
        ax.plot(
            data.time,
            data.speed,
            color=self.style.speed_color,
            linewidth=self.style.speed_linewidth,
            alpha=self.style.speed_alpha,
            label='Instantaneous Speed',
            zorder=5
        )

        # Plot belt speed reference
        if belt_speed is not None:
            ax.axhline(
                y=belt_speed,
                color=self.style.belt_speed_color,
                linestyle=self.style.belt_speed_style,
                linewidth=self.style.belt_speed_width,
                label=f'Belt Speed ({belt_speed:.1f} cm/s)',
                zorder=3
            )

        # Plot average speed reference
        if avg_speed is None:
            avg_speed = np.mean(data.speed)

        ax.axhline(
            y=avg_speed,
            color=self.style.avg_speed_color,
            linestyle=self.style.avg_speed_style,
            linewidth=self.style.avg_speed_width,
            label=f'Avg. Speed ({avg_speed:.1f} cm/s)',
            zorder=3
        )

        # Style axes
        ax.set_xlabel('Time (s)', color=self.style.label_color,
                     fontsize=self.style.label_fontsize)
        ax.set_ylabel('Speed (cm/s)', color=self.style.label_color,
                     fontsize=self.style.label_fontsize)
        ax.set_title(title, color=self.style.title_color,
                    fontsize=self.style.title_fontsize, fontweight='bold')

        # Grid
        ax.grid(True, color=self.style.grid_color, alpha=self.style.grid_alpha)

        # Tick colors
        ax.tick_params(colors=self.style.tick_color)
        for spine in ax.spines.values():
            spine.set_color(self.style.grid_color)

        # Legend
        if show_legend:
            ax.legend(
                loc='upper right',
                framealpha=0.8,
                facecolor=self.style.plot_face_color
            )

        # Set reasonable limits
        y_margin = (data.speed.max() - data.speed.min()) * 0.1
        ax.set_ylim(
            data.speed.min() - y_margin - (data.speed_std.max() if data.speed_std is not None else 0),
            data.speed.max() + y_margin + (data.speed_std.max() if data.speed_std is not None else 0)
        )
        ax.set_xlim(data.time.min(), data.time.max())

        return ax

    def add_acceleration_overlay(
        self,
        ax: plt.Axes,
        data: SpeedData,
        secondary_axis: bool = True,
        threshold: float = 0.0
    ) -> Optional[plt.Axes]:
        """
        Add acceleration overlay to an existing speed plot.

        Args:
            ax: Existing axes with speed plot
            data: SpeedData object
            secondary_axis: Whether to use secondary y-axis
            threshold: Acceleration threshold for coloring

        Returns:
            Secondary axes if created, None otherwise
        """
        # Calculate acceleration if not provided
        if data.acceleration is None:
            accel = self.calculate_acceleration(data.speed, data.time)
            # Pad to match speed length
            accel = np.concatenate([[accel[0]], accel])
        else:
            accel = data.acceleration

        # Create secondary axis if requested
        if secondary_axis:
            ax2 = ax.twinx()
            ax2.set_facecolor('none')
        else:
            ax2 = ax

        # Create color-coded line collection
        points = np.array([data.time, accel]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        # Color based on sign
        colors = np.where(
            accel[:-1] > threshold,
            self.style.accel_positive_color,
            self.style.accel_negative_color
        )

        # Plot as colored segments
        for i in range(len(segments)):
            ax2.plot(
                [segments[i, 0, 0], segments[i, 1, 0]],
                [segments[i, 0, 1], segments[i, 1, 1]],
                color=colors[i],
                alpha=self.style.accel_alpha,
                linewidth=1.5
            )

        if secondary_axis:
            ax2.set_ylabel('Acceleration (cm/s^2)',
                          color=self.style.label_color,
                          fontsize=self.style.label_fontsize)
            ax2.tick_params(axis='y', colors=self.style.tick_color)

            # Add zero line
            ax2.axhline(y=0, color=self.style.grid_color, linestyle='-',
                       linewidth=0.5, alpha=0.5)

        return ax2 if secondary_axis else None

    def plot_with_events(
        self,
        data: SpeedData,
        events: Dict[str, List[Tuple[float, float]]],
        belt_speed: Optional[float] = None,
        title: str = "Speed Profile with Events"
    ) -> plt.Figure:
        """
        Create speed profile with event markers (drag, recovery, etc.).

        Args:
            data: SpeedData object
            events: Dictionary mapping event names to list of (start, end) times
            belt_speed: Optional belt speed reference
            title: Plot title

        Returns:
            Matplotlib figure
        """
        fig = plt.figure(
            figsize=self.style.figure_size,
            facecolor=self.style.background_color
        )
        ax = fig.add_subplot(111)

        # Plot base profile
        self.plot_profile(data, belt_speed, title=title, ax=ax, show_legend=False)

        # Event colors
        event_colors = {
            'drag': '#ef476f',     # Red
            'recovery': '#06d6a0', # Green
            'stable': '#ffd166',   # Yellow
            'unknown': '#b0b0b0',  # Gray
        }

        # Shade event regions
        y_min, y_max = ax.get_ylim()
        for event_name, time_ranges in events.items():
            color = event_colors.get(event_name.lower(), event_colors['unknown'])

            for start_t, end_t in time_ranges:
                ax.axvspan(
                    start_t, end_t,
                    alpha=0.2,
                    color=color,
                    label=event_name.capitalize() if time_ranges.index((start_t, end_t)) == 0 else None
                )

        # Update legend
        ax.legend(
            loc='upper right',
            framealpha=0.8,
            facecolor=self.style.plot_face_color
        )

        plt.tight_layout()
        return fig

    def plot_comparison(
        self,
        data_dict: Dict[str, SpeedData],
        colors: Optional[Dict[str, str]] = None,
        title: str = "Speed Comparison"
    ) -> plt.Figure:
        """
        Create comparison plot of multiple speed profiles.

        Args:
            data_dict: Dictionary mapping names to SpeedData objects
            colors: Optional custom colors for each profile
            title: Plot title

        Returns:
            Matplotlib figure
        """
        default_colors = ['#4cc9f0', '#f72585', '#7209b7', '#3a0ca3', '#4361ee', '#06d6a0']

        fig = plt.figure(
            figsize=self.style.figure_size,
            facecolor=self.style.background_color
        )
        ax = fig.add_subplot(111)
        ax.set_facecolor(self.style.plot_face_color)

        for i, (name, data) in enumerate(data_dict.items()):
            color = (colors or {}).get(name, default_colors[i % len(default_colors)])

            # Plot std band with reduced alpha
            if data.speed_std is not None:
                ax.fill_between(
                    data.time,
                    data.speed - data.speed_std,
                    data.speed + data.speed_std,
                    color=color,
                    alpha=0.15,
                    zorder=1
                )

            # Plot main line
            ax.plot(
                data.time,
                data.speed,
                color=color,
                linewidth=2.0,
                label=name,
                zorder=5 + i
            )

        # Style
        ax.set_xlabel('Time (s)', color=self.style.label_color,
                     fontsize=self.style.label_fontsize)
        ax.set_ylabel('Speed (cm/s)', color=self.style.label_color,
                     fontsize=self.style.label_fontsize)
        ax.set_title(title, color=self.style.title_color,
                    fontsize=self.style.title_fontsize, fontweight='bold')
        ax.grid(True, color=self.style.grid_color, alpha=self.style.grid_alpha)
        ax.tick_params(colors=self.style.tick_color)
        ax.legend(loc='upper right', framealpha=0.8, facecolor=self.style.plot_face_color)

        plt.tight_layout()
        return fig

    def plot_distribution(
        self,
        data: SpeedData,
        bins: int = 50,
        title: str = "Speed Distribution"
    ) -> plt.Figure:
        """
        Create a speed distribution histogram.

        Args:
            data: SpeedData object
            bins: Number of histogram bins
            title: Plot title

        Returns:
            Matplotlib figure
        """
        fig = plt.figure(
            figsize=(10, 6),
            facecolor=self.style.background_color
        )
        ax = fig.add_subplot(111)
        ax.set_facecolor(self.style.plot_face_color)

        # Histogram
        n, bins_edges, patches = ax.hist(
            data.speed,
            bins=bins,
            color=self.style.speed_color,
            alpha=0.7,
            edgecolor='white',
            linewidth=0.5
        )

        # Add mean and std lines
        mean_speed = np.mean(data.speed)
        std_speed = np.std(data.speed)

        ax.axvline(mean_speed, color=self.style.avg_speed_color,
                  linestyle='-', linewidth=2, label=f'Mean ({mean_speed:.1f})')
        ax.axvline(mean_speed - std_speed, color=self.style.belt_speed_color,
                  linestyle='--', linewidth=1.5, label=f'-1 Std')
        ax.axvline(mean_speed + std_speed, color=self.style.belt_speed_color,
                  linestyle='--', linewidth=1.5, label=f'+1 Std')

        # Style
        ax.set_xlabel('Speed (cm/s)', color=self.style.label_color,
                     fontsize=self.style.label_fontsize)
        ax.set_ylabel('Count', color=self.style.label_color,
                     fontsize=self.style.label_fontsize)
        ax.set_title(title, color=self.style.title_color,
                    fontsize=self.style.title_fontsize, fontweight='bold')
        ax.grid(True, color=self.style.grid_color, alpha=self.style.grid_alpha, axis='y')
        ax.tick_params(colors=self.style.tick_color)
        ax.legend(loc='upper right', framealpha=0.8, facecolor=self.style.plot_face_color)

        plt.tight_layout()
        return fig

    def create_dashboard(
        self,
        data: SpeedData,
        belt_speed: Optional[float] = None,
        title: str = "Speed Analysis Dashboard"
    ) -> plt.Figure:
        """
        Create a comprehensive speed analysis dashboard.

        Args:
            data: SpeedData object
            belt_speed: Optional belt speed reference
            title: Dashboard title

        Returns:
            Matplotlib figure
        """
        fig = plt.figure(
            figsize=(16, 10),
            facecolor=self.style.background_color
        )

        gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)

        # Main speed profile (top row, full width)
        ax1 = fig.add_subplot(gs[0, :])
        self.plot_profile(data, belt_speed, title="Speed Profile", ax=ax1)
        self.add_acceleration_overlay(ax1, data)

        # Speed distribution (bottom left)
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.set_facecolor(self.style.plot_face_color)
        ax2.hist(data.speed, bins=30, color=self.style.speed_color, alpha=0.7,
                edgecolor='white', linewidth=0.5)
        ax2.axvline(np.mean(data.speed), color=self.style.belt_speed_color,
                   linestyle='-', linewidth=2)
        ax2.set_xlabel('Speed (cm/s)', color=self.style.label_color)
        ax2.set_ylabel('Count', color=self.style.label_color)
        ax2.set_title('Speed Distribution', color=self.style.title_color)
        ax2.grid(True, color=self.style.grid_color, alpha=self.style.grid_alpha, axis='y')
        ax2.tick_params(colors=self.style.tick_color)

        # Statistics panel (bottom center)
        ax3 = fig.add_subplot(gs[1, 1])
        ax3.set_facecolor(self.style.plot_face_color)
        ax3.axis('off')

        stats_text = (
            f"SPEED STATISTICS\n"
            f"{'=' * 25}\n\n"
            f"Mean Speed:     {np.mean(data.speed):.2f} cm/s\n"
            f"Std Deviation:  {np.std(data.speed):.2f} cm/s\n"
            f"Max Speed:      {np.max(data.speed):.2f} cm/s\n"
            f"Min Speed:      {np.min(data.speed):.2f} cm/s\n"
            f"Median Speed:   {np.median(data.speed):.2f} cm/s\n\n"
            f"Duration:       {data.time[-1] - data.time[0]:.2f} s\n"
            f"Samples:        {len(data.speed)}\n"
        )

        if belt_speed is not None:
            diff_pct = (np.mean(data.speed) - belt_speed) / belt_speed * 100
            stats_text += f"\nBelt Speed:     {belt_speed:.2f} cm/s\n"
            stats_text += f"Diff from Belt: {diff_pct:+.1f}%\n"

        ax3.text(
            0.1, 0.9, stats_text,
            transform=ax3.transAxes,
            fontsize=12,
            color=self.style.label_color,
            verticalalignment='top',
            fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor=self.style.plot_face_color, alpha=0.8)
        )

        # Acceleration histogram (bottom right)
        ax4 = fig.add_subplot(gs[1, 2])
        ax4.set_facecolor(self.style.plot_face_color)

        accel = self.calculate_acceleration(data.speed, data.time)
        colors = [self.style.accel_positive_color if a > 0 else self.style.accel_negative_color
                 for a in np.linspace(accel.min(), accel.max(), 30)]

        n, bins_edges, patches = ax4.hist(accel, bins=30, color=self.style.grid_color, alpha=0.7,
                                         edgecolor='white', linewidth=0.5)

        # Color bins based on sign
        bin_centers = (bins_edges[:-1] + bins_edges[1:]) / 2
        for patch, center in zip(patches, bin_centers):
            if center > 0:
                patch.set_facecolor(self.style.accel_positive_color)
            else:
                patch.set_facecolor(self.style.accel_negative_color)
            patch.set_alpha(0.7)

        ax4.axvline(0, color='white', linestyle='-', linewidth=1)
        ax4.set_xlabel('Acceleration (cm/s^2)', color=self.style.label_color)
        ax4.set_ylabel('Count', color=self.style.label_color)
        ax4.set_title('Acceleration Distribution', color=self.style.title_color)
        ax4.grid(True, color=self.style.grid_color, alpha=self.style.grid_alpha, axis='y')
        ax4.tick_params(colors=self.style.tick_color)

        # Main title
        fig.suptitle(title, color=self.style.title_color,
                    fontsize=self.style.title_fontsize + 2, fontweight='bold', y=0.98)

        plt.tight_layout(rect=[0, 0, 1, 0.96])

        return fig

    def save_figure(
        self,
        fig: plt.Figure,
        filepath: Union[str, Path],
        format: str = "png",
        transparent: bool = False
    ) -> None:
        """
        Save figure to file.

        Args:
            fig: Matplotlib figure
            filepath: Output path
            format: Output format
            transparent: Whether to use transparent background
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        fig.savefig(
            filepath,
            format=format,
            dpi=self.style.dpi,
            facecolor=self.style.background_color if not transparent else 'none',
            edgecolor='none',
            bbox_inches='tight',
            transparent=transparent
        )

        print(f"Saved figure to: {filepath}")


def demo_speed_plots():
    """
    Demonstrate speed profile plotting with synthetic data.
    """
    np.random.seed(42)

    plotter = SpeedProfilePlotter()

    # Generate synthetic speed data
    duration = 5.0  # seconds
    fps = 30
    n_samples = int(duration * fps)

    time = np.linspace(0, duration, n_samples)

    # Simulate speed with some variation
    base_speed = 15.0  # cm/s
    speed = base_speed + 3 * np.sin(2 * np.pi * time / 2) + np.random.normal(0, 1, n_samples)
    speed = np.maximum(speed, 0)  # No negative speeds

    speed_std = 1.5 + 0.5 * np.random.random(n_samples)

    data = SpeedData(
        time=time,
        speed=speed,
        speed_std=speed_std
    )

    # Basic profile
    print("Creating basic speed profile...")
    fig1 = plt.figure(figsize=(14, 8), facecolor='#1a1a2e')
    ax1 = fig1.add_subplot(111)
    plotter.plot_profile(data, belt_speed=15.0, title="Mouse Locomotion Speed", ax=ax1)
    plt.show()

    # Profile with acceleration
    print("Creating profile with acceleration overlay...")
    fig2 = plt.figure(figsize=(14, 8), facecolor='#1a1a2e')
    ax2 = fig2.add_subplot(111)
    plotter.plot_profile(data, belt_speed=15.0, title="Speed with Acceleration", ax=ax2)
    plotter.add_acceleration_overlay(ax2, data)
    plt.show()

    # Dashboard
    print("Creating speed dashboard...")
    fig3 = plotter.create_dashboard(data, belt_speed=15.0, title="Speed Analysis Dashboard")
    plt.show()

    print("Demo complete!")


if __name__ == "__main__":
    demo_speed_plots()
