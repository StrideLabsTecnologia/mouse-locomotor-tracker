"""
Circular Coordination Plots
===========================

Generates polar plots for limb coordination analysis using Rayleigh statistics.
Shows phase relationships between limb pairs.

Author: Stride Labs
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyArrowPatch
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import circmean, circstd
from typing import List, Tuple, Optional, Dict, Any, Union
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class CircularPlotStyle:
    """Style configuration for circular plots."""
    # Figure settings
    figure_size: Tuple[float, float] = (14, 10)
    dpi: int = 150
    background_color: str = "#1a1a2e"
    face_color: str = "#16213e"

    # Scatter settings
    scatter_color: str = "#4cc9f0"
    scatter_alpha: float = 0.6
    scatter_size: int = 80
    scatter_edge_color: str = "#ffffff"
    scatter_edge_width: float = 0.5

    # Mean vector settings
    mean_vector_color: str = "#f72585"
    mean_vector_width: float = 4.0
    mean_vector_head_width: float = 0.15
    mean_vector_head_length: float = 0.1

    # Grid settings
    grid_color: str = "#4a4e69"
    grid_alpha: float = 0.5
    grid_linestyle: str = "-"

    # Labels
    title_color: str = "#ffffff"
    title_fontsize: int = 14
    label_color: str = "#e0e0e0"
    label_fontsize: int = 11

    # R-value display
    r_value_color: str = "#4cc9f0"
    r_value_fontsize: int = 12


@dataclass
class LimbPair:
    """Definition of a limb pair for coordination analysis."""
    name: str
    limb1: str
    limb2: str
    display_name: str = ""

    def __post_init__(self):
        if not self.display_name:
            self.display_name = f"{self.limb1} - {self.limb2}"


# Standard limb pairs for quadrupedal locomotion
STANDARD_LIMB_PAIRS = [
    LimbPair("left_right_front", "front_left", "front_right", "Left-Right Front"),
    LimbPair("left_right_hind", "hind_left", "hind_right", "Left-Right Hind"),
    LimbPair("front_hind_left", "front_left", "hind_left", "Front-Hind Left"),
    LimbPair("front_hind_right", "front_right", "hind_right", "Front-Hind Right"),
    LimbPair("diagonal_fl_hr", "front_left", "hind_right", "Diagonal FL-HR"),
    LimbPair("diagonal_fr_hl", "front_right", "hind_left", "Diagonal FR-HL"),
]


class CoordinationPlotter:
    """
    Generates polar coordination plots for limb phase analysis.

    Creates professional-quality polar plots showing the phase relationship
    between limb pairs, including individual phase scatter and Rayleigh
    mean vector visualization.

    Attributes:
        style: Plot style configuration
        limb_pairs: List of limb pairs to analyze
    """

    def __init__(
        self,
        style: Optional[CircularPlotStyle] = None,
        limb_pairs: Optional[List[LimbPair]] = None
    ):
        """
        Initialize the coordination plotter.

        Args:
            style: Plot style configuration
            limb_pairs: List of limb pairs to analyze
        """
        self.style = style or CircularPlotStyle()
        self.limb_pairs = limb_pairs or STANDARD_LIMB_PAIRS

        # Set up matplotlib style
        plt.style.use('dark_background')

    def circular_mean(self, phases: np.ndarray) -> Tuple[float, float]:
        """
        Calculate circular mean and Rayleigh R-value.

        Args:
            phases: Array of phase angles in radians

        Returns:
            Tuple of (mean_angle, r_value)
        """
        if len(phases) == 0:
            return 0.0, 0.0

        # Calculate mean vector
        cos_sum = np.sum(np.cos(phases))
        sin_sum = np.sum(np.sin(phases))
        n = len(phases)

        # Mean angle
        mean_angle = np.arctan2(sin_sum, cos_sum)

        # R-value (vector strength)
        r_value = np.sqrt(cos_sum ** 2 + sin_sum ** 2) / n

        return mean_angle, r_value

    def rayleigh_test(self, phases: np.ndarray) -> Tuple[float, float]:
        """
        Perform Rayleigh test for circular uniformity.

        Args:
            phases: Array of phase angles in radians

        Returns:
            Tuple of (z_statistic, p_value)
        """
        n = len(phases)
        if n < 2:
            return 0.0, 1.0

        _, r = self.circular_mean(phases)
        z = n * r ** 2

        # Approximate p-value
        p_value = np.exp(-z) * (1 + (2 * z - z ** 2) / (4 * n) -
                                (24 * z - 132 * z ** 2 + 76 * z ** 3 - 9 * z ** 4) / (288 * n ** 2))

        return z, max(0, min(1, p_value))

    def plot_single(
        self,
        phases: np.ndarray,
        pair_name: str = "Coordination",
        ax: Optional[plt.Axes] = None,
        show_stats: bool = True
    ) -> plt.Axes:
        """
        Create a single circular coordination plot.

        Args:
            phases: Array of phase angles in radians
            pair_name: Name for the plot title
            ax: Optional existing polar axes
            show_stats: Whether to display statistics

        Returns:
            Matplotlib axes object
        """
        # Create axes if not provided
        if ax is None:
            fig = plt.figure(figsize=(8, 8), facecolor=self.style.background_color)
            ax = fig.add_subplot(111, polar=True)

        ax.set_facecolor(self.style.face_color)

        # Configure polar plot
        ax.set_theta_offset(np.pi / 2)  # 0 at top
        ax.set_theta_direction(-1)  # Clockwise
        ax.set_rlim(0, 1.15)
        ax.set_rticks([0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(['', '0.5', '', '1.0'], color=self.style.label_color,
                          fontsize=self.style.label_fontsize - 2)

        # Style grid
        ax.grid(True, color=self.style.grid_color, alpha=self.style.grid_alpha,
               linestyle=self.style.grid_linestyle)
        ax.spines['polar'].set_visible(False)

        # Plot individual phases as scatter
        if len(phases) > 0:
            ax.scatter(
                phases,
                np.ones(len(phases)),
                s=self.style.scatter_size,
                c=self.style.scatter_color,
                alpha=self.style.scatter_alpha,
                edgecolors=self.style.scatter_edge_color,
                linewidths=self.style.scatter_edge_width,
                zorder=10
            )

            # Calculate and plot mean vector
            mean_angle, r_value = self.circular_mean(phases)

            # Draw mean vector as arrow
            ax.annotate(
                '',
                xy=(mean_angle, r_value),
                xytext=(0, 0),
                arrowprops=dict(
                    arrowstyle=f'->,head_width={self.style.mean_vector_head_width},'
                              f'head_length={self.style.mean_vector_head_length}',
                    color=self.style.mean_vector_color,
                    lw=self.style.mean_vector_width,
                    shrinkA=0,
                    shrinkB=0
                ),
                zorder=20
            )

            # Draw the line part of the vector
            ax.plot([0, mean_angle], [0, r_value],
                   color=self.style.mean_vector_color,
                   linewidth=self.style.mean_vector_width,
                   zorder=15)

            # Add statistics text
            if show_stats:
                z_stat, p_value = self.rayleigh_test(phases)
                circ_std = circstd(phases) if len(phases) > 1 else 0

                stats_text = (f"n = {len(phases)}\n"
                            f"R = {r_value:.3f}\n"
                            f"Mean = {np.degrees(mean_angle):.1f}deg\n"
                            f"p = {p_value:.4f}")

                ax.text(
                    0.02, 0.98, stats_text,
                    transform=ax.transAxes,
                    fontsize=self.style.r_value_fontsize,
                    color=self.style.r_value_color,
                    verticalalignment='top',
                    fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor=self.style.face_color, alpha=0.8)
                )

        # Title
        ax.set_title(
            pair_name,
            color=self.style.title_color,
            fontsize=self.style.title_fontsize,
            fontweight='bold',
            pad=20
        )

        return ax

    def plot_all_pairs(
        self,
        phase_data: Dict[str, np.ndarray],
        title: str = "Limb Coordination Analysis"
    ) -> plt.Figure:
        """
        Create a multi-panel figure with all limb pair coordinations.

        Args:
            phase_data: Dictionary mapping limb pair names to phase arrays
            title: Overall figure title

        Returns:
            Matplotlib figure object
        """
        # Determine grid layout
        n_pairs = len(self.limb_pairs)
        n_cols = min(3, n_pairs)
        n_rows = (n_pairs + n_cols - 1) // n_cols

        fig = plt.figure(
            figsize=(self.style.figure_size[0], self.style.figure_size[1] * n_rows / 2),
            facecolor=self.style.background_color
        )

        gs = GridSpec(n_rows, n_cols, figure=fig, hspace=0.4, wspace=0.3)

        for i, pair in enumerate(self.limb_pairs):
            row = i // n_cols
            col = i % n_cols

            ax = fig.add_subplot(gs[row, col], polar=True)

            # Get phase data for this pair
            phases = phase_data.get(pair.name, np.array([]))

            self.plot_single(phases, pair.display_name, ax)

        # Overall title
        fig.suptitle(
            title,
            color=self.style.title_color,
            fontsize=self.style.title_fontsize + 4,
            fontweight='bold',
            y=0.98
        )

        plt.tight_layout(rect=[0, 0, 1, 0.96])

        return fig

    def plot_comparison(
        self,
        phase_data_groups: Dict[str, Dict[str, np.ndarray]],
        pair_name: str,
        group_colors: Optional[Dict[str, str]] = None
    ) -> plt.Figure:
        """
        Create a comparison plot showing multiple conditions on one polar plot.

        Args:
            phase_data_groups: Dictionary mapping group names to phase data dicts
            pair_name: Which limb pair to plot
            group_colors: Optional custom colors for each group

        Returns:
            Matplotlib figure object
        """
        default_colors = ['#4cc9f0', '#f72585', '#7209b7', '#3a0ca3', '#4361ee']

        fig = plt.figure(figsize=(10, 10), facecolor=self.style.background_color)
        ax = fig.add_subplot(111, polar=True)
        ax.set_facecolor(self.style.face_color)

        # Configure polar plot
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_rlim(0, 1.15)
        ax.grid(True, color=self.style.grid_color, alpha=self.style.grid_alpha)
        ax.spines['polar'].set_visible(False)

        legend_handles = []

        for i, (group_name, phase_data) in enumerate(phase_data_groups.items()):
            phases = phase_data.get(pair_name, np.array([]))
            if len(phases) == 0:
                continue

            color = (group_colors or {}).get(group_name, default_colors[i % len(default_colors)])

            # Scatter
            scatter = ax.scatter(
                phases,
                np.ones(len(phases)) - i * 0.05,  # Slight offset for visibility
                s=self.style.scatter_size * 0.8,
                c=color,
                alpha=self.style.scatter_alpha,
                label=group_name,
                zorder=10 + i
            )
            legend_handles.append(scatter)

            # Mean vector
            mean_angle, r_value = self.circular_mean(phases)
            ax.annotate(
                '',
                xy=(mean_angle, r_value),
                xytext=(0, 0),
                arrowprops=dict(
                    arrowstyle='->',
                    color=color,
                    lw=3,
                    shrinkA=0,
                    shrinkB=0
                ),
                zorder=20 + i
            )

        ax.legend(loc='upper right', framealpha=0.8)

        # Find the pair display name
        display_name = pair_name
        for pair in self.limb_pairs:
            if pair.name == pair_name:
                display_name = pair.display_name
                break

        ax.set_title(
            f"{display_name} - Group Comparison",
            color=self.style.title_color,
            fontsize=self.style.title_fontsize,
            fontweight='bold',
            pad=20
        )

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
            fig: Matplotlib figure to save
            filepath: Output file path
            format: Output format (png, pdf, svg, etc.)
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

    def create_animated_polar(
        self,
        phase_sequence: List[np.ndarray],
        pair_name: str = "Coordination",
        output_path: Optional[Union[str, Path]] = None,
        fps: int = 10
    ) -> List[np.ndarray]:
        """
        Create frames for an animated polar plot.

        Args:
            phase_sequence: List of phase arrays for each time point
            pair_name: Name for the plot
            output_path: Optional path to save as GIF
            fps: Frames per second for animation

        Returns:
            List of frame images as numpy arrays
        """
        frames = []

        for i, phases in enumerate(phase_sequence):
            fig = plt.figure(figsize=(8, 8), facecolor=self.style.background_color)
            ax = fig.add_subplot(111, polar=True)

            self.plot_single(phases, f"{pair_name} (t={i})", ax)

            # Convert figure to image
            fig.canvas.draw()
            img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))

            frames.append(img)
            plt.close(fig)

        # Save as GIF if path provided
        if output_path:
            try:
                import imageio
                output_path = Path(output_path)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                imageio.mimsave(str(output_path), frames, fps=fps)
                print(f"Saved animation to: {output_path}")
            except ImportError:
                print("imageio not installed. Cannot save GIF animation.")

        return frames


def demo_circular_plots():
    """
    Demonstrate circular coordination plots with synthetic data.
    """
    np.random.seed(42)

    plotter = CoordinationPlotter()

    # Generate synthetic phase data
    # Alternating gait: LR pairs at ~180 degrees (pi radians)
    # Diagonal pairs at ~0 degrees (synchronous)

    phase_data = {
        "left_right_front": np.random.vonmises(np.pi, 3, 50),  # ~180 deg
        "left_right_hind": np.random.vonmises(np.pi, 3, 50),   # ~180 deg
        "front_hind_left": np.random.vonmises(np.pi, 2, 50),   # ~180 deg
        "front_hind_right": np.random.vonmises(np.pi, 2, 50),  # ~180 deg
        "diagonal_fl_hr": np.random.vonmises(0, 5, 50),        # ~0 deg (sync)
        "diagonal_fr_hl": np.random.vonmises(0, 5, 50),        # ~0 deg (sync)
    }

    # Single plot demo
    print("Creating single coordination plot...")
    fig1 = plt.figure(figsize=(8, 8), facecolor='#1a1a2e')
    ax1 = fig1.add_subplot(111, polar=True)
    plotter.plot_single(phase_data["left_right_hind"], "Left-Right Hind Coordination", ax1)
    plt.show()

    # Multi-panel plot
    print("Creating multi-panel coordination plot...")
    fig2 = plotter.plot_all_pairs(phase_data, "Mouse Locomotion - Limb Coordination")
    plt.show()

    # Comparison plot
    print("Creating comparison plot...")
    control_data = {
        "left_right_hind": np.random.vonmises(np.pi, 4, 40),
    }
    treatment_data = {
        "left_right_hind": np.random.vonmises(np.pi * 0.8, 2, 40),  # Slightly shifted
    }

    fig3 = plotter.plot_comparison(
        {"Control": control_data, "Treatment": treatment_data},
        "left_right_hind",
        {"Control": "#4cc9f0", "Treatment": "#f72585"}
    )
    plt.show()

    print("Demo complete!")


if __name__ == "__main__":
    demo_circular_plots()
