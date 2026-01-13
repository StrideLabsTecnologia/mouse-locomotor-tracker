#!/usr/bin/env python3
"""
Publication-Ready Figure Generator
===================================

Generates high-quality figures suitable for scientific publications.
Follows journal guidelines for:
- Nature / Nature Methods
- Cell
- PNAS
- Journal of Neurophysiology
- Frontiers

Figure specifications:
- Resolution: 300-600 DPI
- Font: Arial/Helvetica (sans-serif)
- Color: Colorblind-friendly palette
- Format: PDF, SVG, PNG, TIFF

Author: Stride Labs
Version: 1.0.0
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import warnings

# Try to import seaborn for enhanced styling
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False


# =============================================================================
# Publication Style Configuration
# =============================================================================

# Colorblind-friendly palette (Paul Tol's palette)
COLORS = {
    'blue': '#4477AA',
    'cyan': '#66CCEE',
    'green': '#228833',
    'yellow': '#CCBB44',
    'red': '#EE6677',
    'purple': '#AA3377',
    'grey': '#BBBBBB',
    'black': '#000000',
}

# Activity colors
ACTIVITY_COLORS = {
    'rest': '#BBBBBB',
    'walk': '#4477AA',
    'trot': '#CCBB44',
    'gallop': '#EE6677',
}

# Journal-specific configurations
JOURNAL_CONFIGS = {
    'nature': {
        'figsize_single': (3.5, 2.5),   # inches (89mm width)
        'figsize_double': (7.0, 4.0),   # inches (183mm width)
        'figsize_full': (7.0, 9.0),     # Full page
        'fontsize': 7,
        'fontsize_title': 8,
        'fontsize_label': 7,
        'dpi': 300,
        'font_family': 'Arial',
    },
    'cell': {
        'figsize_single': (3.35, 2.5),
        'figsize_double': (6.85, 4.0),
        'figsize_full': (6.85, 9.0),
        'fontsize': 7,
        'fontsize_title': 8,
        'fontsize_label': 7,
        'dpi': 300,
        'font_family': 'Arial',
    },
    'default': {
        'figsize_single': (4.0, 3.0),
        'figsize_double': (8.0, 4.0),
        'figsize_full': (8.0, 10.0),
        'fontsize': 9,
        'fontsize_title': 11,
        'fontsize_label': 9,
        'dpi': 300,
        'font_family': 'Arial',
    }
}


@dataclass
class FigureConfig:
    """Configuration for figure generation."""
    journal: str = 'default'
    dpi: int = 300
    format: str = 'pdf'
    transparent: bool = False
    tight_layout: bool = True


class PublicationFigureGenerator:
    """
    Generate publication-quality figures for gait analysis.

    Example:
        >>> generator = PublicationFigureGenerator(journal='nature')
        >>> generator.plot_velocity_timeseries(velocity, timestamps, output='fig1a.pdf')
        >>> generator.plot_gait_summary(metrics, output='fig2.pdf')
    """

    def __init__(
        self,
        journal: str = 'default',
        output_dir: Optional[Union[str, Path]] = None,
    ):
        """
        Initialize figure generator.

        Args:
            journal: Target journal style ('nature', 'cell', 'default')
            output_dir: Directory for saving figures
        """
        self.journal = journal
        self.config = JOURNAL_CONFIGS.get(journal, JOURNAL_CONFIGS['default'])
        self.output_dir = Path(output_dir) if output_dir else Path('.')

        # Set matplotlib style
        self._setup_style()

    def _setup_style(self):
        """Configure matplotlib for publication quality."""
        plt.rcParams.update({
            # Font
            'font.family': 'sans-serif',
            'font.sans-serif': [self.config['font_family'], 'DejaVu Sans'],
            'font.size': self.config['fontsize'],

            # Axes
            'axes.labelsize': self.config['fontsize_label'],
            'axes.titlesize': self.config['fontsize_title'],
            'axes.linewidth': 0.8,
            'axes.spines.top': False,
            'axes.spines.right': False,

            # Ticks
            'xtick.labelsize': self.config['fontsize'],
            'ytick.labelsize': self.config['fontsize'],
            'xtick.major.width': 0.8,
            'ytick.major.width': 0.8,
            'xtick.major.size': 3,
            'ytick.major.size': 3,

            # Legend
            'legend.fontsize': self.config['fontsize'],
            'legend.frameon': False,

            # Figure
            'figure.dpi': self.config['dpi'],
            'savefig.dpi': self.config['dpi'],
            'savefig.bbox': 'tight',
            'savefig.pad_inches': 0.05,

            # Lines
            'lines.linewidth': 1.0,

            # PDF
            'pdf.fonttype': 42,  # TrueType fonts
            'ps.fonttype': 42,
        })

        if HAS_SEABORN:
            sns.set_style("ticks")
            sns.set_context("paper", font_scale=0.9)

    def _save_figure(
        self,
        fig: plt.Figure,
        output: Union[str, Path],
        formats: List[str] = None,
    ):
        """Save figure in specified formats."""
        output = Path(output)

        if formats is None:
            formats = [output.suffix.lstrip('.') or 'pdf']

        for fmt in formats:
            output_path = output.with_suffix(f'.{fmt}')
            fig.savefig(
                output_path,
                format=fmt,
                dpi=self.config['dpi'],
                bbox_inches='tight',
                transparent=False,
            )
            print(f"  Saved: {output_path}")

    def plot_velocity_timeseries(
        self,
        velocity: np.ndarray,
        timestamps: np.ndarray,
        output: Optional[Union[str, Path]] = None,
        title: str = None,
        show_activity: bool = True,
        activity_thresholds: Tuple[float, float, float] = (2.0, 10.0, 25.0),
    ) -> plt.Figure:
        """
        Plot velocity over time with activity classification.

        Args:
            velocity: Velocity array (cm/s)
            timestamps: Time array (seconds)
            output: Output file path
            title: Figure title
            show_activity: Show activity classification bands
            activity_thresholds: (rest, walk, trot) thresholds in cm/s
        """
        fig, ax = plt.subplots(figsize=self.config['figsize_double'])

        # Activity background bands
        if show_activity:
            rest_t, walk_t, trot_t = activity_thresholds
            max_vel = max(velocity.max() * 1.1, 35)

            ax.axhspan(0, rest_t, alpha=0.2, color=ACTIVITY_COLORS['rest'], label='Rest')
            ax.axhspan(rest_t, walk_t, alpha=0.2, color=ACTIVITY_COLORS['walk'], label='Walk')
            ax.axhspan(walk_t, trot_t, alpha=0.2, color=ACTIVITY_COLORS['trot'], label='Trot')
            ax.axhspan(trot_t, max_vel, alpha=0.2, color=ACTIVITY_COLORS['gallop'], label='Gallop')

        # Velocity line
        ax.plot(timestamps, velocity, color=COLORS['blue'], linewidth=1.0, alpha=0.8)

        # Mean line
        mean_vel = np.mean(velocity[velocity > 0.1])
        ax.axhline(mean_vel, color=COLORS['red'], linestyle='--', linewidth=0.8,
                   label=f'Mean: {mean_vel:.1f} cm/s')

        # Labels
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Velocity (cm/s)')

        if title:
            ax.set_title(title)

        ax.set_xlim(timestamps[0], timestamps[-1])
        ax.set_ylim(0, max(velocity.max() * 1.1, 35))

        # Legend
        ax.legend(loc='upper right', fontsize=self.config['fontsize'] - 1)

        plt.tight_layout()

        if output:
            self._save_figure(fig, output, ['pdf', 'png'])

        return fig

    def plot_trajectory_heatmap(
        self,
        positions: np.ndarray,
        output: Optional[Union[str, Path]] = None,
        title: str = "Locomotion Trajectory",
        velocity: Optional[np.ndarray] = None,
        nbins: int = 50,
    ) -> plt.Figure:
        """
        Plot trajectory with density heatmap.

        Args:
            positions: Array of (x, y) positions
            output: Output file path
            title: Figure title
            velocity: Optional velocity for color coding
            nbins: Number of bins for heatmap
        """
        fig, axes = plt.subplots(1, 2, figsize=self.config['figsize_double'])

        # Left: Trajectory with velocity coloring
        ax1 = axes[0]
        if velocity is not None:
            scatter = ax1.scatter(
                positions[:, 0], positions[:, 1],
                c=velocity, cmap='viridis', s=2, alpha=0.6
            )
            cbar = plt.colorbar(scatter, ax=ax1, label='Velocity (cm/s)')
            cbar.ax.tick_params(labelsize=self.config['fontsize'] - 1)
        else:
            ax1.plot(positions[:, 0], positions[:, 1],
                     color=COLORS['blue'], linewidth=0.5, alpha=0.7)

        ax1.set_xlabel('X (pixels)')
        ax1.set_ylabel('Y (pixels)')
        ax1.set_title('Trajectory')
        ax1.set_aspect('equal')
        ax1.invert_yaxis()

        # Right: Density heatmap
        ax2 = axes[1]
        h, xedges, yedges = np.histogram2d(
            positions[:, 0], positions[:, 1], bins=nbins
        )

        extent = [xedges[0], xedges[-1], yedges[-1], yedges[0]]
        im = ax2.imshow(h.T, extent=extent, origin='upper',
                        cmap='hot', aspect='auto', interpolation='gaussian')

        cbar2 = plt.colorbar(im, ax=ax2, label='Density')
        cbar2.ax.tick_params(labelsize=self.config['fontsize'] - 1)

        ax2.set_xlabel('X (pixels)')
        ax2.set_ylabel('Y (pixels)')
        ax2.set_title('Density Heatmap')

        fig.suptitle(title, fontsize=self.config['fontsize_title'], y=1.02)
        plt.tight_layout()

        if output:
            self._save_figure(fig, output, ['pdf', 'png'])

        return fig

    def plot_gait_summary(
        self,
        metrics: Dict,
        output: Optional[Union[str, Path]] = None,
        title: str = "Gait Analysis Summary",
    ) -> plt.Figure:
        """
        Create comprehensive gait analysis summary figure.

        Args:
            metrics: Dictionary of gait metrics (from ScientificGaitAnalyzer.to_dict())
            output: Output file path
            title: Figure title
        """
        fig = plt.figure(figsize=self.config['figsize_full'])
        gs = GridSpec(3, 3, figure=fig, hspace=0.4, wspace=0.35)

        # =================================================================
        # A: Velocity Statistics (bar chart with error bars)
        # =================================================================
        ax_vel = fig.add_subplot(gs[0, 0])

        vel_data = metrics.get('spatiotemporal', {}).get('velocity', {})
        categories = ['Mean', 'Max', 'Min']
        values = [vel_data.get('mean', 0), vel_data.get('max', 0), vel_data.get('min', 0)]
        errors = [vel_data.get('sd', 0), 0, 0]

        bars = ax_vel.bar(categories, values, color=[COLORS['blue'], COLORS['green'], COLORS['red']],
                          edgecolor='black', linewidth=0.5)
        ax_vel.errorbar(categories[0], values[0], yerr=errors[0], fmt='none',
                        color='black', capsize=3)

        ax_vel.set_ylabel('Velocity (cm/s)')
        ax_vel.set_title('A. Velocity', fontweight='bold', loc='left')

        # Add value labels
        for bar, val in zip(bars, values):
            ax_vel.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                        f'{val:.1f}', ha='center', va='bottom', fontsize=self.config['fontsize'] - 1)

        # =================================================================
        # B: Stride Metrics
        # =================================================================
        ax_stride = fig.add_subplot(gs[0, 1])

        stride_data = metrics.get('spatiotemporal', {}).get('stride_length', {})
        cadence_data = metrics.get('spatiotemporal', {}).get('cadence', {})

        x = ['Stride\nLength', 'Cadence']
        y = [stride_data.get('mean', 0), cadence_data.get('mean', 0) * 10]  # Scale cadence
        yerr = [stride_data.get('sd', 0), cadence_data.get('sd', 0) * 10]

        bars = ax_stride.bar(x, y, color=[COLORS['cyan'], COLORS['purple']],
                             edgecolor='black', linewidth=0.5)
        ax_stride.errorbar(x, y, yerr=yerr, fmt='none', color='black', capsize=3)

        ax_stride.set_ylabel('Value')
        ax_stride.set_title('B. Stride Metrics', fontweight='bold', loc='left')

        # Secondary y-axis label
        ax_stride.text(0, y[0] + yerr[0] + 0.5, f'{stride_data.get("mean", 0):.2f} cm',
                       ha='center', fontsize=self.config['fontsize'] - 1)
        ax_stride.text(1, y[1] + yerr[1] + 0.5, f'{cadence_data.get("mean", 0):.2f} /s',
                       ha='center', fontsize=self.config['fontsize'] - 1)

        # =================================================================
        # C: Activity Distribution (pie chart)
        # =================================================================
        ax_activity = fig.add_subplot(gs[0, 2])

        activity = metrics.get('activity', {})
        labels = ['Rest', 'Walk', 'Trot', 'Gallop']
        sizes = [
            activity.get('time_resting_pct', 0),
            activity.get('time_walking_pct', 0),
            activity.get('time_trotting_pct', 0),
            activity.get('time_galloping_pct', 0),
        ]
        colors = [ACTIVITY_COLORS['rest'], ACTIVITY_COLORS['walk'],
                  ACTIVITY_COLORS['trot'], ACTIVITY_COLORS['gallop']]

        # Filter out zero values
        non_zero = [(l, s, c) for l, s, c in zip(labels, sizes, colors) if s > 0]
        if non_zero:
            labels, sizes, colors = zip(*non_zero)

            wedges, texts, autotexts = ax_activity.pie(
                sizes, labels=labels, colors=colors,
                autopct='%1.1f%%', startangle=90,
                textprops={'fontsize': self.config['fontsize'] - 1}
            )
            for autotext in autotexts:
                autotext.set_fontsize(self.config['fontsize'] - 1)

        ax_activity.set_title('C. Activity Distribution', fontweight='bold', loc='left')

        # =================================================================
        # D: Step Cycle Phases
        # =================================================================
        ax_phase = fig.add_subplot(gs[1, 0])

        swing_pct = metrics.get('spatiotemporal', {}).get('swing_time', {}).get('percent', 40)
        stance_pct = metrics.get('spatiotemporal', {}).get('stance_time', {}).get('percent', 60)

        phases = ['Swing', 'Stance']
        pcts = [swing_pct, stance_pct]
        colors_phase = [COLORS['cyan'], COLORS['yellow']]

        bars = ax_phase.barh(phases, pcts, color=colors_phase, edgecolor='black', linewidth=0.5)
        ax_phase.set_xlim(0, 100)
        ax_phase.set_xlabel('% of Step Cycle')
        ax_phase.set_title('D. Gait Phase', fontweight='bold', loc='left')

        for bar, pct in zip(bars, pcts):
            ax_phase.text(pct + 2, bar.get_y() + bar.get_height()/2,
                          f'{pct:.0f}%', va='center', fontsize=self.config['fontsize'] - 1)

        # =================================================================
        # E: Kinematic Parameters
        # =================================================================
        ax_kin = fig.add_subplot(gs[1, 1])

        kin = metrics.get('kinematic', {})
        acc = kin.get('acceleration', {})
        jerk = kin.get('jerk', {})

        x = ['Acceleration\n(cm/s²)', 'Jerk\n(cm/s³)']
        y = [acc.get('mean', 0), jerk.get('mean', 0) / 10]  # Scale jerk
        yerr = [acc.get('sd', 0), jerk.get('sd', 0) / 10]

        bars = ax_kin.bar(x, y, color=[COLORS['green'], COLORS['red']],
                          edgecolor='black', linewidth=0.5)
        ax_kin.errorbar(x, y, yerr=yerr, fmt='none', color='black', capsize=3)
        ax_kin.set_ylabel('Value (scaled)')
        ax_kin.set_title('E. Kinematics', fontweight='bold', loc='left')

        # =================================================================
        # F: Variability Metrics
        # =================================================================
        ax_var = fig.add_subplot(gs[1, 2])

        var = metrics.get('variability', {})

        labels = ['Regularity\nIndex', 'Velocity\nStability', 'Stride\nVariability']
        values = [
            var.get('regularity_index', 0),
            var.get('velocity_stability', 0),
            100 - var.get('stride_variability', 0),  # Invert for consistency
        ]

        colors_var = [COLORS['green'], COLORS['blue'], COLORS['purple']]
        bars = ax_var.bar(labels, values, color=colors_var, edgecolor='black', linewidth=0.5)
        ax_var.set_ylim(0, 100)
        ax_var.set_ylabel('Score (%)')
        ax_var.set_title('F. Regularity', fontweight='bold', loc='left')

        for bar, val in zip(bars, values):
            ax_var.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                        f'{val:.0f}', ha='center', fontsize=self.config['fontsize'] - 1)

        # =================================================================
        # G: Distance and Duration
        # =================================================================
        ax_dist = fig.add_subplot(gs[2, 0])

        dist = metrics.get('distance_duration', {})

        labels = ['Total\nDistance', 'Active\nTime', 'Rest\nTime']
        values = [
            dist.get('total_distance_cm', 0) / 10,  # Scale to fit
            dist.get('active_duration_s', 0),
            dist.get('rest_duration_s', 0),
        ]

        bars = ax_dist.bar(labels, values, color=[COLORS['blue'], COLORS['green'], COLORS['grey']],
                           edgecolor='black', linewidth=0.5)
        ax_dist.set_ylabel('Value (scaled)')
        ax_dist.set_title('G. Distance & Duration', fontweight='bold', loc='left')

        # Add actual values
        actual = [
            f'{dist.get("total_distance_cm", 0):.0f} cm',
            f'{dist.get("active_duration_s", 0):.1f} s',
            f'{dist.get("rest_duration_s", 0):.1f} s',
        ]
        for bar, txt in zip(bars, actual):
            ax_dist.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                         txt, ha='center', fontsize=self.config['fontsize'] - 1)

        # =================================================================
        # H: Data Quality
        # =================================================================
        ax_qual = fig.add_subplot(gs[2, 1])

        qual = metrics.get('quality', {})

        labels = ['Tracking\nRate', 'Quality\nScore']
        values = [qual.get('tracking_rate', 0), qual.get('data_quality_score', 0)]

        bars = ax_qual.bar(labels, values, color=[COLORS['cyan'], COLORS['green']],
                           edgecolor='black', linewidth=0.5)
        ax_qual.set_ylim(0, 100)
        ax_qual.set_ylabel('Percentage')
        ax_qual.set_title('H. Data Quality', fontweight='bold', loc='left')

        for bar, val in zip(bars, values):
            ax_qual.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                         f'{val:.1f}%', ha='center', fontsize=self.config['fontsize'] - 1)

        # =================================================================
        # I: Summary Statistics Table
        # =================================================================
        ax_table = fig.add_subplot(gs[2, 2])
        ax_table.axis('off')

        table_data = [
            ['Parameter', 'Value', 'Unit'],
            ['Velocity (mean)', f"{vel_data.get('mean', 0):.2f} ± {vel_data.get('sd', 0):.2f}", 'cm/s'],
            ['Stride Length', f"{stride_data.get('mean', 0):.2f} ± {stride_data.get('sd', 0):.2f}", 'cm'],
            ['Cadence', f"{cadence_data.get('mean', 0):.2f}", 'steps/s'],
            ['Duty Factor', f"{metrics.get('spatiotemporal', {}).get('duty_factor', 0):.2f}", '-'],
            ['Total Distance', f"{dist.get('total_distance_cm', 0):.1f}", 'cm'],
            ['Tracking Rate', f"{qual.get('tracking_rate', 0):.1f}", '%'],
        ]

        table = ax_table.table(
            cellText=table_data[1:],
            colLabels=table_data[0],
            loc='center',
            cellLoc='center',
            colColours=[COLORS['grey']] * 3,
        )
        table.auto_set_font_size(False)
        table.set_fontsize(self.config['fontsize'] - 1)
        table.scale(1.2, 1.5)

        ax_table.set_title('I. Summary', fontweight='bold', loc='left')

        # Main title
        fig.suptitle(title, fontsize=self.config['fontsize_title'] + 2,
                     fontweight='bold', y=0.98)

        if output:
            self._save_figure(fig, output, ['pdf', 'png', 'svg'])

        return fig

    def plot_stride_analysis(
        self,
        strides: List,
        output: Optional[Union[str, Path]] = None,
        title: str = "Stride Analysis",
    ) -> plt.Figure:
        """
        Plot detailed stride analysis.

        Args:
            strides: List of StrideMetrics objects
            output: Output file path
            title: Figure title
        """
        if not strides:
            warnings.warn("No strides to plot")
            return None

        fig, axes = plt.subplots(2, 2, figsize=self.config['figsize_double'])

        stride_lengths = [s.length for s in strides]
        stride_durations = [s.duration for s in strides]
        stride_velocities = [s.velocity for s in strides]
        stride_numbers = range(1, len(strides) + 1)

        # A: Stride length over time
        ax1 = axes[0, 0]
        ax1.plot(stride_numbers, stride_lengths, 'o-', color=COLORS['blue'],
                 markersize=4, linewidth=1)
        ax1.axhline(np.mean(stride_lengths), color=COLORS['red'], linestyle='--',
                    label=f'Mean: {np.mean(stride_lengths):.2f} cm')
        ax1.set_xlabel('Stride Number')
        ax1.set_ylabel('Stride Length (cm)')
        ax1.set_title('A. Stride Length', fontweight='bold', loc='left')
        ax1.legend(fontsize=self.config['fontsize'] - 1)

        # B: Stride length distribution
        ax2 = axes[0, 1]
        ax2.hist(stride_lengths, bins=15, color=COLORS['cyan'], edgecolor='black',
                 alpha=0.7)
        ax2.axvline(np.mean(stride_lengths), color=COLORS['red'], linestyle='--',
                    linewidth=1.5)
        ax2.set_xlabel('Stride Length (cm)')
        ax2.set_ylabel('Frequency')
        ax2.set_title('B. Distribution', fontweight='bold', loc='left')

        # C: Stride duration
        ax3 = axes[1, 0]
        ax3.plot(stride_numbers, [d * 1000 for d in stride_durations], 'o-',
                 color=COLORS['green'], markersize=4, linewidth=1)
        ax3.axhline(np.mean(stride_durations) * 1000, color=COLORS['red'],
                    linestyle='--', label=f'Mean: {np.mean(stride_durations)*1000:.0f} ms')
        ax3.set_xlabel('Stride Number')
        ax3.set_ylabel('Stride Duration (ms)')
        ax3.set_title('C. Stride Duration', fontweight='bold', loc='left')
        ax3.legend(fontsize=self.config['fontsize'] - 1)

        # D: Length vs Duration correlation
        ax4 = axes[1, 1]
        ax4.scatter(stride_lengths, [d * 1000 for d in stride_durations],
                    c=stride_velocities, cmap='viridis', s=30, alpha=0.7)
        ax4.set_xlabel('Stride Length (cm)')
        ax4.set_ylabel('Stride Duration (ms)')
        ax4.set_title('D. Length vs Duration', fontweight='bold', loc='left')

        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap='viridis',
                                    norm=plt.Normalize(min(stride_velocities),
                                                       max(stride_velocities)))
        cbar = plt.colorbar(sm, ax=ax4)
        cbar.set_label('Velocity (cm/s)', fontsize=self.config['fontsize'] - 1)

        fig.suptitle(title, fontsize=self.config['fontsize_title'], fontweight='bold')
        plt.tight_layout()

        if output:
            self._save_figure(fig, output, ['pdf', 'png'])

        return fig


# =============================================================================
# Convenience Functions
# =============================================================================

def create_publication_figure(
    metrics: Dict,
    output: Union[str, Path],
    journal: str = 'nature',
) -> plt.Figure:
    """
    Quick function to create publication-ready summary figure.

    Args:
        metrics: Gait metrics dictionary
        output: Output file path
        journal: Target journal style
    """
    generator = PublicationFigureGenerator(journal=journal)
    return generator.plot_gait_summary(metrics, output=output)


if __name__ == "__main__":
    print("Publication Figure Generator - Demo")
    print("=" * 50)

    # Demo with synthetic metrics
    demo_metrics = {
        'spatiotemporal': {
            'velocity': {'mean': 12.5, 'sd': 3.2, 'sem': 0.5, 'cv': 25.6, 'max': 28.3, 'min': 1.2},
            'stride_length': {'mean': 4.2, 'sd': 0.8, 'sem': 0.1, 'cv': 19.0},
            'cadence': {'mean': 3.1, 'sd': 0.5},
            'swing_time': {'mean': 0.12, 'percent': 40},
            'stance_time': {'mean': 0.18, 'percent': 60},
            'duty_factor': 0.60,
        },
        'kinematic': {
            'acceleration': {'mean': 45.2, 'sd': 12.3, 'max': 98.5},
            'jerk': {'mean': 320.5, 'sd': 85.2, 'max': 450.0},
        },
        'variability': {
            'regularity_index': 85.3,
            'stride_variability': 19.0,
            'velocity_stability': 74.4,
        },
        'distance_duration': {
            'total_distance_cm': 156.8,
            'total_duration_s': 12.5,
            'active_duration_s': 10.2,
            'rest_duration_s': 2.3,
        },
        'activity': {
            'time_resting_pct': 18.4,
            'time_walking_pct': 45.2,
            'time_trotting_pct': 28.6,
            'time_galloping_pct': 7.8,
        },
        'quality': {
            'tracking_rate': 98.5,
            'data_quality_score': 92.3,
        }
    }

    generator = PublicationFigureGenerator(journal='nature')
    fig = generator.plot_gait_summary(demo_metrics, output='demo_gait_summary.pdf')
    print("\nDemo figure saved: demo_gait_summary.pdf")
    plt.show()
