"""
Mouse Locomotor Tracker - Visualization Module
===============================================

Professional motion capture-style visualization for mouse locomotion analysis.
Provides trajectory overlays, stick figures, coordination plots, and real-time dashboards.

Author: Stride Labs
Version: 1.0.0
"""

from .trajectory_overlay import TrajectoryVisualizer
from .stick_figures import StickFigureGenerator
from .circular_plots import CoordinationPlotter
from .speed_plots import SpeedProfilePlotter
from .dashboard import RealtimeDashboard
from .video_generator import VideoGenerator

__all__ = [
    "TrajectoryVisualizer",
    "StickFigureGenerator",
    "CoordinationPlotter",
    "SpeedProfilePlotter",
    "RealtimeDashboard",
    "VideoGenerator",
]

__version__ = "1.0.0"
__author__ = "Stride Labs"
