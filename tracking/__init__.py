"""
Mouse Locomotor Tracker - Tracking Module

This module provides tracking functionality using DeepLabCut for analyzing
mouse locomotion in video recordings.

Components:
    - DeepLabCutAdapter: Wrapper for DeepLabCut API
    - MarkerSet: Configuration for tracking markers
    - TrackProcessor: Post-processing of tracking data
    - VideoMetadata: Video file metadata extraction
    - MockTracker: Synthetic data generation for testing

Author: Stride Labs
License: MIT
"""

from .marker_config import MarkerSet, MOUSE_VENTRAL, MOUSE_LATERAL
from .video_metadata import VideoMetadata
from .track_processor import TrackProcessor
from .mock_tracker import MockTracker

# Conditional import for DeepLabCut adapter
try:
    from .dlc_adapter import DeepLabCutAdapter
    DLC_AVAILABLE = True
except ImportError:
    DeepLabCutAdapter = None
    DLC_AVAILABLE = False

__all__ = [
    # Core classes
    "DeepLabCutAdapter",
    "MarkerSet",
    "TrackProcessor",
    "VideoMetadata",
    "MockTracker",
    # Presets
    "MOUSE_VENTRAL",
    "MOUSE_LATERAL",
    # Availability flag
    "DLC_AVAILABLE",
]

__version__ = "0.1.0"
