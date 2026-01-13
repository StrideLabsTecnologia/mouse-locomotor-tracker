"""
Pytest Fixtures for Mouse Locomotor Tracker Tests

This module provides reusable fixtures for testing the locomotor tracking system.
All fixtures generate deterministic synthetic data to ensure reproducible tests.

Fixtures:
    - sample_tracks: DataFrame with synthetic tracking data
    - sample_video_metadata: Dictionary with video metadata
    - sample_markers: MarkerSet configuration for testing
    - sample_velocity_data: Pre-computed velocity data
    - sample_coordination_data: Limb coordination data
    - sample_gait_cycles: Gait cycle detection data

Author: Stride Labs
License: MIT
"""

import pytest
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import tempfile
import os


# =============================================================================
# Data Classes for Test Data
# =============================================================================

@dataclass
class MarkerSet:
    """Configuration for tracking markers."""
    name: str
    markers: List[str]
    limb_pairs: Dict[str, Tuple[str, str]]
    speed_markers: List[str]

    def get_all_markers(self) -> List[str]:
        """Return all marker names."""
        return self.markers.copy()

    def get_limb_pair(self, pair_name: str) -> Tuple[str, str]:
        """Return a specific limb pair."""
        return self.limb_pairs.get(pair_name)


@dataclass
class VideoMetadata:
    """Video file metadata."""
    duration: float  # seconds
    fps: int
    n_frames: int
    width: int
    height: int
    pixel_width_mm: float  # mm per pixel

    @classmethod
    def from_dict(cls, d: dict) -> "VideoMetadata":
        return cls(
            duration=d["dur"],
            fps=d["fps"],
            n_frames=d["nFrame"],
            width=d["imW"],
            height=d["imH"],
            pixel_width_mm=d["xPixW"]
        )

    def to_dict(self) -> dict:
        return {
            "dur": self.duration,
            "fps": self.fps,
            "nFrame": self.n_frames,
            "imW": self.width,
            "imH": self.height,
            "xPixW": self.pixel_width_mm
        }


# =============================================================================
# Core Fixtures
# =============================================================================

@pytest.fixture
def random_seed():
    """Set random seed for reproducibility."""
    np.random.seed(42)
    return 42


@pytest.fixture
def sample_markers() -> MarkerSet:
    """
    Create a MarkerSet configuration for mouse ventral view tracking.

    Returns:
        MarkerSet with standard mouse locomotion markers
    """
    markers = [
        "snout", "snoutL", "snoutR",
        "foreL", "foreR",
        "hindL", "hindR",
        "torso", "torsoL", "torsoR",
        "tail"
    ]

    limb_pairs = {
        "LH_RH": ("hindL", "hindR"),      # Left-Right hind limbs
        "LH_LF": ("hindL", "foreL"),       # Left hind-fore
        "RH_RF": ("hindR", "foreR"),       # Right hind-fore
        "LF_RH": ("foreL", "hindR"),       # Diagonal left fore - right hind
        "RF_LH": ("foreR", "hindL"),       # Diagonal right fore - left hind
        "LF_RF": ("foreL", "foreR"),       # Left-Right fore limbs
    }

    speed_markers = [
        "tail", "snout", "snoutL", "snoutR",
        "torso", "torsoL", "torsoR"
    ]

    return MarkerSet(
        name="mouse_ventral",
        markers=markers,
        limb_pairs=limb_pairs,
        speed_markers=speed_markers
    )


@pytest.fixture
def sample_video_metadata() -> dict:
    """
    Create sample video metadata dictionary.

    Returns:
        Dictionary with video properties matching reference format
    """
    width = 640
    height = 480
    length_mm = 200  # mm across image width

    return {
        "dur": 30.0,           # 30 seconds duration
        "fps": 100,            # 100 frames per second
        "nFrame": 3000,        # Total frames
        "imW": width,          # Image width in pixels
        "imH": height,         # Image height in pixels
        "xPixW": length_mm / width  # mm per pixel
    }


@pytest.fixture
def sample_tracks(sample_markers, sample_video_metadata, random_seed) -> pd.DataFrame:
    """
    Generate synthetic tracking data as a pandas DataFrame.

    Simulates realistic mouse locomotion with:
    - Sinusoidal limb movements
    - Progressive body position
    - Realistic noise levels

    Args:
        sample_markers: MarkerSet fixture
        sample_video_metadata: Video metadata fixture
        random_seed: Random seed for reproducibility

    Returns:
        MultiIndex DataFrame matching DeepLabCut output format
    """
    n_frames = sample_video_metadata["nFrame"]
    fps = sample_video_metadata["fps"]
    width = sample_video_metadata["imW"]
    height = sample_video_metadata["imH"]

    # Time array
    t = np.linspace(0, sample_video_metadata["dur"], n_frames)

    # Base body position - moves across the frame
    base_speed = 15.0  # cm/s
    pixel_per_cm = width / 20.0  # 20 cm across frame
    body_x = 100 + base_speed * pixel_per_cm * t / sample_video_metadata["dur"] * 0.5
    body_y = height / 2 + 10 * np.sin(2 * np.pi * 0.5 * t)  # Slight lateral sway

    # Gait frequency ~4 Hz for trotting mouse
    gait_freq = 4.0
    stride_amplitude = 30  # pixels

    # Create multi-index columns
    model_name = "DLC_mouse_model"
    columns = pd.MultiIndex.from_tuples(
        [(model_name, marker, coord)
         for marker in sample_markers.markers
         for coord in ["x", "y", "likelihood"]],
        names=["scorer", "bodyparts", "coords"]
    )

    # Initialize data array
    data = np.zeros((n_frames, len(columns)))

    # Generate marker positions
    for i, marker in enumerate(sample_markers.markers):
        base_idx = i * 3  # x, y, likelihood columns

        # Marker-specific offsets and movements
        if marker == "snout":
            offset_x, offset_y = 50, 0
            phase = 0
        elif marker == "snoutL":
            offset_x, offset_y = 45, -10
            phase = 0
        elif marker == "snoutR":
            offset_x, offset_y = 45, 10
            phase = 0
        elif marker == "foreL":
            offset_x, offset_y = 25, -15
            phase = 0  # In phase with right hind (diagonal)
        elif marker == "foreR":
            offset_x, offset_y = 25, 15
            phase = np.pi  # Out of phase with foreL
        elif marker == "hindL":
            offset_x, offset_y = -25, -15
            phase = np.pi  # Out of phase with foreL (diagonal coordination)
        elif marker == "hindR":
            offset_x, offset_y = -25, 15
            phase = 0  # In phase with foreL
        elif marker == "torso":
            offset_x, offset_y = 0, 0
            phase = 0
        elif marker == "torsoL":
            offset_x, offset_y = 0, -12
            phase = 0
        elif marker == "torsoR":
            offset_x, offset_y = 0, 12
            phase = 0
        elif marker == "tail":
            offset_x, offset_y = -60, 0
            phase = np.pi / 4  # Slight phase lag
        else:
            offset_x, offset_y = 0, 0
            phase = 0

        # Limb markers have sinusoidal movement
        if marker in ["foreL", "foreR", "hindL", "hindR"]:
            limb_movement = stride_amplitude * np.sin(2 * np.pi * gait_freq * t + phase)
        else:
            limb_movement = 0

        # Add noise
        noise_x = np.random.normal(0, 1.5, n_frames)
        noise_y = np.random.normal(0, 1.5, n_frames)

        # X position
        data[:, base_idx] = body_x + offset_x + limb_movement + noise_x
        # Y position
        data[:, base_idx + 1] = body_y + offset_y + noise_y
        # Likelihood (high confidence with occasional drops)
        likelihood = np.ones(n_frames) * 0.95 + np.random.normal(0, 0.02, n_frames)
        likelihood = np.clip(likelihood, 0.5, 1.0)
        data[:, base_idx + 2] = likelihood

    return pd.DataFrame(data, columns=columns)


@pytest.fixture
def sample_tracks_constant_velocity(sample_markers, sample_video_metadata) -> pd.DataFrame:
    """
    Generate tracks with perfectly constant velocity for velocity tests.

    Returns:
        DataFrame with linear body movement (constant velocity)
    """
    n_frames = sample_video_metadata["nFrame"]
    fps = sample_video_metadata["fps"]
    width = sample_video_metadata["imW"]
    height = sample_video_metadata["imH"]

    # Constant velocity: 10 cm/s
    velocity_cm_s = 10.0
    pixel_per_mm = width / 200.0  # 200 mm across frame
    pixel_per_cm = pixel_per_mm * 10

    t = np.linspace(0, sample_video_metadata["dur"], n_frames)

    # Linear position
    body_x = 100 + velocity_cm_s * pixel_per_cm * t
    body_y = np.full(n_frames, height / 2)

    model_name = "DLC_mouse_model"
    columns = pd.MultiIndex.from_tuples(
        [(model_name, marker, coord)
         for marker in sample_markers.markers
         for coord in ["x", "y", "likelihood"]],
        names=["scorer", "bodyparts", "coords"]
    )

    data = np.zeros((n_frames, len(columns)))

    for i, marker in enumerate(sample_markers.markers):
        base_idx = i * 3

        if marker == "snout":
            offset_x = 50
        elif marker == "tail":
            offset_x = -60
        elif marker == "torso":
            offset_x = 0
        else:
            offset_x = np.random.uniform(-30, 30)

        data[:, base_idx] = body_x + offset_x
        data[:, base_idx + 1] = body_y + np.random.uniform(-20, 20)
        data[:, base_idx + 2] = 0.98  # High likelihood

    return pd.DataFrame(data, columns=columns)


@pytest.fixture
def sample_tracks_with_acceleration(sample_markers, sample_video_metadata) -> pd.DataFrame:
    """
    Generate tracks with varying acceleration for velocity tests.

    Pattern:
        - First third: acceleration (0 to 20 cm/s)
        - Middle third: constant velocity (20 cm/s)
        - Last third: deceleration (20 to 0 cm/s)

    Returns:
        DataFrame with accelerating/decelerating body movement
    """
    n_frames = sample_video_metadata["nFrame"]
    fps = sample_video_metadata["fps"]
    width = sample_video_metadata["imW"]
    height = sample_video_metadata["imH"]

    t = np.linspace(0, sample_video_metadata["dur"], n_frames)
    dur = sample_video_metadata["dur"]

    pixel_per_mm = width / 200.0
    pixel_per_cm = pixel_per_mm * 10

    # Create velocity profile
    velocity = np.zeros(n_frames)
    third = n_frames // 3

    # Acceleration phase: 0 to 20 cm/s
    velocity[:third] = np.linspace(0, 20, third)
    # Constant phase: 20 cm/s
    velocity[third:2*third] = 20
    # Deceleration phase: 20 to 0 cm/s
    velocity[2*third:] = np.linspace(20, 0, n_frames - 2*third)

    # Integrate velocity to get position
    dt = dur / n_frames
    position = np.cumsum(velocity) * dt * pixel_per_cm
    body_x = 100 + position
    body_y = np.full(n_frames, height / 2)

    model_name = "DLC_mouse_model"
    columns = pd.MultiIndex.from_tuples(
        [(model_name, marker, coord)
         for marker in sample_markers.markers
         for coord in ["x", "y", "likelihood"]],
        names=["scorer", "bodyparts", "coords"]
    )

    data = np.zeros((n_frames, len(columns)))

    for i, marker in enumerate(sample_markers.markers):
        base_idx = i * 3
        offset_x = np.random.uniform(-30, 30) if marker not in ["snout", "tail", "torso"] else 0

        data[:, base_idx] = body_x + offset_x
        data[:, base_idx + 1] = body_y
        data[:, base_idx + 2] = 0.98

    return pd.DataFrame(data, columns=columns)


@pytest.fixture
def sample_synchronized_limbs(sample_video_metadata) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate perfectly synchronized limb stride data.

    Returns:
        Tuple of (stride_0, stride_1) with zero phase difference (R ~ 1)
    """
    n_frames = sample_video_metadata["nFrame"]
    t = np.linspace(0, sample_video_metadata["dur"], n_frames)

    freq = 4.0  # Hz
    amplitude = 10.0  # mm

    # Both limbs in perfect sync
    stride_0 = amplitude * np.sin(2 * np.pi * freq * t)
    stride_1 = amplitude * np.sin(2 * np.pi * freq * t)

    return stride_0, stride_1


@pytest.fixture
def sample_antiphase_limbs(sample_video_metadata) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate anti-phase limb stride data (typical diagonal coordination).

    Returns:
        Tuple of (stride_0, stride_1) with 180 degree phase difference
    """
    n_frames = sample_video_metadata["nFrame"]
    t = np.linspace(0, sample_video_metadata["dur"], n_frames)

    freq = 4.0  # Hz
    amplitude = 10.0  # mm

    stride_0 = amplitude * np.sin(2 * np.pi * freq * t)
    stride_1 = amplitude * np.sin(2 * np.pi * freq * t + np.pi)  # 180 degree phase

    return stride_0, stride_1


@pytest.fixture
def sample_random_limbs(sample_video_metadata, random_seed) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate random (uncorrelated) limb stride data.

    Returns:
        Tuple of (stride_0, stride_1) with random phase relationship (R ~ 0)
    """
    n_frames = sample_video_metadata["nFrame"]

    # Random noise - no coordination
    stride_0 = np.random.normal(0, 5, n_frames)
    stride_1 = np.random.normal(0, 5, n_frames)

    return stride_0, stride_1


@pytest.fixture
def sample_gait_cycles_data(sample_video_metadata) -> np.ndarray:
    """
    Generate stride data with clear gait cycles.

    Returns:
        Array with distinct peaks for cycle detection
    """
    n_frames = sample_video_metadata["nFrame"]
    t = np.linspace(0, sample_video_metadata["dur"], n_frames)

    freq = 4.0  # 4 Hz = 4 steps per second
    amplitude = 15.0

    # Clean sinusoidal stride pattern
    stride = amplitude * np.sin(2 * np.pi * freq * t)

    # Add small noise
    stride += np.random.normal(0, 0.5, n_frames)

    return stride


@pytest.fixture
def temp_output_dir():
    """
    Create a temporary directory for test outputs.

    Yields:
        Path to temporary directory (cleaned up after test)
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def mock_video_path(temp_output_dir) -> str:
    """
    Create a mock video file path for testing.

    Returns:
        Path to (non-existent) mock video file
    """
    return os.path.join(temp_output_dir, "test_video.avi")


# =============================================================================
# Analysis Module Fixtures
# =============================================================================

@pytest.fixture
def velocity_analyzer_config() -> dict:
    """Configuration for VelocityAnalyzer."""
    return {
        "smoothing_factor": 10,
        "speed_threshold": 5.0,  # cm/s minimum for movement
        "acceleration_smoothing": 12,
        "drag_threshold": 0.25,  # seconds
    }


@pytest.fixture
def coordination_analyzer_config() -> dict:
    """Configuration for CircularCoordinationAnalyzer."""
    return {
        "interpolation_factor": 4,
        "smoothing_factor": 10,
    }


@pytest.fixture
def gait_cycle_config() -> dict:
    """Configuration for GaitCycleDetector."""
    return {
        "min_peak_distance": None,  # Auto-calculated
        "interpolation_factor": 4,
    }


# =============================================================================
# Helper Functions for Tests
# =============================================================================

def create_synthetic_speed_data(
    duration: float,
    fps: int,
    pattern: str = "constant",
    base_speed: float = 10.0
) -> np.ndarray:
    """
    Create synthetic speed data with various patterns.

    Args:
        duration: Video duration in seconds
        fps: Frames per second
        pattern: "constant", "accelerating", "decelerating", "sinusoidal"
        base_speed: Base speed in cm/s

    Returns:
        1D array of speed values
    """
    n_frames = int(duration * fps)
    t = np.linspace(0, duration, n_frames)

    if pattern == "constant":
        speed = np.full(n_frames, base_speed)
    elif pattern == "accelerating":
        speed = np.linspace(0, base_speed * 2, n_frames)
    elif pattern == "decelerating":
        speed = np.linspace(base_speed * 2, 0, n_frames)
    elif pattern == "sinusoidal":
        speed = base_speed * (1 + 0.5 * np.sin(2 * np.pi * 0.5 * t))
    else:
        raise ValueError(f"Unknown pattern: {pattern}")

    return speed


def compute_expected_circular_mean(phi: np.ndarray) -> Tuple[float, float]:
    """
    Compute expected circular mean for verification.

    Args:
        phi: Array of phase angles in radians

    Returns:
        Tuple of (mean_angle, resultant_length)
    """
    X = np.cos(phi).mean()
    Y = np.sin(phi).mean()
    R = np.sqrt(X**2 + Y**2)
    mean_phi = np.arctan2(Y, X)
    return mean_phi, R


def count_expected_cycles(stride: np.ndarray, fps: int) -> int:
    """
    Count expected gait cycles using peak detection.

    Args:
        stride: Stride position array
        fps: Frames per second

    Returns:
        Number of detected cycles
    """
    from scipy.signal import find_peaks

    peaks, _ = find_peaks(stride)
    if len(peaks) < 2:
        return 0

    # Refine with distance threshold
    mean_distance = np.diff(peaks).mean() / 2
    peaks, _ = find_peaks(stride, distance=mean_distance)

    return len(peaks)
