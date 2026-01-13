"""
Mock Tracker Module

Provides synthetic tracking data generation for testing and development
without requiring DeepLabCut or GPU resources.

Author: Stride Labs
License: MIT
"""

from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass
import logging

import numpy as np
import pandas as pd

from .marker_config import MarkerSet, MOUSE_VENTRAL, MOUSE_LATERAL

logger = logging.getLogger(__name__)


@dataclass
class MockTrackingConfig:
    """
    Configuration for synthetic tracking data generation.

    Attributes:
        fps: Frame rate (frames per second)
        duration: Duration in seconds
        resolution: Video resolution as (width, height)
        noise_level: Standard deviation of position noise (pixels)
        dropout_rate: Probability of missing detection per frame per marker
        confidence_mean: Mean confidence score
        confidence_std: Standard deviation of confidence scores
        base_speed: Base movement speed in pixels/frame
    """
    fps: float = 30.0
    duration: float = 10.0
    resolution: Tuple[int, int] = (640, 480)
    noise_level: float = 2.0
    dropout_rate: float = 0.05
    confidence_mean: float = 0.85
    confidence_std: float = 0.1
    base_speed: float = 5.0


class MockTracker:
    """
    Mock tracker for generating synthetic pose estimation data.

    Useful for testing analysis pipelines without requiring actual
    DeepLabCut analysis or GPU resources.

    Example:
        >>> from tracking import MockTracker, MOUSE_VENTRAL
        >>> tracker = MockTracker(marker_set=MOUSE_VENTRAL, fps=30, duration=5)
        >>> tracks = tracker.generate_tracks()
        >>> print(tracks.shape)  # (150, 33) for 11 markers * 3 coords
    """

    def __init__(
        self,
        marker_set: Optional[MarkerSet] = None,
        fps: float = 30.0,
        duration: float = 10.0,
        resolution: Tuple[int, int] = (640, 480),
        noise_level: float = 2.0,
        dropout_rate: float = 0.05,
        seed: Optional[int] = None
    ):
        """
        Initialize the mock tracker.

        Args:
            marker_set: MarkerSet defining body parts to track
            fps: Frames per second
            duration: Video duration in seconds
            resolution: Frame resolution as (width, height)
            noise_level: Position noise standard deviation (pixels)
            dropout_rate: Probability of detection dropout
            seed: Random seed for reproducibility
        """
        self.marker_set = marker_set or MOUSE_VENTRAL
        self.config = MockTrackingConfig(
            fps=fps,
            duration=duration,
            resolution=resolution,
            noise_level=noise_level,
            dropout_rate=dropout_rate
        )

        self._rng = np.random.default_rng(seed)
        self._scorer = "MockTracker"

        logger.info(
            f"Initialized MockTracker with {len(self.marker_set)} markers, "
            f"{fps}fps, {duration}s duration"
        )

    def generate_tracks(
        self,
        movement_type: str = 'walking',
        start_position: Optional[Tuple[float, float]] = None,
        heading: float = 0.0
    ) -> pd.DataFrame:
        """
        Generate synthetic tracking data.

        Args:
            movement_type: Type of movement pattern
                - 'walking': Normal walking/locomotion
                - 'stationary': Minimal movement (grooming, resting)
                - 'exploring': Random exploration with direction changes
                - 'circle': Circular movement pattern
            start_position: Initial (x, y) position (center of frame if None)
            heading: Initial heading in radians

        Returns:
            DataFrame with tracking data in DeepLabCut format
            MultiIndex columns: (scorer, bodypart, coord)

        Example:
            >>> tracks = tracker.generate_tracks(movement_type='walking')
        """
        n_frames = int(self.config.fps * self.config.duration)
        width, height = self.config.resolution

        # Default start position
        if start_position is None:
            start_position = (width / 2, height / 2)

        logger.info(
            f"Generating {n_frames} frames of '{movement_type}' movement"
        )

        # Generate center of mass trajectory
        com_trajectory = self._generate_trajectory(
            n_frames=n_frames,
            start_position=start_position,
            initial_heading=heading,
            movement_type=movement_type
        )

        # Generate marker positions relative to COM
        tracks_data = {}

        for marker in self.marker_set.markers:
            x_coords, y_coords, likelihood = self._generate_marker_trajectory(
                com_trajectory=com_trajectory,
                marker_name=marker,
                n_frames=n_frames
            )

            tracks_data[(self._scorer, marker, 'x')] = x_coords
            tracks_data[(self._scorer, marker, 'y')] = y_coords
            tracks_data[(self._scorer, marker, 'likelihood')] = likelihood

        # Create DataFrame with MultiIndex columns
        df = pd.DataFrame(tracks_data)
        df.columns = pd.MultiIndex.from_tuples(
            df.columns,
            names=['scorer', 'bodyparts', 'coords']
        )

        logger.info(f"Generated tracks: {df.shape}")
        return df

    def simulate_gait_cycle(
        self,
        n_cycles: int = 5,
        cycle_duration: float = 0.5,
        stride_length: float = 30.0
    ) -> pd.DataFrame:
        """
        Generate synthetic data with realistic gait cycle patterns.

        Simulates coordinated limb movements with proper phase relationships
        for quadrupedal locomotion.

        Args:
            n_cycles: Number of complete gait cycles
            cycle_duration: Duration of one gait cycle in seconds
            stride_length: Distance traveled per stride in pixels

        Returns:
            DataFrame with gait-patterned tracking data

        Example:
            >>> gait_data = tracker.simulate_gait_cycle(n_cycles=10)
        """
        total_duration = n_cycles * cycle_duration
        n_frames = int(self.config.fps * total_duration)

        logger.info(
            f"Simulating {n_cycles} gait cycles ({cycle_duration}s each)"
        )

        # Time array
        t = np.linspace(0, total_duration, n_frames)

        # Base trajectory - straight line
        speed = stride_length / cycle_duration
        base_x = self.config.resolution[0] / 4 + (speed * t)
        base_y = np.full(n_frames, self.config.resolution[1] / 2)

        # Gait cycle parameters
        # Phase offsets for typical trot gait (diagonal pairs synchronized)
        phase_offsets = {
            'foreL': 0.0,        # Left front
            'foreR': 0.5,        # Right front (opposite phase)
            'hindL': 0.5,        # Left hind (diagonal with foreR)
            'hindR': 0.0,        # Right hind (diagonal with foreL)
        }

        # Stride amplitude for each limb
        stride_amplitude = {
            'foreL': stride_length * 0.4,
            'foreR': stride_length * 0.4,
            'hindL': stride_length * 0.5,
            'hindR': stride_length * 0.5,
        }

        tracks_data = {}
        cycle_freq = 1.0 / cycle_duration

        for marker in self.marker_set.markers:
            # Get base offset for this marker
            offset_x, offset_y = self._get_marker_offset(marker)

            # Calculate position
            x = base_x + offset_x
            y = base_y + offset_y

            # Add gait oscillation for limbs
            if marker in phase_offsets:
                phase = phase_offsets[marker]
                amplitude = stride_amplitude[marker]

                # Sinusoidal stride pattern
                gait_phase = 2 * np.pi * cycle_freq * t + phase * 2 * np.pi

                # X oscillation (forward-backward during stride)
                x += amplitude * 0.5 * np.cos(gait_phase)

                # Y oscillation (lateral sway)
                y += amplitude * 0.1 * np.sin(gait_phase)

            # Add noise
            x += self._rng.normal(0, self.config.noise_level, n_frames)
            y += self._rng.normal(0, self.config.noise_level, n_frames)

            # Generate likelihood with occasional dropouts
            likelihood = self._generate_likelihood(n_frames)

            # Apply dropouts
            dropout_mask = self._rng.random(n_frames) < self.config.dropout_rate
            x[dropout_mask] = np.nan
            y[dropout_mask] = np.nan
            likelihood[dropout_mask] = self._rng.uniform(0.1, 0.4, dropout_mask.sum())

            tracks_data[(self._scorer, marker, 'x')] = x
            tracks_data[(self._scorer, marker, 'y')] = y
            tracks_data[(self._scorer, marker, 'likelihood')] = likelihood

        df = pd.DataFrame(tracks_data)
        df.columns = pd.MultiIndex.from_tuples(
            df.columns,
            names=['scorer', 'bodyparts', 'coords']
        )

        return df

    def generate_with_artifacts(
        self,
        n_frames: int = 300,
        occlusion_frames: Optional[List[Tuple[int, int]]] = None,
        jump_frames: Optional[List[int]] = None,
        low_light_frames: Optional[Tuple[int, int]] = None
    ) -> pd.DataFrame:
        """
        Generate tracking data with common artifacts for testing robustness.

        Args:
            n_frames: Total number of frames
            occlusion_frames: List of (start, end) frame ranges for occlusions
            jump_frames: Frame indices where tracking jumps occur
            low_light_frames: (start, end) range with reduced confidence

        Returns:
            DataFrame with artifacts included

        Example:
            >>> # Generate data with occlusion from frames 50-100
            >>> data = tracker.generate_with_artifacts(
            ...     occlusion_frames=[(50, 100)],
            ...     jump_frames=[150, 200]
            ... )
        """
        # Temporarily adjust duration
        original_duration = self.config.duration
        self.config.duration = n_frames / self.config.fps

        # Generate base tracks
        df = self.generate_tracks()

        self.config.duration = original_duration

        # Apply occlusions (set all markers to NaN)
        if occlusion_frames:
            for start, end in occlusion_frames:
                for marker in self.marker_set.markers:
                    x_col = (self._scorer, marker, 'x')
                    y_col = (self._scorer, marker, 'y')
                    lik_col = (self._scorer, marker, 'likelihood')

                    df.loc[start:end, x_col] = np.nan
                    df.loc[start:end, y_col] = np.nan
                    df.loc[start:end, lik_col] = self._rng.uniform(0, 0.3, end - start + 1)

                logger.debug(f"Added occlusion at frames {start}-{end}")

        # Apply tracking jumps (sudden position changes)
        if jump_frames:
            for frame_idx in jump_frames:
                if 0 <= frame_idx < len(df):
                    jump_x = self._rng.normal(0, 50)
                    jump_y = self._rng.normal(0, 50)

                    for marker in self.marker_set.markers:
                        x_col = (self._scorer, marker, 'x')
                        y_col = (self._scorer, marker, 'y')

                        df.loc[frame_idx, x_col] += jump_x
                        df.loc[frame_idx, y_col] += jump_y

                    logger.debug(f"Added tracking jump at frame {frame_idx}")

        # Apply low light (reduced confidence)
        if low_light_frames:
            start, end = low_light_frames
            for marker in self.marker_set.markers:
                lik_col = (self._scorer, marker, 'likelihood')
                df.loc[start:end, lik_col] *= 0.5  # Reduce confidence by half

            logger.debug(f"Added low light condition at frames {start}-{end}")

        return df

    # =========================================================================
    # Private helper methods
    # =========================================================================

    def _generate_trajectory(
        self,
        n_frames: int,
        start_position: Tuple[float, float],
        initial_heading: float,
        movement_type: str
    ) -> np.ndarray:
        """Generate center of mass trajectory."""
        width, height = self.config.resolution
        x, y = start_position
        heading = initial_heading

        trajectory = np.zeros((n_frames, 2))
        trajectory[0] = [x, y]

        if movement_type == 'stationary':
            # Minimal movement
            for i in range(1, n_frames):
                dx = self._rng.normal(0, 0.5)
                dy = self._rng.normal(0, 0.5)
                x = np.clip(x + dx, 50, width - 50)
                y = np.clip(y + dy, 50, height - 50)
                trajectory[i] = [x, y]

        elif movement_type == 'walking':
            # Directed walking with slight variations
            for i in range(1, n_frames):
                heading += self._rng.normal(0, 0.05)
                speed = self.config.base_speed + self._rng.normal(0, 1)
                dx = speed * np.cos(heading)
                dy = speed * np.sin(heading)
                x = np.clip(x + dx, 50, width - 50)
                y = np.clip(y + dy, 50, height - 50)
                trajectory[i] = [x, y]

                # Bounce off walls
                if x <= 50 or x >= width - 50:
                    heading = np.pi - heading
                if y <= 50 or y >= height - 50:
                    heading = -heading

        elif movement_type == 'exploring':
            # Random exploration with direction changes
            for i in range(1, n_frames):
                if self._rng.random() < 0.02:  # 2% chance of direction change
                    heading = self._rng.uniform(-np.pi, np.pi)

                heading += self._rng.normal(0, 0.1)
                speed = self.config.base_speed * self._rng.uniform(0.5, 1.5)
                dx = speed * np.cos(heading)
                dy = speed * np.sin(heading)
                x = np.clip(x + dx, 50, width - 50)
                y = np.clip(y + dy, 50, height - 50)
                trajectory[i] = [x, y]

        elif movement_type == 'circle':
            # Circular movement
            radius = min(width, height) / 4
            center_x, center_y = width / 2, height / 2
            angular_speed = 2 * np.pi / (self.config.fps * 5)  # 5 second circle

            for i in range(n_frames):
                angle = initial_heading + i * angular_speed
                trajectory[i, 0] = center_x + radius * np.cos(angle)
                trajectory[i, 1] = center_y + radius * np.sin(angle)

        else:
            raise ValueError(f"Unknown movement type: {movement_type}")

        return trajectory

    def _generate_marker_trajectory(
        self,
        com_trajectory: np.ndarray,
        marker_name: str,
        n_frames: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate trajectory for a single marker."""
        # Get marker offset from body center
        offset_x, offset_y = self._get_marker_offset(marker_name)

        # Calculate heading from trajectory
        dx = np.diff(com_trajectory[:, 0], prepend=com_trajectory[0, 0])
        dy = np.diff(com_trajectory[:, 1], prepend=com_trajectory[0, 1])
        headings = np.arctan2(dy, dx)

        # Smooth heading
        from scipy.ndimage import gaussian_filter1d
        headings = gaussian_filter1d(headings, sigma=3)

        # Rotate offset based on heading
        cos_h = np.cos(headings)
        sin_h = np.sin(headings)

        rotated_offset_x = offset_x * cos_h - offset_y * sin_h
        rotated_offset_y = offset_x * sin_h + offset_y * cos_h

        # Calculate positions
        x = com_trajectory[:, 0] + rotated_offset_x
        y = com_trajectory[:, 1] + rotated_offset_y

        # Add noise
        x += self._rng.normal(0, self.config.noise_level, n_frames)
        y += self._rng.normal(0, self.config.noise_level, n_frames)

        # Generate likelihood
        likelihood = self._generate_likelihood(n_frames)

        # Apply random dropouts
        dropout_mask = self._rng.random(n_frames) < self.config.dropout_rate
        x[dropout_mask] = np.nan
        y[dropout_mask] = np.nan
        likelihood[dropout_mask] = self._rng.uniform(0.1, 0.4, dropout_mask.sum())

        return x, y, likelihood

    def _get_marker_offset(self, marker_name: str) -> Tuple[float, float]:
        """Get (x, y) offset for a marker relative to body center."""
        # Offsets in pixels, assuming body length ~60px
        offsets = {
            # Ventral view markers
            'snout': (35, 0),
            'snoutL': (32, -8),
            'snoutR': (32, 8),
            'foreL': (15, -20),
            'foreR': (15, 20),
            'hindL': (-20, -25),
            'hindR': (-20, 25),
            'torso': (0, 0),
            'torsoL': (0, -15),
            'torsoR': (0, 15),
            'tail': (-35, 0),

            # Lateral view markers
            'crest': (10, 0),
            'hip': (0, 0),
            'knee': (-5, 15),
            'ankle': (-8, 25),
            'foot': (-5, 30),
            'toe': (0, 32),
        }

        return offsets.get(marker_name, (0, 0))

    def _generate_likelihood(self, n_frames: int) -> np.ndarray:
        """Generate realistic likelihood values."""
        likelihood = self._rng.normal(
            self.config.confidence_mean,
            self.config.confidence_std,
            n_frames
        )
        # Clip to valid range
        return np.clip(likelihood, 0.0, 1.0)


def create_test_dataset(
    n_videos: int = 5,
    frames_per_video: int = 300,
    seed: int = 42
) -> Dict[str, pd.DataFrame]:
    """
    Create a dataset of mock tracking results for testing.

    Args:
        n_videos: Number of mock video results to generate
        frames_per_video: Frames per video
        seed: Random seed

    Returns:
        Dictionary mapping video names to tracking DataFrames

    Example:
        >>> dataset = create_test_dataset(n_videos=3)
        >>> for name, tracks in dataset.items():
        ...     print(f"{name}: {tracks.shape}")
    """
    tracker = MockTracker(seed=seed)
    dataset = {}

    movement_types = ['walking', 'exploring', 'stationary', 'circle']

    for i in range(n_videos):
        video_name = f"mock_video_{i:03d}"
        movement = movement_types[i % len(movement_types)]

        tracker.config.duration = frames_per_video / tracker.config.fps
        tracks = tracker.generate_tracks(movement_type=movement)
        dataset[video_name] = tracks

    logger.info(f"Created test dataset with {n_videos} videos")
    return dataset
