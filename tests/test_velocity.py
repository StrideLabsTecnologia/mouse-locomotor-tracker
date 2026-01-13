"""
Tests for VelocityAnalyzer Module

This module tests the velocity analysis functionality including:
- Speed calculation from position data
- Acceleration computation
- Smoothing filters
- Drag event detection

Test Coverage Target: 90%+

Author: Stride Labs
License: MIT
"""

import pytest
import numpy as np
import pandas as pd
from typing import Tuple
from scipy.signal import savgol_filter


# =============================================================================
# VelocityAnalyzer Implementation (for testing)
# =============================================================================

class VelocityAnalyzer:
    """
    Analyzes velocity and acceleration from tracking data.

    This class computes instantaneous speed, acceleration, and detects
    drag/recovery events during mouse locomotion.

    Attributes:
        smoothing_factor: Window size for speed smoothing
        accel_smoothing_factor: Window size for acceleration smoothing
        speed_threshold: Minimum speed to consider as movement (cm/s)
        drag_threshold: Minimum duration for drag events (seconds)
    """

    def __init__(
        self,
        smoothing_factor: int = 10,
        accel_smoothing_factor: int = 12,
        speed_threshold: float = 5.0,
        drag_threshold: float = 0.25
    ):
        self.smoothing_factor = smoothing_factor
        self.accel_smoothing_factor = accel_smoothing_factor
        self.speed_threshold = speed_threshold
        self.drag_threshold = drag_threshold

    def compute_speed(
        self,
        positions: np.ndarray,
        fps: int,
        pixel_to_mm: float
    ) -> np.ndarray:
        """
        Compute instantaneous speed from position data.

        Args:
            positions: 2D array of (x, y) positions, shape (n_frames, 2)
            fps: Frames per second
            pixel_to_mm: Conversion factor from pixels to millimeters

        Returns:
            1D array of speed values in cm/s
        """
        if len(positions) < 2:
            return np.array([0.0])

        # Compute displacements
        dx = np.diff(positions[:, 0])
        dy = np.diff(positions[:, 1])

        # Euclidean distance per frame
        dist = np.sqrt(dx**2 + dy**2)

        # Convert to cm/s
        speed = dist * pixel_to_mm / 10.0 * fps  # mm to cm

        # Apply smoothing
        if len(speed) > self.smoothing_factor:
            kernel = np.ones(self.smoothing_factor) / self.smoothing_factor
            speed = np.convolve(speed, kernel, mode='valid')

        return speed

    def compute_speed_from_markers(
        self,
        tracks: pd.DataFrame,
        model_name: str,
        speed_markers: list,
        fps: int,
        pixel_to_mm: float
    ) -> np.ndarray:
        """
        Compute speed using multiple body markers.

        Args:
            tracks: DeepLabCut-format DataFrame with tracking data
            model_name: Name of the DLC model
            speed_markers: List of marker names to average
            fps: Frames per second
            pixel_to_mm: Conversion factor

        Returns:
            1D array of smoothed speed values in cm/s
        """
        # Extract x positions for all speed markers
        x_positions = []
        for marker in speed_markers:
            try:
                x = tracks[model_name][marker]['x'].values
                x_positions.append(x)
            except KeyError:
                continue

        if not x_positions:
            return np.array([0.0])

        # Average body position
        body_x = np.mean(x_positions, axis=0)

        # Compute speed from position changes
        dx = np.diff(body_x)
        speed = np.abs(dx) * pixel_to_mm / 10.0 * fps  # mm to cm

        # Apply smoothing
        if len(speed) > self.smoothing_factor:
            kernel = np.ones(self.smoothing_factor) / self.smoothing_factor
            speed = np.convolve(speed, kernel, mode='valid')

        return speed

    def compute_acceleration(
        self,
        speed: np.ndarray,
        fps: int
    ) -> np.ndarray:
        """
        Compute acceleration from speed data.

        Args:
            speed: 1D array of speed values in cm/s
            fps: Frames per second

        Returns:
            1D array of acceleration values in cm/s^2
        """
        if len(speed) < 2:
            return np.array([0.0])

        # Compute acceleration
        accel = np.diff(speed) * fps  # change in speed per second

        # Apply smoothing
        if len(accel) > self.accel_smoothing_factor:
            kernel = np.ones(self.accel_smoothing_factor) / self.accel_smoothing_factor
            accel = np.convolve(accel, kernel, mode='valid')

        return accel

    def detect_drag_events(
        self,
        acceleration: np.ndarray,
        fps: int
    ) -> Tuple[np.ndarray, np.ndarray, dict]:
        """
        Detect drag and recovery events from acceleration data.

        Drag events: sustained negative acceleration
        Recovery events: sustained positive acceleration

        Args:
            acceleration: 1D array of acceleration values
            fps: Frames per second

        Returns:
            Tuple of (drag_indices, recovery_indices, statistics)
        """
        min_frames = int(self.drag_threshold * fps)

        # Find negative acceleration segments (drag)
        is_negative = acceleration < 0
        drag_segments = self._find_segments(is_negative, min_frames)

        # Find positive acceleration segments (recovery)
        is_positive = acceleration > 0
        recovery_segments = self._find_segments(is_positive, min_frames)

        # Compute statistics
        stats = {
            'drag_count': len(drag_segments),
            'recovery_count': len(recovery_segments),
            'drag_duration': sum(e - s for s, e in drag_segments) / fps,
            'recovery_duration': sum(e - s for s, e in recovery_segments) / fps,
            'peak_acceleration': acceleration.max() if len(acceleration) > 0 else 0,
            'min_acceleration': acceleration.min() if len(acceleration) > 0 else 0,
        }

        return (
            np.array(drag_segments) if drag_segments else np.array([]).reshape(0, 2),
            np.array(recovery_segments) if recovery_segments else np.array([]).reshape(0, 2),
            stats
        )

    def _find_segments(
        self,
        mask: np.ndarray,
        min_length: int
    ) -> list:
        """Find continuous segments in a boolean mask."""
        segments = []
        start = None

        for i, val in enumerate(mask):
            if val and start is None:
                start = i
            elif not val and start is not None:
                if i - start >= min_length:
                    segments.append((start, i))
                start = None

        # Check final segment
        if start is not None and len(mask) - start >= min_length:
            segments.append((start, len(mask)))

        return segments

    def apply_smoothing(
        self,
        data: np.ndarray,
        method: str = 'moving_average',
        window_size: int = None
    ) -> np.ndarray:
        """
        Apply smoothing filter to data.

        Args:
            data: Input data array
            method: 'moving_average' or 'savgol'
            window_size: Window size (default uses smoothing_factor)

        Returns:
            Smoothed data array
        """
        if window_size is None:
            window_size = self.smoothing_factor

        if len(data) <= window_size:
            return data.copy()

        if method == 'moving_average':
            kernel = np.ones(window_size) / window_size
            return np.convolve(data, kernel, mode='valid')
        elif method == 'savgol':
            # Ensure odd window size for Savitzky-Golay
            if window_size % 2 == 0:
                window_size += 1
            return savgol_filter(data, window_size, polyorder=3)
        else:
            raise ValueError(f"Unknown smoothing method: {method}")


# =============================================================================
# Test Classes
# =============================================================================

class TestVelocityConstantMotion:
    """Tests for constant velocity scenarios."""

    def test_velocity_constant_motion_basic(
        self,
        sample_video_metadata,
        velocity_analyzer_config
    ):
        """Test velocity calculation with constant motion."""
        analyzer = VelocityAnalyzer(**velocity_analyzer_config)

        # Create constant velocity data: 10 cm/s
        n_frames = 1000
        fps = sample_video_metadata["fps"]
        pixel_to_mm = sample_video_metadata["xPixW"]

        # Position moving at constant velocity
        expected_velocity = 10.0  # cm/s
        velocity_pixel_per_frame = expected_velocity * 10 / pixel_to_mm / fps  # convert back

        positions = np.zeros((n_frames, 2))
        positions[:, 0] = np.arange(n_frames) * velocity_pixel_per_frame
        positions[:, 1] = 100  # constant y

        speed = analyzer.compute_speed(positions, fps, pixel_to_mm)

        # Speed should be approximately constant
        assert len(speed) > 0
        mean_speed = np.mean(speed)
        std_speed = np.std(speed)

        # Allow 10% tolerance for smoothing effects
        assert abs(mean_speed - expected_velocity) < expected_velocity * 0.1
        # Standard deviation should be very low for constant motion
        assert std_speed < expected_velocity * 0.05

    def test_velocity_constant_motion_zero(
        self,
        sample_video_metadata,
        velocity_analyzer_config
    ):
        """Test velocity calculation when stationary."""
        analyzer = VelocityAnalyzer(**velocity_analyzer_config)

        n_frames = 500
        fps = sample_video_metadata["fps"]
        pixel_to_mm = sample_video_metadata["xPixW"]

        # Stationary positions
        positions = np.zeros((n_frames, 2))
        positions[:, 0] = 100
        positions[:, 1] = 100

        speed = analyzer.compute_speed(positions, fps, pixel_to_mm)

        # Speed should be zero
        assert np.allclose(speed, 0, atol=0.01)

    def test_velocity_constant_motion_diagonal(
        self,
        sample_video_metadata,
        velocity_analyzer_config
    ):
        """Test velocity calculation with diagonal constant motion."""
        analyzer = VelocityAnalyzer(**velocity_analyzer_config)

        n_frames = 1000
        fps = sample_video_metadata["fps"]
        pixel_to_mm = sample_video_metadata["xPixW"]

        # Diagonal motion at 45 degrees
        expected_velocity = 10.0  # cm/s total
        component = expected_velocity / np.sqrt(2)  # x and y components
        velocity_pixel_per_frame = expected_velocity * 10 / pixel_to_mm / fps
        component_pixel = velocity_pixel_per_frame / np.sqrt(2)

        positions = np.zeros((n_frames, 2))
        positions[:, 0] = np.arange(n_frames) * component_pixel
        positions[:, 1] = np.arange(n_frames) * component_pixel

        speed = analyzer.compute_speed(positions, fps, pixel_to_mm)

        mean_speed = np.mean(speed)
        # Should match expected velocity magnitude
        assert abs(mean_speed - expected_velocity) < expected_velocity * 0.1


class TestVelocityWithAcceleration:
    """Tests for acceleration scenarios."""

    def test_velocity_with_acceleration_linear(
        self,
        sample_video_metadata,
        velocity_analyzer_config
    ):
        """Test velocity calculation with linear acceleration."""
        analyzer = VelocityAnalyzer(**velocity_analyzer_config)

        n_frames = 1000
        fps = sample_video_metadata["fps"]
        pixel_to_mm = sample_video_metadata["xPixW"]
        duration = n_frames / fps

        # Linear acceleration: 0 to 20 cm/s over duration
        t = np.linspace(0, duration, n_frames)
        velocity_profile = 20 * t / duration  # Linear increase

        # Integrate velocity to get position
        dt = duration / n_frames
        position_cm = np.cumsum(velocity_profile) * dt
        position_pixel = position_cm * 10 / pixel_to_mm  # cm to pixel

        positions = np.zeros((n_frames, 2))
        positions[:, 0] = position_pixel
        positions[:, 1] = 100

        speed = analyzer.compute_speed(positions, fps, pixel_to_mm)

        # Speed should increase over time
        first_quarter = np.mean(speed[:len(speed)//4])
        last_quarter = np.mean(speed[-len(speed)//4:])

        assert last_quarter > first_quarter
        # Final speed should be approximately 20 cm/s
        assert abs(last_quarter - 20) < 5  # Allow some tolerance

    def test_velocity_with_deceleration(
        self,
        sample_video_metadata,
        velocity_analyzer_config
    ):
        """Test velocity calculation with deceleration (negative acceleration)."""
        analyzer = VelocityAnalyzer(**velocity_analyzer_config)

        n_frames = 1000
        fps = sample_video_metadata["fps"]
        pixel_to_mm = sample_video_metadata["xPixW"]
        duration = n_frames / fps

        # Deceleration: 20 to 0 cm/s
        t = np.linspace(0, duration, n_frames)
        velocity_profile = 20 * (1 - t / duration)

        dt = duration / n_frames
        position_cm = np.cumsum(velocity_profile) * dt
        position_pixel = position_cm * 10 / pixel_to_mm

        positions = np.zeros((n_frames, 2))
        positions[:, 0] = position_pixel
        positions[:, 1] = 100

        speed = analyzer.compute_speed(positions, fps, pixel_to_mm)

        # Speed should decrease over time
        first_quarter = np.mean(speed[:len(speed)//4])
        last_quarter = np.mean(speed[-len(speed)//4:])

        assert first_quarter > last_quarter

    def test_acceleration_computation(
        self,
        sample_video_metadata,
        velocity_analyzer_config
    ):
        """Test acceleration calculation from speed data."""
        analyzer = VelocityAnalyzer(**velocity_analyzer_config)

        n_frames = 500
        fps = sample_video_metadata["fps"]

        # Create speed data with known acceleration
        # Linear increase in speed = constant acceleration
        speed = np.linspace(0, 20, n_frames)  # 0 to 20 cm/s

        accel = analyzer.compute_acceleration(speed, fps)

        # Acceleration should be approximately constant
        expected_accel = 20 * fps / n_frames  # (delta_v / delta_t)
        mean_accel = np.mean(accel)

        # Allow tolerance for smoothing
        assert abs(mean_accel - expected_accel) < expected_accel * 0.2

    def test_acceleration_with_varying_phases(
        self,
        sample_video_metadata,
        velocity_analyzer_config
    ):
        """Test acceleration with acceleration/constant/deceleration phases."""
        analyzer = VelocityAnalyzer(**velocity_analyzer_config)

        fps = sample_video_metadata["fps"]

        # Create three-phase speed profile
        n_phase = 200
        speed_accel = np.linspace(0, 20, n_phase)
        speed_const = np.full(n_phase, 20)
        speed_decel = np.linspace(20, 0, n_phase)

        speed = np.concatenate([speed_accel, speed_const, speed_decel])

        accel = analyzer.compute_acceleration(speed, fps)

        # Check phases
        phase_len = len(accel) // 3

        # First phase: positive acceleration
        accel_phase = np.mean(accel[:phase_len])
        assert accel_phase > 0

        # Last phase: negative acceleration
        decel_phase = np.mean(accel[-phase_len:])
        assert decel_phase < 0


class TestSmoothingFilter:
    """Tests for smoothing filter functionality."""

    def test_smoothing_filter_moving_average(self, velocity_analyzer_config):
        """Test moving average smoothing."""
        analyzer = VelocityAnalyzer(**velocity_analyzer_config)

        # Noisy data
        np.random.seed(42)
        n_points = 500
        clean_signal = np.sin(np.linspace(0, 4 * np.pi, n_points)) * 10
        noisy_signal = clean_signal + np.random.normal(0, 2, n_points)

        smoothed = analyzer.apply_smoothing(
            noisy_signal,
            method='moving_average',
            window_size=10
        )

        # Smoothed should be closer to clean signal
        # Compare on overlapping region
        offset = (len(noisy_signal) - len(smoothed)) // 2
        clean_trimmed = clean_signal[offset:offset + len(smoothed)]
        noisy_trimmed = noisy_signal[offset:offset + len(smoothed)]

        error_smoothed = np.mean((smoothed - clean_trimmed)**2)
        error_noisy = np.mean((noisy_trimmed - clean_trimmed)**2)

        assert error_smoothed < error_noisy

    def test_smoothing_filter_savgol(self, velocity_analyzer_config):
        """Test Savitzky-Golay smoothing."""
        analyzer = VelocityAnalyzer(**velocity_analyzer_config)

        np.random.seed(42)
        n_points = 500
        clean_signal = np.sin(np.linspace(0, 4 * np.pi, n_points)) * 10
        noisy_signal = clean_signal + np.random.normal(0, 2, n_points)

        smoothed = analyzer.apply_smoothing(
            noisy_signal,
            method='savgol',
            window_size=11
        )

        # Savgol should preserve peaks better than moving average
        error_smoothed = np.mean((smoothed - clean_signal)**2)
        error_noisy = np.mean((noisy_signal - clean_signal)**2)

        assert error_smoothed < error_noisy

    def test_smoothing_preserves_length_savgol(self, velocity_analyzer_config):
        """Test that Savgol preserves array length."""
        analyzer = VelocityAnalyzer(**velocity_analyzer_config)

        data = np.random.randn(100)
        smoothed = analyzer.apply_smoothing(data, method='savgol', window_size=11)

        assert len(smoothed) == len(data)

    def test_smoothing_reduces_variance(self, velocity_analyzer_config):
        """Test that smoothing reduces noise variance."""
        analyzer = VelocityAnalyzer(**velocity_analyzer_config)

        np.random.seed(42)
        noisy_data = np.random.randn(500)

        smoothed = analyzer.apply_smoothing(noisy_data, method='moving_average')

        assert np.var(smoothed) < np.var(noisy_data)

    def test_smoothing_invalid_method(self, velocity_analyzer_config):
        """Test that invalid smoothing method raises error."""
        analyzer = VelocityAnalyzer(**velocity_analyzer_config)

        data = np.random.randn(100)

        with pytest.raises(ValueError, match="Unknown smoothing method"):
            analyzer.apply_smoothing(data, method='invalid_method')


class TestDragDetection:
    """Tests for drag event detection."""

    def test_drag_detection_no_events(
        self,
        sample_video_metadata,
        velocity_analyzer_config
    ):
        """Test drag detection with constant velocity (no drag events)."""
        analyzer = VelocityAnalyzer(**velocity_analyzer_config)

        fps = sample_video_metadata["fps"]

        # Constant speed = zero acceleration = no drag events
        speed = np.full(1000, 10.0)
        accel = analyzer.compute_acceleration(speed, fps)

        drag_idx, recovery_idx, stats = analyzer.detect_drag_events(accel, fps)

        # No significant drag or recovery events expected
        assert stats['drag_count'] == 0 or stats['drag_duration'] < 0.1
        assert stats['recovery_count'] == 0 or stats['recovery_duration'] < 0.1

    def test_drag_detection_with_drag_events(
        self,
        sample_video_metadata,
        velocity_analyzer_config
    ):
        """Test drag detection with clear drag events."""
        analyzer = VelocityAnalyzer(**velocity_analyzer_config)

        fps = sample_video_metadata["fps"]

        # Create acceleration with clear drag events
        # Drag = sustained negative acceleration
        n_frames = 2000
        accel = np.zeros(n_frames)

        # Add drag event (negative acceleration for 0.5 seconds)
        drag_start = 500
        drag_duration_frames = int(0.5 * fps)
        accel[drag_start:drag_start + drag_duration_frames] = -50

        # Add recovery event
        recovery_start = 1000
        recovery_duration_frames = int(0.3 * fps)
        accel[recovery_start:recovery_start + recovery_duration_frames] = 50

        drag_idx, recovery_idx, stats = analyzer.detect_drag_events(accel, fps)

        assert stats['drag_count'] >= 1
        assert stats['recovery_count'] >= 1
        assert stats['drag_duration'] > 0.25  # At least threshold duration

    def test_drag_detection_short_events_ignored(
        self,
        sample_video_metadata,
        velocity_analyzer_config
    ):
        """Test that events shorter than threshold are ignored."""
        analyzer = VelocityAnalyzer(**velocity_analyzer_config)

        fps = sample_video_metadata["fps"]

        # Create very short acceleration spikes
        n_frames = 1000
        accel = np.zeros(n_frames)

        # Short spike (< threshold)
        short_duration = int(0.1 * fps)  # 0.1s < 0.25s threshold
        accel[100:100 + short_duration] = -100

        drag_idx, recovery_idx, stats = analyzer.detect_drag_events(accel, fps)

        # Short event should be ignored
        assert stats['drag_count'] == 0

    def test_drag_detection_statistics(
        self,
        sample_video_metadata,
        velocity_analyzer_config
    ):
        """Test drag detection statistics calculation."""
        analyzer = VelocityAnalyzer(**velocity_analyzer_config)

        fps = sample_video_metadata["fps"]

        # Create known acceleration pattern
        n_frames = 1000
        accel = np.random.randn(n_frames) * 10  # Random fluctuations

        # Set known extremes
        accel[200] = 100  # Peak
        accel[500] = -80  # Minimum

        _, _, stats = analyzer.detect_drag_events(accel, fps)

        assert 'drag_count' in stats
        assert 'recovery_count' in stats
        assert 'drag_duration' in stats
        assert 'recovery_duration' in stats
        assert 'peak_acceleration' in stats
        assert 'min_acceleration' in stats

        assert stats['peak_acceleration'] >= 100
        assert stats['min_acceleration'] <= -80


class TestVelocityFromTracking:
    """Tests for velocity computation from DeepLabCut tracking data."""

    def test_velocity_from_markers(
        self,
        sample_tracks,
        sample_markers,
        sample_video_metadata,
        velocity_analyzer_config
    ):
        """Test velocity calculation from tracking DataFrame."""
        analyzer = VelocityAnalyzer(**velocity_analyzer_config)

        model_name = "DLC_mouse_model"
        fps = sample_video_metadata["fps"]
        pixel_to_mm = sample_video_metadata["xPixW"]

        speed = analyzer.compute_speed_from_markers(
            sample_tracks,
            model_name,
            sample_markers.speed_markers,
            fps,
            pixel_to_mm
        )

        # Should return valid speed array
        assert len(speed) > 0
        assert np.all(np.isfinite(speed))
        assert np.all(speed >= 0)  # Speed is non-negative

    def test_velocity_from_constant_tracks(
        self,
        sample_tracks_constant_velocity,
        sample_markers,
        sample_video_metadata,
        velocity_analyzer_config
    ):
        """Test velocity from tracks with constant velocity."""
        analyzer = VelocityAnalyzer(**velocity_analyzer_config)

        model_name = "DLC_mouse_model"
        fps = sample_video_metadata["fps"]
        pixel_to_mm = sample_video_metadata["xPixW"]

        speed = analyzer.compute_speed_from_markers(
            sample_tracks_constant_velocity,
            model_name,
            sample_markers.speed_markers,
            fps,
            pixel_to_mm
        )

        # Speed should be relatively constant
        std_speed = np.std(speed)
        mean_speed = np.mean(speed)

        # Coefficient of variation should be low
        if mean_speed > 0:
            cv = std_speed / mean_speed
            assert cv < 0.2  # Less than 20% variation


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_positions(self, velocity_analyzer_config):
        """Test handling of empty position data."""
        analyzer = VelocityAnalyzer(**velocity_analyzer_config)

        positions = np.array([]).reshape(0, 2)
        speed = analyzer.compute_speed(positions, 100, 0.3125)

        assert len(speed) == 1
        assert speed[0] == 0.0

    def test_single_frame(self, velocity_analyzer_config):
        """Test handling of single frame."""
        analyzer = VelocityAnalyzer(**velocity_analyzer_config)

        positions = np.array([[100, 200]])
        speed = analyzer.compute_speed(positions, 100, 0.3125)

        assert len(speed) == 1
        assert speed[0] == 0.0

    def test_two_frames(self, velocity_analyzer_config):
        """Test handling of two frames."""
        analyzer = VelocityAnalyzer(**velocity_analyzer_config)

        positions = np.array([[100, 200], [110, 200]])
        speed = analyzer.compute_speed(positions, 100, 0.3125)

        assert len(speed) == 1
        assert speed[0] > 0

    def test_nan_handling(self, velocity_analyzer_config):
        """Test handling of NaN values in data."""
        analyzer = VelocityAnalyzer(**velocity_analyzer_config)

        # Data with NaN
        speed = np.array([10, 12, np.nan, 14, 15])

        # Smoothing should handle NaN gracefully or raise
        # Current implementation may propagate NaN
        smoothed = analyzer.apply_smoothing(speed, method='moving_average')

        # Just verify it doesn't crash
        assert len(smoothed) > 0


# =============================================================================
# Parametrized Tests
# =============================================================================

@pytest.mark.parametrize("velocity,expected_range", [
    (5.0, (4.0, 6.0)),
    (10.0, (8.0, 12.0)),
    (20.0, (16.0, 24.0)),
    (50.0, (40.0, 60.0)),
])
def test_velocity_range(
    velocity,
    expected_range,
    sample_video_metadata,
    velocity_analyzer_config
):
    """Test velocity calculation across different speeds."""
    analyzer = VelocityAnalyzer(**velocity_analyzer_config)

    n_frames = 500
    fps = sample_video_metadata["fps"]
    pixel_to_mm = sample_video_metadata["xPixW"]

    velocity_pixel_per_frame = velocity * 10 / pixel_to_mm / fps

    positions = np.zeros((n_frames, 2))
    positions[:, 0] = np.arange(n_frames) * velocity_pixel_per_frame
    positions[:, 1] = 100

    speed = analyzer.compute_speed(positions, fps, pixel_to_mm)
    mean_speed = np.mean(speed)

    assert expected_range[0] <= mean_speed <= expected_range[1]


@pytest.mark.parametrize("smoothing_factor", [5, 10, 20, 50])
def test_smoothing_factor_effect(smoothing_factor, velocity_analyzer_config):
    """Test effect of different smoothing factors."""
    config = velocity_analyzer_config.copy()
    config['smoothing_factor'] = smoothing_factor
    analyzer = VelocityAnalyzer(**config)

    np.random.seed(42)
    noisy_data = np.random.randn(500)

    smoothed = analyzer.apply_smoothing(noisy_data, method='moving_average')

    # Higher smoothing = lower variance
    var_ratio = np.var(smoothed) / np.var(noisy_data)

    # Variance should decrease with smoothing
    assert var_ratio < 1.0
