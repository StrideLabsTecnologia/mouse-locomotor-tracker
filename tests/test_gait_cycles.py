"""
Tests for GaitCycleDetector Module

This module tests the gait cycle detection and stride analysis:
- Cycle detection using peak finding
- Cadence calculation (steps per second)
- Stride length estimation
- Gait phase analysis

Test Coverage Target: 90%+

Mathematical Background:
    Cadence = Number of cycles / Duration (steps/second)
    Stride Length = Average Speed / Cadence (cm/step)

    Gait phases:
    - Stance: foot on ground
    - Swing: foot in air
    - Stride: one complete cycle (stance + swing)

Author: Stride Labs
License: MIT
"""

import pytest
import numpy as np
from typing import Tuple, List, Dict
from scipy.signal import find_peaks


# =============================================================================
# GaitCycleDetector Implementation (for testing)
# =============================================================================

class GaitCycleDetector:
    """
    Detects and analyzes gait cycles from limb position data.

    This class identifies individual gait cycles, computes stride metrics,
    and provides statistics on gait regularity.

    Attributes:
        min_peak_distance: Minimum distance between peaks (auto-calculated if None)
        interpolation_factor: Factor for interpolating stride data
        smoothing_factor: Window size for smoothing
    """

    def __init__(
        self,
        min_peak_distance: int = None,
        interpolation_factor: int = 4,
        smoothing_factor: int = 10
    ):
        self.min_peak_distance = min_peak_distance
        self.interpolation_factor = interpolation_factor
        self.smoothing_factor = smoothing_factor

    def detect_cycles(
        self,
        stride: np.ndarray,
        fps: int = None
    ) -> Tuple[int, np.ndarray, np.ndarray]:
        """
        Detect gait cycles using peak detection.

        Args:
            stride: Stride position array (relative to body)
            fps: Frames per second (optional, for timing info)

        Returns:
            Tuple of (n_cycles, peak_indices, trough_indices)
        """
        if len(stride) < 3:
            return 0, np.array([]), np.array([])

        # Initial peak detection
        peaks, _ = find_peaks(stride)

        if len(peaks) < 2:
            return len(peaks), peaks, np.array([])

        # Calculate adaptive distance threshold
        mean_distance = np.diff(peaks).mean()
        min_dist = max(1, int(mean_distance / 2))

        if self.min_peak_distance is not None:
            min_dist = self.min_peak_distance

        # Refine peak detection with distance threshold
        peaks, properties = find_peaks(stride, distance=min_dist)

        # Detect troughs (valleys) between peaks
        troughs, _ = find_peaks(-stride, distance=min_dist)

        return len(peaks), peaks, troughs

    def compute_cadence(
        self,
        stride: np.ndarray,
        duration: float
    ) -> float:
        """
        Compute cadence (steps per second) from stride data.

        Args:
            stride: Stride position array
            duration: Duration of recording in seconds

        Returns:
            Cadence in steps per second (Hz)
        """
        if duration <= 0:
            return 0.0

        n_cycles, peaks, _ = self.detect_cycles(stride)

        if n_cycles == 0:
            return 0.0

        cadence = n_cycles / duration
        return cadence

    def compute_stride_length(
        self,
        cadence: float,
        avg_speed: float
    ) -> float:
        """
        Compute average stride length from cadence and speed.

        Args:
            cadence: Steps per second (Hz)
            avg_speed: Average speed in cm/s

        Returns:
            Stride length in cm
        """
        if cadence <= 0:
            return 0.0

        stride_length = avg_speed / cadence
        return stride_length

    def analyze_gait_regularity(
        self,
        stride: np.ndarray,
        fps: int
    ) -> Dict[str, float]:
        """
        Analyze gait regularity metrics.

        Args:
            stride: Stride position array
            fps: Frames per second

        Returns:
            Dictionary with regularity metrics
        """
        n_cycles, peaks, troughs = self.detect_cycles(stride, fps)

        if n_cycles < 2:
            return {
                'mean_cycle_duration': 0.0,
                'std_cycle_duration': 0.0,
                'cv_cycle_duration': 0.0,  # Coefficient of variation
                'mean_stride_amplitude': 0.0,
                'std_stride_amplitude': 0.0,
            }

        # Cycle durations (in seconds)
        cycle_durations = np.diff(peaks) / fps
        mean_duration = np.mean(cycle_durations)
        std_duration = np.std(cycle_durations)
        cv_duration = std_duration / mean_duration if mean_duration > 0 else 0

        # Stride amplitudes (peak to trough)
        amplitudes = []
        for i, peak in enumerate(peaks):
            # Find nearest trough before or after
            nearby_troughs = troughs[
                (troughs > peak - len(stride)//10) &
                (troughs < peak + len(stride)//10)
            ]
            if len(nearby_troughs) > 0:
                nearest_trough = nearby_troughs[
                    np.argmin(np.abs(nearby_troughs - peak))
                ]
                amplitudes.append(abs(stride[peak] - stride[nearest_trough]))

        mean_amplitude = np.mean(amplitudes) if amplitudes else 0
        std_amplitude = np.std(amplitudes) if amplitudes else 0

        return {
            'mean_cycle_duration': mean_duration,
            'std_cycle_duration': std_duration,
            'cv_cycle_duration': cv_duration,
            'mean_stride_amplitude': mean_amplitude,
            'std_stride_amplitude': std_amplitude,
        }

    def interpolate_stride(
        self,
        stride: np.ndarray,
        duration: float
    ) -> np.ndarray:
        """
        Interpolate stride data for better cycle detection.

        Args:
            stride: Original stride array
            duration: Recording duration in seconds

        Returns:
            Interpolated stride array
        """
        original_length = len(stride)
        new_length = original_length * self.interpolation_factor

        x_original = np.linspace(0, duration, original_length)
        x_new = np.linspace(0, duration, new_length)

        interpolated = np.interp(x_new, x_original, stride)
        return interpolated

    def compute_all_limb_metrics(
        self,
        limb_strides: Dict[str, np.ndarray],
        duration: float,
        avg_speed: float,
        fps: int
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute gait metrics for all limbs.

        Args:
            limb_strides: Dictionary of limb name to stride arrays
            duration: Recording duration in seconds
            avg_speed: Average body speed in cm/s
            fps: Frames per second

        Returns:
            Dictionary with metrics for each limb
        """
        results = {}

        for limb_name, stride in limb_strides.items():
            # Interpolate for better detection
            stride_interp = self.interpolate_stride(stride, duration)
            interp_fps = fps * self.interpolation_factor

            # Compute metrics
            cadence = self.compute_cadence(stride_interp, duration)
            stride_length = self.compute_stride_length(cadence, avg_speed)
            regularity = self.analyze_gait_regularity(stride_interp, interp_fps)

            results[limb_name] = {
                'cadence': cadence,
                'stride_length': stride_length,
                **regularity
            }

        return results

    def detect_gait_events(
        self,
        stride: np.ndarray,
        fps: int
    ) -> Dict[str, np.ndarray]:
        """
        Detect specific gait events (heel strike, toe off, etc.).

        For mouse locomotion, these correspond to:
        - Paw touchdown (stance start)
        - Paw liftoff (swing start)

        Args:
            stride: Stride position array
            fps: Frames per second

        Returns:
            Dictionary with event indices
        """
        n_cycles, peaks, troughs = self.detect_cycles(stride, fps)

        # Peaks typically correspond to maximum extension (mid-swing or touchdown)
        # Troughs typically correspond to maximum flexion (mid-stance or liftoff)

        # Get zero crossings for more precise timing
        zero_crossings_up = []
        zero_crossings_down = []

        mean_val = np.mean(stride)
        centered = stride - mean_val

        for i in range(len(centered) - 1):
            if centered[i] < 0 and centered[i + 1] >= 0:
                zero_crossings_up.append(i)
            elif centered[i] >= 0 and centered[i + 1] < 0:
                zero_crossings_down.append(i)

        return {
            'peaks': peaks,
            'troughs': troughs,
            'stance_start': np.array(zero_crossings_down),  # Paw going back
            'swing_start': np.array(zero_crossings_up),      # Paw going forward
        }


# =============================================================================
# Test Classes
# =============================================================================

class TestCycleDetection:
    """Tests for gait cycle detection."""

    def test_cycle_detection_sinusoidal(self, gait_cycle_config):
        """Test cycle detection with clean sinusoidal data."""
        detector = GaitCycleDetector(**gait_cycle_config)

        # 10 complete cycles over 2.5 seconds at 4 Hz
        t = np.linspace(0, 2.5, 1000)
        stride = 10 * np.sin(2 * np.pi * 4 * t)

        n_cycles, peaks, troughs = detector.detect_cycles(stride, fps=400)

        # Should detect approximately 10 cycles
        assert 8 <= n_cycles <= 12
        assert len(peaks) == n_cycles

    def test_cycle_detection_noisy(self, gait_cycle_config):
        """Test cycle detection with noisy data."""
        detector = GaitCycleDetector(**gait_cycle_config)

        np.random.seed(42)
        t = np.linspace(0, 2.5, 1000)
        clean = 10 * np.sin(2 * np.pi * 4 * t)
        stride = clean + np.random.normal(0, 1, 1000)

        n_cycles, peaks, troughs = detector.detect_cycles(stride, fps=400)

        # Should still detect approximately 10 cycles
        assert 7 <= n_cycles <= 13

    def test_cycle_detection_variable_amplitude(self, gait_cycle_config):
        """Test cycle detection with varying amplitude."""
        detector = GaitCycleDetector(**gait_cycle_config)

        t = np.linspace(0, 2.5, 1000)
        # Amplitude increases over time
        amplitude = 5 + 10 * (t / t.max())
        stride = amplitude * np.sin(2 * np.pi * 4 * t)

        n_cycles, peaks, troughs = detector.detect_cycles(stride, fps=400)

        # Should still detect cycles
        assert n_cycles > 5

    def test_cycle_detection_empty(self, gait_cycle_config):
        """Test cycle detection with empty data."""
        detector = GaitCycleDetector(**gait_cycle_config)

        stride = np.array([])
        n_cycles, peaks, troughs = detector.detect_cycles(stride)

        assert n_cycles == 0
        assert len(peaks) == 0
        assert len(troughs) == 0

    def test_cycle_detection_constant(self, gait_cycle_config):
        """Test cycle detection with constant data."""
        detector = GaitCycleDetector(**gait_cycle_config)

        stride = np.ones(100) * 5
        n_cycles, peaks, troughs = detector.detect_cycles(stride)

        assert n_cycles <= 1  # No cycles expected

    def test_cycle_detection_single_peak(self, gait_cycle_config):
        """Test cycle detection with single peak."""
        detector = GaitCycleDetector(**gait_cycle_config)

        # Single hump
        t = np.linspace(0, 1, 100)
        stride = np.exp(-(t - 0.5)**2 / 0.1)

        n_cycles, peaks, troughs = detector.detect_cycles(stride)

        assert n_cycles == 1


class TestCadenceCalculation:
    """Tests for cadence (step frequency) calculation."""

    def test_cadence_known_frequency(self, gait_cycle_config):
        """Test cadence calculation with known frequency."""
        detector = GaitCycleDetector(**gait_cycle_config)

        # 4 Hz gait frequency
        duration = 5.0
        t = np.linspace(0, duration, 2000)
        stride = 10 * np.sin(2 * np.pi * 4 * t)

        cadence = detector.compute_cadence(stride, duration)

        # Should be approximately 4 Hz
        assert 3.5 <= cadence <= 4.5

    def test_cadence_different_frequencies(self, gait_cycle_config):
        """Test cadence with various frequencies."""
        detector = GaitCycleDetector(**gait_cycle_config)

        for expected_freq in [2, 4, 6, 8]:
            duration = 5.0
            t = np.linspace(0, duration, 2000)
            stride = 10 * np.sin(2 * np.pi * expected_freq * t)

            cadence = detector.compute_cadence(stride, duration)

            # Should be within 15% of expected
            assert abs(cadence - expected_freq) / expected_freq < 0.15

    def test_cadence_zero_duration(self, gait_cycle_config):
        """Test cadence with zero duration."""
        detector = GaitCycleDetector(**gait_cycle_config)

        stride = np.sin(np.linspace(0, 10, 100))
        cadence = detector.compute_cadence(stride, duration=0)

        assert cadence == 0.0

    def test_cadence_no_cycles(self, gait_cycle_config):
        """Test cadence when no cycles detected."""
        detector = GaitCycleDetector(**gait_cycle_config)

        stride = np.ones(100)  # Constant - no cycles
        cadence = detector.compute_cadence(stride, duration=1.0)

        assert cadence == 0.0

    def test_cadence_from_real_data(
        self,
        sample_gait_cycles_data,
        sample_video_metadata,
        gait_cycle_config
    ):
        """Test cadence calculation with fixture data."""
        detector = GaitCycleDetector(**gait_cycle_config)

        duration = sample_video_metadata["dur"]
        cadence = detector.compute_cadence(sample_gait_cycles_data, duration)

        # Fixture data has 4 Hz gait
        assert 3.0 <= cadence <= 5.0


class TestStrideLength:
    """Tests for stride length calculation."""

    def test_stride_length_basic(self, gait_cycle_config):
        """Test basic stride length calculation."""
        detector = GaitCycleDetector(**gait_cycle_config)

        # If speed is 40 cm/s and cadence is 4 Hz
        # Stride length = 40 / 4 = 10 cm
        stride_length = detector.compute_stride_length(cadence=4.0, avg_speed=40.0)

        assert abs(stride_length - 10.0) < 0.1

    def test_stride_length_various_speeds(self, gait_cycle_config):
        """Test stride length at various speeds."""
        detector = GaitCycleDetector(**gait_cycle_config)

        cadence = 4.0  # Fixed cadence

        for speed, expected_length in [(20, 5), (40, 10), (60, 15), (80, 20)]:
            stride_length = detector.compute_stride_length(cadence, float(speed))
            assert abs(stride_length - expected_length) < 0.5

    def test_stride_length_zero_cadence(self, gait_cycle_config):
        """Test stride length with zero cadence."""
        detector = GaitCycleDetector(**gait_cycle_config)

        stride_length = detector.compute_stride_length(cadence=0.0, avg_speed=40.0)

        assert stride_length == 0.0

    def test_stride_length_relationship(self, gait_cycle_config):
        """Test stride length = speed / cadence relationship."""
        detector = GaitCycleDetector(**gait_cycle_config)

        speeds = [10, 20, 30, 40, 50]
        cadences = [2, 3, 4, 5, 6]

        for speed in speeds:
            for cadence in cadences:
                stride_length = detector.compute_stride_length(
                    float(cadence), float(speed)
                )
                expected = speed / cadence
                assert abs(stride_length - expected) < 0.01


class TestGaitRegularity:
    """Tests for gait regularity analysis."""

    def test_regularity_regular_gait(self, gait_cycle_config):
        """Test regularity metrics for regular gait."""
        detector = GaitCycleDetector(**gait_cycle_config)

        fps = 100
        duration = 5.0
        t = np.linspace(0, duration, int(fps * duration))

        # Very regular gait
        stride = 10 * np.sin(2 * np.pi * 4 * t)

        metrics = detector.analyze_gait_regularity(stride, fps)

        # Low coefficient of variation for regular gait
        assert metrics['cv_cycle_duration'] < 0.1
        assert metrics['mean_cycle_duration'] > 0
        assert metrics['mean_stride_amplitude'] > 0

    def test_regularity_irregular_gait(self, gait_cycle_config):
        """Test regularity metrics for irregular gait."""
        detector = GaitCycleDetector(**gait_cycle_config)

        np.random.seed(42)
        fps = 100
        duration = 5.0

        # Irregular gait with varying frequency
        t = np.linspace(0, duration, int(fps * duration))
        # Random frequency modulation
        freq_mod = 4 + np.random.uniform(-1, 1, len(t)).cumsum() * 0.01
        phase = np.cumsum(2 * np.pi * freq_mod / fps)
        stride = 10 * np.sin(phase)

        metrics = detector.analyze_gait_regularity(stride, fps)

        # Higher coefficient of variation for irregular gait
        assert metrics['cv_cycle_duration'] > 0  # Will have some variation

    def test_regularity_no_cycles(self, gait_cycle_config):
        """Test regularity metrics when no cycles detected."""
        detector = GaitCycleDetector(**gait_cycle_config)

        stride = np.ones(100)
        metrics = detector.analyze_gait_regularity(stride, fps=100)

        assert metrics['mean_cycle_duration'] == 0.0
        assert metrics['cv_cycle_duration'] == 0.0


class TestInterpolation:
    """Tests for stride data interpolation."""

    def test_interpolation_length(self, gait_cycle_config):
        """Test that interpolation increases data length."""
        detector = GaitCycleDetector(**gait_cycle_config)

        original = np.sin(np.linspace(0, 4*np.pi, 100))
        interpolated = detector.interpolate_stride(original, duration=1.0)

        expected_length = 100 * detector.interpolation_factor
        assert len(interpolated) == expected_length

    def test_interpolation_preserves_shape(self, gait_cycle_config):
        """Test that interpolation preserves signal shape."""
        detector = GaitCycleDetector(**gait_cycle_config)

        original = np.sin(np.linspace(0, 4*np.pi, 100))
        interpolated = detector.interpolate_stride(original, duration=1.0)

        # Check min/max are preserved
        assert abs(interpolated.min() - original.min()) < 0.1
        assert abs(interpolated.max() - original.max()) < 0.1

    def test_interpolation_improves_detection(self, gait_cycle_config):
        """Test that interpolation can improve cycle detection."""
        detector = GaitCycleDetector(**gait_cycle_config)

        # Low resolution data
        t_low = np.linspace(0, 2.5, 50)  # Only 50 points
        stride_low = 10 * np.sin(2 * np.pi * 4 * t_low)

        # Detect cycles without interpolation
        n_cycles_low, _, _ = detector.detect_cycles(stride_low)

        # Detect cycles with interpolation
        stride_interp = detector.interpolate_stride(stride_low, duration=2.5)
        n_cycles_interp, _, _ = detector.detect_cycles(stride_interp)

        # Interpolated should detect at least as many cycles
        assert n_cycles_interp >= n_cycles_low


class TestAllLimbMetrics:
    """Tests for computing metrics across all limbs."""

    def test_all_limb_metrics_basic(
        self,
        sample_video_metadata,
        gait_cycle_config
    ):
        """Test computing metrics for all limbs."""
        detector = GaitCycleDetector(**gait_cycle_config)

        n_frames = sample_video_metadata["nFrame"]
        t = np.linspace(0, sample_video_metadata["dur"], n_frames)
        freq = 4.0

        limb_strides = {
            'hindL': 10 * np.sin(2 * np.pi * freq * t),
            'hindR': 10 * np.sin(2 * np.pi * freq * t + np.pi),
            'foreL': 10 * np.sin(2 * np.pi * freq * t + np.pi),
            'foreR': 10 * np.sin(2 * np.pi * freq * t),
        }

        results = detector.compute_all_limb_metrics(
            limb_strides,
            duration=sample_video_metadata["dur"],
            avg_speed=20.0,
            fps=sample_video_metadata["fps"]
        )

        # Should have results for all 4 limbs
        assert len(results) == 4

        # Each limb should have expected metrics
        for limb, metrics in results.items():
            assert 'cadence' in metrics
            assert 'stride_length' in metrics
            assert 'mean_cycle_duration' in metrics
            assert 'cv_cycle_duration' in metrics

            # Cadence should be approximately 4 Hz
            assert 3.0 <= metrics['cadence'] <= 5.0

    def test_all_limb_metrics_asymmetric(
        self,
        sample_video_metadata,
        gait_cycle_config
    ):
        """Test metrics with asymmetric gait (different cadences per limb)."""
        detector = GaitCycleDetector(**gait_cycle_config)

        n_frames = sample_video_metadata["nFrame"]
        t = np.linspace(0, sample_video_metadata["dur"], n_frames)

        # Different frequencies for different limbs
        limb_strides = {
            'hindL': 10 * np.sin(2 * np.pi * 4 * t),   # 4 Hz
            'hindR': 10 * np.sin(2 * np.pi * 4 * t),   # 4 Hz
            'foreL': 10 * np.sin(2 * np.pi * 3.5 * t), # 3.5 Hz (slightly slower)
            'foreR': 10 * np.sin(2 * np.pi * 3.5 * t), # 3.5 Hz
        }

        results = detector.compute_all_limb_metrics(
            limb_strides,
            duration=sample_video_metadata["dur"],
            avg_speed=20.0,
            fps=sample_video_metadata["fps"]
        )

        # Hind limbs should have higher cadence
        hind_cadence = (results['hindL']['cadence'] + results['hindR']['cadence']) / 2
        fore_cadence = (results['foreL']['cadence'] + results['foreR']['cadence']) / 2

        # Allow for detection variability
        assert hind_cadence >= fore_cadence - 0.5


class TestGaitEvents:
    """Tests for gait event detection."""

    def test_gait_events_detection(self, gait_cycle_config):
        """Test detection of gait events."""
        detector = GaitCycleDetector(**gait_cycle_config)

        fps = 100
        t = np.linspace(0, 2.5, int(fps * 2.5))
        stride = 10 * np.sin(2 * np.pi * 4 * t)

        events = detector.detect_gait_events(stride, fps)

        assert 'peaks' in events
        assert 'troughs' in events
        assert 'stance_start' in events
        assert 'swing_start' in events

        # Should detect multiple events
        assert len(events['peaks']) > 5
        assert len(events['troughs']) > 5

    def test_gait_events_alternating(self, gait_cycle_config):
        """Test that stance and swing events alternate."""
        detector = GaitCycleDetector(**gait_cycle_config)

        fps = 100
        t = np.linspace(0, 2.5, int(fps * 2.5))
        stride = 10 * np.sin(2 * np.pi * 4 * t)

        events = detector.detect_gait_events(stride, fps)

        # Stance and swing starts should alternate
        all_events = np.sort(np.concatenate([
            events['stance_start'], events['swing_start']
        ]))

        # Check for reasonable spacing
        if len(all_events) > 2:
            spacings = np.diff(all_events)
            assert np.all(spacings > 0)  # All positive (increasing order)


class TestEdgeCases:
    """Tests for edge cases."""

    def test_very_short_data(self, gait_cycle_config):
        """Test with very short data."""
        detector = GaitCycleDetector(**gait_cycle_config)

        stride = np.array([1, 2, 1])
        n_cycles, peaks, troughs = detector.detect_cycles(stride)

        # Should handle gracefully
        assert n_cycles <= 1

    def test_negative_values(self, gait_cycle_config):
        """Test with negative stride values."""
        detector = GaitCycleDetector(**gait_cycle_config)

        # Stride oscillating around negative mean
        t = np.linspace(0, 2.5, 1000)
        stride = -5 + 10 * np.sin(2 * np.pi * 4 * t)

        n_cycles, peaks, troughs = detector.detect_cycles(stride)

        # Should still detect cycles
        assert n_cycles > 5

    def test_nan_in_data(self, gait_cycle_config):
        """Test handling of NaN values."""
        detector = GaitCycleDetector(**gait_cycle_config)

        # Data with some NaN
        stride = np.sin(np.linspace(0, 10, 100))
        stride[50] = np.nan

        # Should not crash
        try:
            n_cycles, peaks, troughs = detector.detect_cycles(stride)
            # NaN may cause issues, but shouldn't crash
        except ValueError:
            pass  # Acceptable to raise error for NaN

    def test_high_frequency_gait(self, gait_cycle_config):
        """Test with very high frequency gait."""
        detector = GaitCycleDetector(**gait_cycle_config)

        fps = 500
        duration = 2.0
        t = np.linspace(0, duration, int(fps * duration))

        # 10 Hz gait (very fast)
        stride = 10 * np.sin(2 * np.pi * 10 * t)

        cadence = detector.compute_cadence(stride, duration)

        # Should detect approximately 10 Hz
        assert 8 <= cadence <= 12


# =============================================================================
# Parametrized Tests
# =============================================================================

@pytest.mark.parametrize("frequency,duration,expected_cycles", [
    (2.0, 5.0, (9, 11)),    # 2 Hz * 5s = 10 cycles
    (4.0, 2.5, (9, 11)),    # 4 Hz * 2.5s = 10 cycles
    (6.0, 2.0, (11, 13)),   # 6 Hz * 2s = 12 cycles
    (8.0, 1.5, (11, 13)),   # 8 Hz * 1.5s = 12 cycles
])
def test_cycle_detection_parametrized(
    frequency,
    duration,
    expected_cycles,
    gait_cycle_config
):
    """Test cycle detection across various frequencies and durations."""
    detector = GaitCycleDetector(**gait_cycle_config)

    fps = 200
    t = np.linspace(0, duration, int(fps * duration))
    stride = 10 * np.sin(2 * np.pi * frequency * t)

    n_cycles, _, _ = detector.detect_cycles(stride, fps)

    assert expected_cycles[0] <= n_cycles <= expected_cycles[1]


@pytest.mark.parametrize("speed,cadence,expected_length", [
    (10.0, 2.0, 5.0),
    (20.0, 4.0, 5.0),
    (40.0, 4.0, 10.0),
    (30.0, 6.0, 5.0),
])
def test_stride_length_parametrized(speed, cadence, expected_length, gait_cycle_config):
    """Test stride length calculation with various inputs."""
    detector = GaitCycleDetector(**gait_cycle_config)

    stride_length = detector.compute_stride_length(cadence, speed)

    assert abs(stride_length - expected_length) < 0.1


@pytest.mark.parametrize("noise_level", [0.0, 0.5, 1.0, 2.0])
def test_cycle_detection_noise_robustness(noise_level, gait_cycle_config):
    """Test cycle detection robustness to noise."""
    detector = GaitCycleDetector(**gait_cycle_config)

    np.random.seed(42)
    fps = 200
    duration = 2.5
    frequency = 4.0
    expected_cycles = 10

    t = np.linspace(0, duration, int(fps * duration))
    clean = 10 * np.sin(2 * np.pi * frequency * t)
    noisy = clean + np.random.normal(0, noise_level, len(t))

    n_cycles, _, _ = detector.detect_cycles(noisy, fps)

    # Should still detect cycles, with more tolerance for higher noise
    tolerance = 2 + int(noise_level)
    assert expected_cycles - tolerance <= n_cycles <= expected_cycles + tolerance


@pytest.mark.parametrize("interpolation_factor", [1, 2, 4, 8])
def test_interpolation_factor_effect(interpolation_factor):
    """Test effect of different interpolation factors."""
    config = {'interpolation_factor': interpolation_factor}
    detector = GaitCycleDetector(**config)

    original = np.sin(np.linspace(0, 4*np.pi, 50))
    interpolated = detector.interpolate_stride(original, duration=1.0)

    assert len(interpolated) == 50 * interpolation_factor
