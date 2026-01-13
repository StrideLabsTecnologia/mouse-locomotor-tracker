"""
Gait Cycle Detection Module
===========================

Detects gait cycles from locomotion data and computes
temporal-spatial gait parameters.

Based on Locomotor-Allodi2021 methodology with enhancements.

Author: Stride Labs - Mouse Locomotor Tracker
"""

import numpy as np
from scipy import signal
from scipy.ndimage import uniform_filter1d
from typing import Optional, Tuple, List, Dict, Any, Union
from dataclasses import dataclass
import warnings

from .metrics import GaitCycleInfo, GaitMetrics, LocomotionPhase


class GaitCycleDetector:
    """
    Detects and analyzes gait cycles from locomotion data.

    Gait cycles are typically detected from:
    - Toe/paw vertical position (foot strikes and toe-offs)
    - Velocity profiles (peaks correspond to swing phases)
    - Ground reaction force (if available)

    A complete gait cycle is defined as:
    - Heel strike to next heel strike (stance-centric)
    - Or toe-off to next toe-off (swing-centric)

    Attributes:
        frame_rate: Video frame rate in Hz
        pixel_to_mm: Pixel to mm conversion factor
        detection_method: 'peaks', 'threshold', or 'derivative'
        min_cycle_duration: Minimum cycle duration in seconds
        max_cycle_duration: Maximum cycle duration in seconds

    Example:
        >>> detector = GaitCycleDetector(frame_rate=30)
        >>> metrics = detector.detect_cycles(toe_y_position)
        >>> print(f"Cadence: {metrics.cadence:.2f} Hz")
        >>> print(f"Stride length: {metrics.mean_stride_length:.2f} mm")
    """

    def __init__(
        self,
        frame_rate: float = 30.0,
        pixel_to_mm: float = 1.0,
        detection_method: str = 'peaks',
        min_cycle_duration: float = 0.1,  # 100ms minimum
        max_cycle_duration: float = 2.0,   # 2s maximum
    ):
        """
        Initialize the GaitCycleDetector.

        Args:
            frame_rate: Video frame rate in Hz
            pixel_to_mm: Pixel to mm conversion
            detection_method: Method for cycle detection
            min_cycle_duration: Minimum valid cycle duration (seconds)
            max_cycle_duration: Maximum valid cycle duration (seconds)
        """
        if frame_rate <= 0:
            raise ValueError("frame_rate must be positive")
        if min_cycle_duration >= max_cycle_duration:
            raise ValueError(
                "min_cycle_duration must be less than max_cycle_duration"
            )

        self.frame_rate = frame_rate
        self.pixel_to_mm = pixel_to_mm
        self.detection_method = detection_method
        self.min_cycle_duration = min_cycle_duration
        self.max_cycle_duration = max_cycle_duration

        # Derived parameters
        self._dt = 1.0 / frame_rate
        self._min_cycle_frames = int(min_cycle_duration * frame_rate)
        self._max_cycle_frames = int(max_cycle_duration * frame_rate)

    def _preprocess_signal(
        self,
        signal_data: np.ndarray,
        smooth_window: int = 5,
    ) -> np.ndarray:
        """
        Preprocess signal for cycle detection.

        Args:
            signal_data: Raw signal
            smooth_window: Smoothing window size

        Returns:
            Preprocessed signal
        """
        signal_data = np.asarray(signal_data, dtype=np.float64)

        # Handle NaN values
        nan_mask = np.isnan(signal_data)
        if np.any(nan_mask):
            # Interpolate NaN values
            valid_indices = np.where(~nan_mask)[0]
            if len(valid_indices) < 2:
                raise ValueError("Not enough valid data points")

            all_indices = np.arange(len(signal_data))
            signal_data = np.interp(
                all_indices, valid_indices, signal_data[valid_indices]
            )

        # Apply smoothing
        if smooth_window > 1:
            signal_data = uniform_filter1d(
                signal_data, size=smooth_window, mode='nearest'
            )

        return signal_data

    def _detect_peaks_method(
        self,
        signal_data: np.ndarray,
        prominence: Optional[float] = None,
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Detect gait cycles using peak detection.

        Peaks in toe vertical position indicate foot-off events.
        Valleys indicate foot-strike events.

        Args:
            signal_data: Preprocessed signal (e.g., toe y-position)
            prominence: Minimum peak prominence

        Returns:
            Tuple of (peak_indices, properties)
        """
        if prominence is None:
            # Auto-calculate prominence based on signal range
            signal_range = np.nanmax(signal_data) - np.nanmin(signal_data)
            prominence = signal_range * 0.2  # 20% of range

        peaks, properties = signal.find_peaks(
            signal_data,
            prominence=prominence,
            distance=self._min_cycle_frames,
        )

        return peaks, properties

    def _detect_threshold_method(
        self,
        signal_data: np.ndarray,
        threshold: Optional[float] = None,
    ) -> np.ndarray:
        """
        Detect gait cycles using threshold crossing.

        Detects when signal crosses a threshold from below.

        Args:
            signal_data: Preprocessed signal
            threshold: Crossing threshold

        Returns:
            Array of crossing indices
        """
        if threshold is None:
            threshold = np.nanmean(signal_data)

        # Find upward crossings
        above = signal_data > threshold
        crossings = np.where(np.diff(above.astype(int)) == 1)[0] + 1

        return crossings

    def _detect_derivative_method(
        self,
        signal_data: np.ndarray,
    ) -> np.ndarray:
        """
        Detect gait cycles using derivative zero-crossings.

        Identifies where velocity changes sign (minima and maxima).

        Args:
            signal_data: Preprocessed signal

        Returns:
            Array of zero-crossing indices
        """
        # Compute derivative
        derivative = np.gradient(signal_data)

        # Smooth derivative
        derivative = uniform_filter1d(derivative, size=3, mode='nearest')

        # Find zero crossings (negative to positive = minima)
        positive = derivative > 0
        crossings = np.where(np.diff(positive.astype(int)) == 1)[0] + 1

        return crossings

    def _filter_cycles(
        self,
        cycle_starts: np.ndarray,
        cycle_ends: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Filter out invalid cycles based on duration constraints.

        Args:
            cycle_starts: Array of cycle start frames
            cycle_ends: Array of cycle end frames

        Returns:
            Tuple of (valid_starts, valid_ends)
        """
        valid_starts = []
        valid_ends = []

        for start, end in zip(cycle_starts, cycle_ends):
            duration_frames = end - start

            if (self._min_cycle_frames <= duration_frames <=
                self._max_cycle_frames):
                valid_starts.append(start)
                valid_ends.append(end)

        return np.array(valid_starts), np.array(valid_ends)

    def detect_cycles(
        self,
        signal_data: np.ndarray,
        x_position: Optional[np.ndarray] = None,
        prominence: Optional[float] = None,
        threshold: Optional[float] = None,
    ) -> GaitMetrics:
        """
        Detect gait cycles and compute metrics.

        Args:
            signal_data: Signal for cycle detection (e.g., toe y-position)
            x_position: X-position for stride length calculation
            prominence: Peak detection prominence (for 'peaks' method)
            threshold: Threshold value (for 'threshold' method)

        Returns:
            GaitMetrics containing all gait analysis results
        """
        # Preprocess signal
        signal_clean = self._preprocess_signal(signal_data)

        # Detect cycle events based on method
        if self.detection_method == 'peaks':
            # For vertical toe position, peaks = foot-off events
            # Use valleys (inverted peaks) for foot-strike
            valleys, _ = signal.find_peaks(
                -signal_clean,
                prominence=prominence,
                distance=self._min_cycle_frames,
            )
            cycle_events = valleys
        elif self.detection_method == 'threshold':
            cycle_events = self._detect_threshold_method(
                signal_clean, threshold
            )
        elif self.detection_method == 'derivative':
            cycle_events = self._detect_derivative_method(signal_clean)
        else:
            raise ValueError(f"Unknown method: {self.detection_method}")

        # Need at least 2 events for one cycle
        if len(cycle_events) < 2:
            return self._create_empty_metrics()

        # Create cycles from consecutive events
        cycle_starts = cycle_events[:-1]
        cycle_ends = cycle_events[1:]

        # Filter invalid cycles
        cycle_starts, cycle_ends = self._filter_cycles(cycle_starts, cycle_ends)

        if len(cycle_starts) == 0:
            return self._create_empty_metrics()

        # Create cycle info objects
        cycles = []
        stride_lengths = []

        for start, end in zip(cycle_starts, cycle_ends):
            duration_frames = end - start
            duration_seconds = duration_frames * self._dt

            # Calculate stride length if x-position provided
            stride_length = 0.0
            if x_position is not None:
                stride_length = abs(
                    x_position[end] - x_position[start]
                ) * self.pixel_to_mm
                stride_lengths.append(stride_length)

            cycle = GaitCycleInfo(
                start_frame=int(start),
                end_frame=int(end),
                duration_frames=int(duration_frames),
                duration_seconds=float(duration_seconds),
                stride_length=stride_length,
            )
            cycles.append(cycle)

        # Compute aggregate metrics
        num_cycles = len(cycles)
        durations = [c.duration_seconds for c in cycles]
        mean_duration = float(np.mean(durations))
        duration_cv = float(np.std(durations) / mean_duration * 100) if mean_duration > 0 else 0.0

        # Cadence (cycles per second)
        cadence = 1.0 / mean_duration if mean_duration > 0 else 0.0

        # Step frequency (steps per minute)
        step_frequency = cadence * 60.0

        # Stride length metrics
        if stride_lengths:
            mean_stride = float(np.mean(stride_lengths))
            stride_cv = float(np.std(stride_lengths) / mean_stride * 100) if mean_stride > 0 else 0.0
        else:
            mean_stride = 0.0
            stride_cv = 0.0

        return GaitMetrics(
            cycles=cycles,
            num_cycles=num_cycles,
            cadence=cadence,
            mean_stride_length=mean_stride,
            stride_length_variability=stride_cv,
            mean_cycle_duration=mean_duration,
            cycle_duration_variability=duration_cv,
            mean_duty_factor=0.0,  # Calculated separately
            step_frequency=step_frequency,
            symmetry_index=1.0,  # Calculated with left/right comparison
        )

    def _create_empty_metrics(self) -> GaitMetrics:
        """Create empty GaitMetrics when no cycles detected."""
        return GaitMetrics(
            cycles=[],
            num_cycles=0,
            cadence=0.0,
            mean_stride_length=0.0,
            stride_length_variability=0.0,
            mean_cycle_duration=0.0,
            cycle_duration_variability=0.0,
            mean_duty_factor=0.0,
            step_frequency=0.0,
            symmetry_index=1.0,
        )

    def detect_stance_swing_phases(
        self,
        toe_y: np.ndarray,
        threshold_percentile: float = 25.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect stance and swing phases from toe vertical position.

        In image coordinates:
        - Larger y = lower position = closer to ground = STANCE
        - Smaller y = higher position = in air = SWING

        Args:
            toe_y: Toe vertical position
            threshold_percentile: Percentile for stance/swing threshold

        Returns:
            Tuple of (stance_mask, swing_mask) as boolean arrays
        """
        toe_y = self._preprocess_signal(toe_y)

        # Calculate threshold based on percentile
        threshold = np.percentile(toe_y, 100 - threshold_percentile)

        # Stance = toe near ground (larger y in image coords)
        stance_mask = toe_y > threshold
        swing_mask = ~stance_mask

        return stance_mask, swing_mask

    def calculate_duty_factor(
        self,
        stance_mask: np.ndarray,
        cycle_starts: np.ndarray,
        cycle_ends: np.ndarray,
    ) -> List[float]:
        """
        Calculate duty factor for each gait cycle.

        Duty factor = stance duration / total cycle duration
        Normal mouse: ~0.6-0.7 during walking

        Args:
            stance_mask: Boolean mask of stance frames
            cycle_starts: Frame indices of cycle starts
            cycle_ends: Frame indices of cycle ends

        Returns:
            List of duty factors for each cycle
        """
        duty_factors = []

        for start, end in zip(cycle_starts, cycle_ends):
            cycle_stance = stance_mask[start:end]
            if len(cycle_stance) > 0:
                duty_factor = np.sum(cycle_stance) / len(cycle_stance)
                duty_factors.append(float(duty_factor))

        return duty_factors

    def get_cadence(
        self,
        signal_data: np.ndarray,
        method: str = 'autocorrelation',
    ) -> float:
        """
        Calculate step cadence (frequency) from signal.

        Args:
            signal_data: Signal for cadence estimation
            method: 'autocorrelation' or 'fft'

        Returns:
            Cadence in Hz (cycles per second)
        """
        signal_clean = self._preprocess_signal(signal_data)

        if method == 'autocorrelation':
            return self._cadence_autocorrelation(signal_clean)
        elif method == 'fft':
            return self._cadence_fft(signal_clean)
        else:
            raise ValueError(f"Unknown method: {method}")

    def _cadence_autocorrelation(self, signal_data: np.ndarray) -> float:
        """
        Estimate cadence using autocorrelation.

        Args:
            signal_data: Preprocessed signal

        Returns:
            Cadence in Hz
        """
        # Remove mean
        signal_centered = signal_data - np.mean(signal_data)

        # Compute autocorrelation
        n = len(signal_centered)
        autocorr = np.correlate(signal_centered, signal_centered, mode='full')
        autocorr = autocorr[n-1:]  # Keep only positive lags
        autocorr = autocorr / autocorr[0]  # Normalize

        # Find first peak after initial decay
        min_lag = self._min_cycle_frames
        max_lag = min(self._max_cycle_frames, len(autocorr) - 1)

        if max_lag <= min_lag:
            return 0.0

        peaks, _ = signal.find_peaks(
            autocorr[min_lag:max_lag],
            prominence=0.1,
        )

        if len(peaks) == 0:
            return 0.0

        # First peak corresponds to fundamental period
        period_frames = peaks[0] + min_lag
        period_seconds = period_frames * self._dt
        cadence = 1.0 / period_seconds if period_seconds > 0 else 0.0

        return float(cadence)

    def _cadence_fft(self, signal_data: np.ndarray) -> float:
        """
        Estimate cadence using FFT.

        Args:
            signal_data: Preprocessed signal

        Returns:
            Cadence in Hz
        """
        # Compute FFT
        n = len(signal_data)
        fft_result = np.fft.rfft(signal_data - np.mean(signal_data))
        freqs = np.fft.rfftfreq(n, self._dt)
        power = np.abs(fft_result) ** 2

        # Find frequency range corresponding to valid cycle durations
        min_freq = 1.0 / self.max_cycle_duration
        max_freq = 1.0 / self.min_cycle_duration

        # Mask valid frequency range
        valid_mask = (freqs >= min_freq) & (freqs <= max_freq)

        if not np.any(valid_mask):
            return 0.0

        # Find peak frequency
        valid_power = power.copy()
        valid_power[~valid_mask] = 0

        peak_idx = np.argmax(valid_power)
        cadence = freqs[peak_idx]

        return float(cadence)

    def get_stride_length(
        self,
        x_position: np.ndarray,
        cycle_starts: np.ndarray,
        cycle_ends: np.ndarray,
    ) -> Tuple[float, float, List[float]]:
        """
        Calculate stride length from x-position.

        Stride length = horizontal distance traveled in one gait cycle.

        Args:
            x_position: X-coordinates of tracked point
            cycle_starts: Frame indices of cycle starts
            cycle_ends: Frame indices of cycle ends

        Returns:
            Tuple of (mean_stride, std_stride, all_strides) in mm
        """
        x_position = np.asarray(x_position, dtype=np.float64)

        stride_lengths = []

        for start, end in zip(cycle_starts, cycle_ends):
            if start < len(x_position) and end < len(x_position):
                stride = abs(x_position[end] - x_position[start])
                stride_mm = stride * self.pixel_to_mm
                stride_lengths.append(stride_mm)

        if not stride_lengths:
            return 0.0, 0.0, []

        mean_stride = float(np.mean(stride_lengths))
        std_stride = float(np.std(stride_lengths))

        return mean_stride, std_stride, stride_lengths

    def calculate_symmetry_index(
        self,
        left_metrics: GaitMetrics,
        right_metrics: GaitMetrics,
    ) -> float:
        """
        Calculate gait symmetry between left and right sides.

        Symmetry Index = 1 - |left - right| / (0.5 * (left + right))

        A value of 1.0 indicates perfect symmetry.

        Args:
            left_metrics: GaitMetrics for left limb
            right_metrics: GaitMetrics for right limb

        Returns:
            Symmetry index (0 to 1)
        """
        left_stride = left_metrics.mean_stride_length
        right_stride = right_metrics.mean_stride_length

        if left_stride == 0 and right_stride == 0:
            return 1.0

        mean_stride = 0.5 * (left_stride + right_stride)

        if mean_stride == 0:
            return 1.0

        asymmetry = abs(left_stride - right_stride) / mean_stride
        symmetry = 1.0 - asymmetry

        return float(max(0.0, symmetry))

    def normalize_to_cycle_percent(
        self,
        signal_data: np.ndarray,
        cycle_starts: np.ndarray,
        cycle_ends: np.ndarray,
        n_points: int = 100,
    ) -> np.ndarray:
        """
        Normalize signal to percentage of gait cycle.

        Interpolates each cycle to a standard 0-100% representation.

        Args:
            signal_data: Signal to normalize
            cycle_starts: Frame indices of cycle starts
            cycle_ends: Frame indices of cycle ends
            n_points: Number of points in normalized cycle

        Returns:
            Array of shape (n_cycles, n_points)
        """
        normalized_cycles = []

        for start, end in zip(cycle_starts, cycle_ends):
            if end <= start:
                continue

            cycle_data = signal_data[start:end]
            if len(cycle_data) < 2:
                continue

            # Interpolate to standard length
            x_orig = np.linspace(0, 100, len(cycle_data))
            x_norm = np.linspace(0, 100, n_points)

            normalized = np.interp(x_norm, x_orig, cycle_data)
            normalized_cycles.append(normalized)

        if not normalized_cycles:
            return np.array([])

        return np.array(normalized_cycles)


def create_gait_detector(
    frame_rate: float = 30.0,
    pixel_to_mm: float = 1.0,
    config: Optional[Dict[str, Any]] = None,
) -> GaitCycleDetector:
    """
    Factory function to create a GaitCycleDetector.

    Args:
        frame_rate: Video frame rate
        pixel_to_mm: Pixel to mm conversion
        config: Optional configuration dictionary

    Returns:
        Configured GaitCycleDetector instance
    """
    if config is None:
        config = {}

    return GaitCycleDetector(
        frame_rate=frame_rate,
        pixel_to_mm=pixel_to_mm,
        detection_method=config.get('detection_method', 'peaks'),
        min_cycle_duration=config.get('min_cycle_duration', 0.1),
        max_cycle_duration=config.get('max_cycle_duration', 2.0),
    )


def detect_footfall_events(
    toe_y: np.ndarray,
    frame_rate: float = 30.0,
    min_prominence: Optional[float] = None,
) -> Dict[str, np.ndarray]:
    """
    Convenience function to detect foot-strike and toe-off events.

    Args:
        toe_y: Toe vertical position (y-coordinate)
        frame_rate: Video frame rate
        min_prominence: Minimum peak prominence

    Returns:
        Dictionary with 'foot_strike' and 'toe_off' indices
    """
    toe_y = np.asarray(toe_y, dtype=np.float64)

    # Smooth signal
    toe_smooth = uniform_filter1d(toe_y, size=5, mode='nearest')

    if min_prominence is None:
        signal_range = np.nanmax(toe_smooth) - np.nanmin(toe_smooth)
        min_prominence = signal_range * 0.15

    # Detect events
    # Foot strike: local maxima (toe reaches lowest point = highest y in image)
    # Toe off: local minima (toe lifts off = lowest y in image)

    foot_strikes, _ = signal.find_peaks(
        toe_smooth,
        prominence=min_prominence,
        distance=int(0.1 * frame_rate),  # Min 100ms between events
    )

    toe_offs, _ = signal.find_peaks(
        -toe_smooth,
        prominence=min_prominence,
        distance=int(0.1 * frame_rate),
    )

    return {
        'foot_strike': foot_strikes,
        'toe_off': toe_offs,
    }
