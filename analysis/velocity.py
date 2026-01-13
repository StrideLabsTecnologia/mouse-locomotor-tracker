"""
Velocity Analysis Module
========================

Calculates instantaneous velocity, acceleration, and detects
locomotor events (drag/recovery) from mouse tracking data.

Based on Locomotor-Allodi2021 methodology with enhancements.

Author: Stride Labs - Mouse Locomotor Tracker
"""

import numpy as np
from scipy import signal
from scipy.ndimage import uniform_filter1d
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass

from .metrics import VelocityMetrics, LocomotionPhase


class VelocityAnalyzer:
    """
    Analyzes velocity and acceleration from coordinate time series.

    This class computes instantaneous velocity from (x, y) coordinates,
    applies smoothing filters, calculates acceleration, and detects
    drag/recovery events indicative of locomotor impairment.

    Attributes:
        frame_rate: Video frame rate in Hz (default: 30)
        pixel_to_mm: Conversion factor from pixels to mm (default: 1.0)
        smoothing_window: Window size for moving average filter (default: 5)
        drag_velocity_threshold: Velocity below which drag is detected
        recovery_velocity_threshold: Velocity above which recovery is detected

    Example:
        >>> analyzer = VelocityAnalyzer(frame_rate=30, pixel_to_mm=0.1)
        >>> metrics = analyzer.analyze(x_coords, y_coords)
        >>> print(f"Mean velocity: {metrics.mean_velocity:.2f} mm/s")
    """

    def __init__(
        self,
        frame_rate: float = 30.0,
        pixel_to_mm: float = 1.0,
        smoothing_window: int = 5,
        drag_velocity_threshold: float = 5.0,
        recovery_velocity_threshold: float = 15.0,
    ):
        """
        Initialize the VelocityAnalyzer.

        Args:
            frame_rate: Video frame rate in Hz
            pixel_to_mm: Conversion factor from pixels to millimeters
            smoothing_window: Window size for moving average smoothing
            drag_velocity_threshold: Velocity (mm/s) below which drag is detected
            recovery_velocity_threshold: Velocity (mm/s) above which recovery is detected
        """
        if frame_rate <= 0:
            raise ValueError("frame_rate must be positive")
        if pixel_to_mm <= 0:
            raise ValueError("pixel_to_mm must be positive")
        if smoothing_window < 1:
            raise ValueError("smoothing_window must be at least 1")

        self.frame_rate = frame_rate
        self.pixel_to_mm = pixel_to_mm
        self.smoothing_window = smoothing_window
        self.drag_velocity_threshold = drag_velocity_threshold
        self.recovery_velocity_threshold = recovery_velocity_threshold
        self._dt = 1.0 / frame_rate  # Time step in seconds

    def _validate_coordinates(
        self, x: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Validate and preprocess coordinate arrays.

        Args:
            x: X coordinates array
            y: Y coordinates array

        Returns:
            Tuple of validated (x, y) arrays

        Raises:
            ValueError: If arrays are invalid or incompatible
        """
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)

        if x.shape != y.shape:
            raise ValueError(
                f"x and y must have same shape: {x.shape} vs {y.shape}"
            )

        if x.ndim != 1:
            raise ValueError(f"Coordinates must be 1D, got {x.ndim}D")

        if len(x) < 3:
            raise ValueError(
                f"Need at least 3 points for velocity calculation, got {len(x)}"
            )

        return x, y

    def _handle_missing_data(
        self, x: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Handle NaN values in coordinate data.

        Uses linear interpolation for small gaps and marks large gaps.

        Args:
            x: X coordinates with potential NaN values
            y: Y coordinates with potential NaN values

        Returns:
            Tuple of (interpolated_x, interpolated_y, valid_mask)
        """
        # Create mask of valid points
        valid_mask = ~(np.isnan(x) | np.isnan(y))

        if not np.any(valid_mask):
            raise ValueError("All coordinate values are NaN")

        if np.all(valid_mask):
            return x.copy(), y.copy(), valid_mask

        # Interpolate missing values
        indices = np.arange(len(x))
        valid_indices = indices[valid_mask]

        x_interp = np.interp(indices, valid_indices, x[valid_mask])
        y_interp = np.interp(indices, valid_indices, y[valid_mask])

        return x_interp, y_interp, valid_mask

    def _calculate_displacement(
        self, x: np.ndarray, y: np.ndarray
    ) -> np.ndarray:
        """
        Calculate frame-to-frame displacement.

        Args:
            x: X coordinates
            y: Y coordinates

        Returns:
            Array of displacements in pixels
        """
        dx = np.diff(x)
        dy = np.diff(y)
        displacement = np.sqrt(dx**2 + dy**2)

        # Pad to maintain original length
        return np.concatenate([[0], displacement])

    def _calculate_velocity(self, displacement: np.ndarray) -> np.ndarray:
        """
        Calculate instantaneous velocity from displacement.

        Args:
            displacement: Frame-to-frame displacement in pixels

        Returns:
            Velocity in mm/s
        """
        # Convert pixels to mm and divide by time step
        velocity = (displacement * self.pixel_to_mm) / self._dt
        return velocity

    def _smooth_signal(
        self, signal_data: np.ndarray, window: Optional[int] = None
    ) -> np.ndarray:
        """
        Apply moving average smoothing to a signal.

        Args:
            signal_data: Input signal array
            window: Smoothing window size (default: self.smoothing_window)

        Returns:
            Smoothed signal
        """
        if window is None:
            window = self.smoothing_window

        if window <= 1:
            return signal_data.copy()

        # Use uniform filter for efficient moving average
        return uniform_filter1d(signal_data, size=window, mode='nearest')

    def _calculate_acceleration(self, velocity: np.ndarray) -> np.ndarray:
        """
        Calculate acceleration from velocity.

        Uses central difference for interior points and
        forward/backward difference at boundaries.

        Args:
            velocity: Velocity time series in mm/s

        Returns:
            Acceleration in mm/s^2
        """
        n = len(velocity)
        acceleration = np.zeros(n)

        if n < 2:
            return acceleration

        # Central difference for interior points
        if n > 2:
            acceleration[1:-1] = (velocity[2:] - velocity[:-2]) / (2 * self._dt)

        # Forward difference for first point
        acceleration[0] = (velocity[1] - velocity[0]) / self._dt

        # Backward difference for last point
        acceleration[-1] = (velocity[-1] - velocity[-2]) / self._dt

        return acceleration

    def _detect_drag_events(
        self, velocity: np.ndarray
    ) -> Tuple[List[int], List[int]]:
        """
        Detect drag and recovery events based on velocity thresholds.

        Drag events occur when velocity drops below threshold,
        recovery events when velocity rises above recovery threshold.

        Args:
            velocity: Velocity time series

        Returns:
            Tuple of (drag_event_indices, recovery_event_indices)
        """
        drag_events = []
        recovery_events = []

        in_drag = False
        drag_start = 0

        for i, v in enumerate(velocity):
            if not in_drag and v < self.drag_velocity_threshold:
                # Start of drag event
                in_drag = True
                drag_start = i
                drag_events.append(i)

            elif in_drag and v > self.recovery_velocity_threshold:
                # Recovery from drag
                in_drag = False
                recovery_events.append(i)

        return drag_events, recovery_events

    def _classify_locomotion_phase(
        self,
        velocity: np.ndarray,
        drag_events: List[int],
        recovery_events: List[int],
    ) -> np.ndarray:
        """
        Classify each frame into a locomotion phase.

        Args:
            velocity: Velocity time series
            drag_events: Indices of drag event starts
            recovery_events: Indices of recovery events

        Returns:
            Array of LocomotionPhase values for each frame
        """
        n = len(velocity)
        phases = np.full(n, LocomotionPhase.STANCE, dtype=object)

        # Mark drag periods
        drag_idx = 0
        recovery_idx = 0

        for i in range(n):
            # Check if we're in a drag period
            if drag_idx < len(drag_events):
                drag_start = drag_events[drag_idx]

                # Find corresponding recovery
                recovery_end = n  # Default to end if no recovery
                if recovery_idx < len(recovery_events):
                    recovery_end = recovery_events[recovery_idx]

                if drag_start <= i < recovery_end:
                    phases[i] = LocomotionPhase.DRAG
                elif i == recovery_end:
                    phases[i] = LocomotionPhase.RECOVERY
                    recovery_idx += 1
                    drag_idx += 1

        return phases

    def analyze(
        self,
        x: np.ndarray,
        y: np.ndarray,
        compute_phases: bool = True,
    ) -> VelocityMetrics:
        """
        Perform complete velocity analysis on coordinate data.

        Args:
            x: X coordinates (pixels)
            y: Y coordinates (pixels)
            compute_phases: Whether to compute locomotion phases

        Returns:
            VelocityMetrics containing all velocity analysis results
        """
        # Validate input
        x, y = self._validate_coordinates(x, y)

        # Handle missing data
        x_clean, y_clean, valid_mask = self._handle_missing_data(x, y)

        # Calculate displacement and velocity
        displacement = self._calculate_displacement(x_clean, y_clean)
        velocity = self._calculate_velocity(displacement)

        # Smooth velocity
        smoothed_velocity = self._smooth_signal(velocity)

        # Calculate acceleration from smoothed velocity
        acceleration = self._calculate_acceleration(smoothed_velocity)

        # Detect drag/recovery events
        drag_events, recovery_events = self._detect_drag_events(smoothed_velocity)

        # Compute statistics (excluding the first frame which has 0 velocity)
        valid_velocity = smoothed_velocity[1:]
        if len(valid_velocity) > 0:
            mean_vel = float(np.nanmean(valid_velocity))
            max_vel = float(np.nanmax(valid_velocity))
            min_vel = float(np.nanmin(valid_velocity))
            std_vel = float(np.nanstd(valid_velocity))
        else:
            mean_vel = max_vel = min_vel = std_vel = 0.0

        valid_accel = acceleration[1:-1]
        if len(valid_accel) > 0:
            mean_accel = float(np.nanmean(np.abs(valid_accel)))
            max_accel = float(np.nanmax(np.abs(valid_accel)))
        else:
            mean_accel = max_accel = 0.0

        return VelocityMetrics(
            mean_velocity=mean_vel,
            max_velocity=max_vel,
            min_velocity=min_vel,
            velocity_std=std_vel,
            mean_acceleration=mean_accel,
            max_acceleration=max_accel,
            velocity_profile=velocity,
            acceleration_profile=acceleration,
            smoothed_velocity=smoothed_velocity,
            drag_events=drag_events,
            recovery_events=recovery_events,
            drag_count=len(drag_events),
        )

    def get_speed_profile(
        self,
        x: np.ndarray,
        y: np.ndarray,
        smooth: bool = True,
    ) -> np.ndarray:
        """
        Get the speed profile (magnitude of velocity).

        Convenience method for quick speed extraction.

        Args:
            x: X coordinates
            y: Y coordinates
            smooth: Whether to apply smoothing

        Returns:
            Speed profile in mm/s
        """
        x, y = self._validate_coordinates(x, y)
        x_clean, y_clean, _ = self._handle_missing_data(x, y)

        displacement = self._calculate_displacement(x_clean, y_clean)
        velocity = self._calculate_velocity(displacement)

        if smooth:
            return self._smooth_signal(velocity)
        return velocity

    def get_acceleration_profile(
        self,
        x: np.ndarray,
        y: np.ndarray,
        smooth: bool = True,
    ) -> np.ndarray:
        """
        Get the acceleration profile.

        Convenience method for quick acceleration extraction.

        Args:
            x: X coordinates
            y: Y coordinates
            smooth: Whether to smooth velocity before differentiation

        Returns:
            Acceleration profile in mm/s^2
        """
        speed = self.get_speed_profile(x, y, smooth=smooth)
        return self._calculate_acceleration(speed)

    def analyze_segment(
        self,
        x: np.ndarray,
        y: np.ndarray,
        start_frame: int,
        end_frame: int,
    ) -> VelocityMetrics:
        """
        Analyze velocity for a specific segment of the recording.

        Args:
            x: Full X coordinate array
            y: Full Y coordinate array
            start_frame: Start frame index
            end_frame: End frame index (exclusive)

        Returns:
            VelocityMetrics for the specified segment
        """
        x_segment = x[start_frame:end_frame]
        y_segment = y[start_frame:end_frame]
        return self.analyze(x_segment, y_segment)

    def compute_velocity_histogram(
        self,
        velocity: np.ndarray,
        bins: int = 50,
        range_mm_s: Optional[Tuple[float, float]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute histogram of velocity distribution.

        Args:
            velocity: Velocity time series
            bins: Number of histogram bins
            range_mm_s: Optional (min, max) range for histogram

        Returns:
            Tuple of (counts, bin_edges)
        """
        valid_velocity = velocity[~np.isnan(velocity)]

        if range_mm_s is None:
            range_mm_s = (0, np.percentile(valid_velocity, 99))

        counts, bin_edges = np.histogram(
            valid_velocity, bins=bins, range=range_mm_s
        )
        return counts, bin_edges

    def detect_velocity_peaks(
        self,
        velocity: np.ndarray,
        min_prominence: float = 10.0,
        min_distance_frames: int = 5,
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Detect peaks in velocity profile (step events).

        Args:
            velocity: Velocity time series
            min_prominence: Minimum peak prominence in mm/s
            min_distance_frames: Minimum frames between peaks

        Returns:
            Tuple of (peak_indices, peak_properties)
        """
        peaks, properties = signal.find_peaks(
            velocity,
            prominence=min_prominence,
            distance=min_distance_frames,
        )
        return peaks, properties


def create_velocity_analyzer(
    frame_rate: float = 30.0,
    pixel_to_mm: float = 1.0,
    config: Optional[Dict[str, Any]] = None,
) -> VelocityAnalyzer:
    """
    Factory function to create a VelocityAnalyzer with configuration.

    Args:
        frame_rate: Video frame rate
        pixel_to_mm: Pixel to mm conversion
        config: Optional configuration dictionary

    Returns:
        Configured VelocityAnalyzer instance
    """
    if config is None:
        config = {}

    return VelocityAnalyzer(
        frame_rate=frame_rate,
        pixel_to_mm=pixel_to_mm,
        smoothing_window=config.get('smoothing_window', 5),
        drag_velocity_threshold=config.get('drag_velocity_threshold', 5.0),
        recovery_velocity_threshold=config.get('recovery_velocity_threshold', 15.0),
    )
