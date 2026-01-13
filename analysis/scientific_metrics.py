#!/usr/bin/env python3
"""
Scientific Gait Metrics Module
==============================

Comprehensive gait analysis metrics following standards from:
- CatWalk XT (Noldus)
- DigiGait (Mouse Specifics)
- Published literature (Nature, Frontiers, J. Neurophysiology)

Metrics Categories:
1. Spatiotemporal Parameters
2. Kinematic Parameters
3. Variability & Regularity
4. Statistical Analysis

References:
- Frontiers Behav. Neurosci. 2023 (CatWalk XT review)
- Sci. Rep. 2021 (Gait performance in mice)
- J. Neurophysiol. 2023 (High-throughput gait acquisition)

Author: Stride Labs
Version: 1.0.0
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from scipy import signal
from scipy.stats import sem, pearsonr
from scipy.ndimage import uniform_filter1d
import warnings


@dataclass
class StrideMetrics:
    """Metrics for a single stride cycle."""
    start_frame: int
    end_frame: int
    duration: float          # seconds
    length: float            # cm
    frequency: float         # Hz (1/duration)
    velocity: float          # cm/s (length/duration)


@dataclass
class GaitPhase:
    """Gait phase classification."""
    phase: str               # 'swing', 'stance', 'transition'
    start_frame: int
    end_frame: int
    duration: float          # seconds


@dataclass
class ScientificGaitMetrics:
    """
    Complete scientific gait metrics following CatWalk XT / DigiGait standards.

    All metrics include mean, SD, SEM, and CV where applicable.
    """
    # =========================================================================
    # Spatiotemporal Parameters
    # =========================================================================

    # Velocity (cm/s)
    velocity_mean: float = 0.0
    velocity_sd: float = 0.0
    velocity_sem: float = 0.0
    velocity_cv: float = 0.0        # Coefficient of Variation (%)
    velocity_max: float = 0.0
    velocity_min: float = 0.0

    # Stride Length (cm) - distance between consecutive steps
    stride_length_mean: float = 0.0
    stride_length_sd: float = 0.0
    stride_length_sem: float = 0.0
    stride_length_cv: float = 0.0

    # Stride Frequency / Cadence (steps/s)
    cadence_mean: float = 0.0
    cadence_sd: float = 0.0

    # Step Cycle Duration (s)
    step_cycle_mean: float = 0.0
    step_cycle_sd: float = 0.0

    # Swing Time (s) - paw in air
    swing_time_mean: float = 0.0
    swing_time_sd: float = 0.0
    swing_time_percent: float = 0.0  # % of step cycle

    # Stance Time (s) - paw on ground
    stance_time_mean: float = 0.0
    stance_time_sd: float = 0.0
    stance_time_percent: float = 0.0  # % of step cycle

    # Duty Factor (stance_time / step_cycle)
    duty_factor: float = 0.0
    duty_factor_sd: float = 0.0

    # Swing Speed (cm/s) - stride_length / swing_time
    swing_speed_mean: float = 0.0
    swing_speed_sd: float = 0.0

    # =========================================================================
    # Kinematic Parameters
    # =========================================================================

    # Acceleration (cm/s²)
    acceleration_mean: float = 0.0
    acceleration_sd: float = 0.0
    acceleration_max: float = 0.0

    # Jerk (cm/s³) - rate of change of acceleration
    jerk_mean: float = 0.0
    jerk_sd: float = 0.0
    jerk_max: float = 0.0

    # =========================================================================
    # Variability & Regularity
    # =========================================================================

    # Gait Regularity Index (0-100%)
    regularity_index: float = 0.0

    # Stride-to-stride variability
    stride_variability: float = 0.0

    # Velocity stability (inverse of CV)
    velocity_stability: float = 0.0

    # =========================================================================
    # Distance & Duration
    # =========================================================================

    total_distance: float = 0.0      # cm
    total_duration: float = 0.0      # s
    active_duration: float = 0.0     # s (excluding rest periods)
    rest_duration: float = 0.0       # s

    # =========================================================================
    # Activity Classification
    # =========================================================================

    time_resting: float = 0.0        # % of time
    time_walking: float = 0.0        # % of time
    time_trotting: float = 0.0       # % of time
    time_galloping: float = 0.0      # % of time

    n_bouts_walk: int = 0
    n_bouts_trot: int = 0
    n_bouts_gallop: int = 0

    mean_bout_duration: float = 0.0  # s

    # =========================================================================
    # Quality Metrics
    # =========================================================================

    tracking_rate: float = 0.0       # %
    n_frames_total: int = 0
    n_frames_tracked: int = 0
    n_strides_detected: int = 0
    data_quality_score: float = 0.0  # 0-100

    # Raw data for further analysis
    strides: List[StrideMetrics] = field(default_factory=list)
    velocity_timeseries: np.ndarray = field(default_factory=lambda: np.array([]))
    acceleration_timeseries: np.ndarray = field(default_factory=lambda: np.array([]))


class ScientificGaitAnalyzer:
    """
    Scientific-grade gait analysis following published standards.

    Implements metrics from:
    - CatWalk XT (Noldus Information Technology)
    - DigiGait (Mouse Specifics, Inc.)
    - Published neuroscience literature

    Example:
        >>> analyzer = ScientificGaitAnalyzer(fps=30.0, cm_per_pixel=0.03125)
        >>> metrics = analyzer.analyze(positions, timestamps)
        >>> print(f"Stride Length: {metrics.stride_length_mean:.2f} ± {metrics.stride_length_sd:.2f} cm")
    """

    def __init__(
        self,
        fps: float = 30.0,
        cm_per_pixel: float = 0.03125,
        velocity_threshold_rest: float = 2.0,      # cm/s
        velocity_threshold_walk: float = 10.0,     # cm/s
        velocity_threshold_trot: float = 25.0,     # cm/s
        smoothing_window: int = 5,
        min_bout_frames: int = 5,
    ):
        """
        Initialize analyzer with calibration parameters.

        Args:
            fps: Video frame rate (Hz)
            cm_per_pixel: Spatial calibration (cm/pixel)
            velocity_threshold_rest: Velocity below this = resting (cm/s)
            velocity_threshold_walk: Velocity below this = walking (cm/s)
            velocity_threshold_trot: Velocity below this = trotting (cm/s)
            smoothing_window: Window size for velocity smoothing
            min_bout_frames: Minimum frames to count as activity bout
        """
        self.fps = fps
        self.dt = 1.0 / fps
        self.cm_per_pixel = cm_per_pixel

        self.thresh_rest = velocity_threshold_rest
        self.thresh_walk = velocity_threshold_walk
        self.thresh_trot = velocity_threshold_trot

        self.smoothing_window = smoothing_window
        self.min_bout_frames = min_bout_frames

    def analyze(
        self,
        positions: np.ndarray,
        timestamps: Optional[np.ndarray] = None,
    ) -> ScientificGaitMetrics:
        """
        Perform complete scientific gait analysis.

        Args:
            positions: Array of (x, y) positions in pixels, shape (N, 2)
            timestamps: Optional array of timestamps in seconds, shape (N,)

        Returns:
            ScientificGaitMetrics with all computed metrics
        """
        if len(positions) < 10:
            warnings.warn("Insufficient data points for analysis")
            return ScientificGaitMetrics()

        positions = np.array(positions)

        if timestamps is None:
            timestamps = np.arange(len(positions)) / self.fps

        # Convert to cm
        positions_cm = positions * self.cm_per_pixel

        # Calculate velocity
        velocity = self._calculate_velocity(positions_cm, timestamps)
        velocity_smooth = self._smooth(velocity)

        # Calculate acceleration and jerk
        acceleration = self._calculate_acceleration(velocity_smooth, timestamps)
        jerk = self._calculate_jerk(acceleration, timestamps)

        # Detect strides
        strides = self._detect_strides(positions_cm, velocity_smooth, timestamps)

        # Classify activity
        activity_labels = self._classify_activity(velocity_smooth)

        # Calculate all metrics
        metrics = self._compute_all_metrics(
            positions_cm=positions_cm,
            timestamps=timestamps,
            velocity=velocity_smooth,
            acceleration=acceleration,
            jerk=jerk,
            strides=strides,
            activity_labels=activity_labels,
        )

        return metrics

    def _calculate_velocity(
        self,
        positions_cm: np.ndarray,
        timestamps: np.ndarray,
    ) -> np.ndarray:
        """Calculate instantaneous velocity from positions."""
        if len(positions_cm) < 2:
            return np.array([0.0])

        # Displacement
        dx = np.diff(positions_cm[:, 0])
        dy = np.diff(positions_cm[:, 1])
        displacement = np.sqrt(dx**2 + dy**2)

        # Time difference
        dt = np.diff(timestamps)
        dt[dt == 0] = self.dt  # Avoid division by zero

        # Velocity
        velocity = displacement / dt

        # Prepend 0 to match length
        velocity = np.concatenate([[0], velocity])

        # Cap unrealistic values (artifacts)
        velocity = np.clip(velocity, 0, 100)

        return velocity

    def _calculate_acceleration(
        self,
        velocity: np.ndarray,
        timestamps: np.ndarray,
    ) -> np.ndarray:
        """Calculate acceleration from velocity."""
        if len(velocity) < 2:
            return np.array([0.0])

        dv = np.diff(velocity)
        dt = np.diff(timestamps)
        dt[dt == 0] = self.dt

        acceleration = dv / dt
        acceleration = np.concatenate([[0], acceleration])

        return acceleration

    def _calculate_jerk(
        self,
        acceleration: np.ndarray,
        timestamps: np.ndarray,
    ) -> np.ndarray:
        """Calculate jerk (rate of change of acceleration)."""
        if len(acceleration) < 2:
            return np.array([0.0])

        da = np.diff(acceleration)
        dt = np.diff(timestamps)
        dt[dt == 0] = self.dt

        jerk = da / dt
        jerk = np.concatenate([[0], jerk])

        return jerk

    def _smooth(self, data: np.ndarray) -> np.ndarray:
        """Apply smoothing filter."""
        if len(data) < self.smoothing_window:
            return data
        return uniform_filter1d(data, size=self.smoothing_window)

    def _detect_strides(
        self,
        positions_cm: np.ndarray,
        velocity: np.ndarray,
        timestamps: np.ndarray,
    ) -> List[StrideMetrics]:
        """
        Detect individual stride cycles using velocity peaks.

        A stride is defined as the period between consecutive velocity minima
        during locomotion (not rest).
        """
        strides = []

        # Only analyze moving periods
        moving_mask = velocity > self.thresh_rest

        if not np.any(moving_mask):
            return strides

        # Find velocity peaks (mid-stride)
        peaks, properties = signal.find_peaks(
            velocity,
            height=self.thresh_rest,
            distance=int(self.fps * 0.1),  # Min 100ms between peaks
            prominence=1.0,
        )

        if len(peaks) < 2:
            return strides

        # Find valleys between peaks (stride boundaries)
        for i in range(len(peaks) - 1):
            start_idx = peaks[i]
            end_idx = peaks[i + 1]

            # Find minimum velocity between peaks
            segment = velocity[start_idx:end_idx]
            if len(segment) == 0:
                continue

            min_idx = start_idx + np.argmin(segment)

            # Calculate stride metrics
            duration = timestamps[end_idx] - timestamps[start_idx]
            if duration <= 0:
                continue

            # Stride length from displacement
            dx = positions_cm[end_idx, 0] - positions_cm[start_idx, 0]
            dy = positions_cm[end_idx, 1] - positions_cm[start_idx, 1]
            length = np.sqrt(dx**2 + dy**2)

            stride = StrideMetrics(
                start_frame=start_idx,
                end_frame=end_idx,
                duration=duration,
                length=length,
                frequency=1.0 / duration if duration > 0 else 0,
                velocity=length / duration if duration > 0 else 0,
            )
            strides.append(stride)

        return strides

    def _classify_activity(self, velocity: np.ndarray) -> np.ndarray:
        """
        Classify each frame into activity categories.

        Categories:
        - 0: Rest (< 2 cm/s)
        - 1: Walk (2-10 cm/s)
        - 2: Trot (10-25 cm/s)
        - 3: Gallop (> 25 cm/s)
        """
        labels = np.zeros(len(velocity), dtype=int)

        labels[velocity >= self.thresh_rest] = 1   # Walk
        labels[velocity >= self.thresh_walk] = 2   # Trot
        labels[velocity >= self.thresh_trot] = 3   # Gallop

        return labels

    def _count_bouts(
        self,
        activity_labels: np.ndarray,
        activity_code: int,
    ) -> Tuple[int, List[int]]:
        """Count activity bouts and their durations."""
        in_bout = False
        bout_count = 0
        bout_durations = []
        current_duration = 0

        for label in activity_labels:
            if label == activity_code:
                if not in_bout:
                    in_bout = True
                    current_duration = 1
                else:
                    current_duration += 1
            else:
                if in_bout:
                    if current_duration >= self.min_bout_frames:
                        bout_count += 1
                        bout_durations.append(current_duration)
                    in_bout = False
                    current_duration = 0

        # Handle last bout
        if in_bout and current_duration >= self.min_bout_frames:
            bout_count += 1
            bout_durations.append(current_duration)

        return bout_count, bout_durations

    def _compute_all_metrics(
        self,
        positions_cm: np.ndarray,
        timestamps: np.ndarray,
        velocity: np.ndarray,
        acceleration: np.ndarray,
        jerk: np.ndarray,
        strides: List[StrideMetrics],
        activity_labels: np.ndarray,
    ) -> ScientificGaitMetrics:
        """Compute all scientific metrics."""

        metrics = ScientificGaitMetrics()

        n_frames = len(velocity)
        metrics.n_frames_total = n_frames
        metrics.n_frames_tracked = np.sum(velocity > 0)
        metrics.tracking_rate = metrics.n_frames_tracked / n_frames * 100 if n_frames > 0 else 0

        # Filter valid velocity values
        valid_vel = velocity[velocity > 0.1]

        # =================================================================
        # Velocity Statistics
        # =================================================================
        if len(valid_vel) > 0:
            metrics.velocity_mean = float(np.mean(valid_vel))
            metrics.velocity_sd = float(np.std(valid_vel))
            metrics.velocity_sem = float(sem(valid_vel))
            metrics.velocity_cv = (metrics.velocity_sd / metrics.velocity_mean * 100) if metrics.velocity_mean > 0 else 0
            metrics.velocity_max = float(np.max(valid_vel))
            metrics.velocity_min = float(np.min(valid_vel))
            metrics.velocity_stability = 100 - metrics.velocity_cv

        # =================================================================
        # Stride Metrics
        # =================================================================
        metrics.n_strides_detected = len(strides)
        metrics.strides = strides

        if strides:
            stride_lengths = [s.length for s in strides]
            stride_durations = [s.duration for s in strides]
            stride_frequencies = [s.frequency for s in strides]

            metrics.stride_length_mean = float(np.mean(stride_lengths))
            metrics.stride_length_sd = float(np.std(stride_lengths))
            metrics.stride_length_sem = float(sem(stride_lengths)) if len(stride_lengths) > 1 else 0
            metrics.stride_length_cv = (metrics.stride_length_sd / metrics.stride_length_mean * 100) if metrics.stride_length_mean > 0 else 0

            metrics.step_cycle_mean = float(np.mean(stride_durations))
            metrics.step_cycle_sd = float(np.std(stride_durations))

            metrics.cadence_mean = float(np.mean(stride_frequencies))
            metrics.cadence_sd = float(np.std(stride_frequencies))

            # Stride variability (CV of stride lengths)
            metrics.stride_variability = metrics.stride_length_cv

            # Estimate swing/stance based on typical mouse gait (40% swing, 60% stance)
            # This is an approximation - precise measurement requires paw tracking
            metrics.swing_time_percent = 40.0
            metrics.stance_time_percent = 60.0
            metrics.swing_time_mean = metrics.step_cycle_mean * 0.4
            metrics.stance_time_mean = metrics.step_cycle_mean * 0.6
            metrics.duty_factor = 0.6

            if metrics.swing_time_mean > 0:
                metrics.swing_speed_mean = metrics.stride_length_mean / metrics.swing_time_mean

        # =================================================================
        # Acceleration & Jerk
        # =================================================================
        valid_acc = acceleration[np.abs(acceleration) < 1000]  # Filter artifacts
        if len(valid_acc) > 0:
            metrics.acceleration_mean = float(np.mean(np.abs(valid_acc)))
            metrics.acceleration_sd = float(np.std(valid_acc))
            metrics.acceleration_max = float(np.max(np.abs(valid_acc)))

        valid_jerk = jerk[np.abs(jerk) < 10000]
        if len(valid_jerk) > 0:
            metrics.jerk_mean = float(np.mean(np.abs(valid_jerk)))
            metrics.jerk_sd = float(np.std(valid_jerk))
            metrics.jerk_max = float(np.max(np.abs(valid_jerk)))

        # =================================================================
        # Distance & Duration
        # =================================================================
        metrics.total_duration = float(timestamps[-1] - timestamps[0]) if len(timestamps) > 1 else 0

        # Calculate total distance from displacements
        if len(positions_cm) > 1:
            dx = np.diff(positions_cm[:, 0])
            dy = np.diff(positions_cm[:, 1])
            displacements = np.sqrt(dx**2 + dy**2)
            metrics.total_distance = float(np.sum(displacements))

        # Active vs rest duration
        active_frames = np.sum(activity_labels > 0)
        rest_frames = np.sum(activity_labels == 0)
        metrics.active_duration = active_frames * self.dt
        metrics.rest_duration = rest_frames * self.dt

        # =================================================================
        # Activity Classification
        # =================================================================
        if n_frames > 0:
            metrics.time_resting = np.sum(activity_labels == 0) / n_frames * 100
            metrics.time_walking = np.sum(activity_labels == 1) / n_frames * 100
            metrics.time_trotting = np.sum(activity_labels == 2) / n_frames * 100
            metrics.time_galloping = np.sum(activity_labels == 3) / n_frames * 100

        # Count activity bouts
        metrics.n_bouts_walk, walk_durations = self._count_bouts(activity_labels, 1)
        metrics.n_bouts_trot, trot_durations = self._count_bouts(activity_labels, 2)
        metrics.n_bouts_gallop, gallop_durations = self._count_bouts(activity_labels, 3)

        all_bout_durations = walk_durations + trot_durations + gallop_durations
        if all_bout_durations:
            metrics.mean_bout_duration = float(np.mean(all_bout_durations)) * self.dt

        # =================================================================
        # Regularity Index
        # =================================================================
        # Based on stride-to-stride consistency
        if strides and len(strides) > 2:
            stride_lengths = [s.length for s in strides]
            mean_stride = np.mean(stride_lengths)
            if mean_stride > 0:
                deviations = np.abs(np.array(stride_lengths) - mean_stride) / mean_stride
                metrics.regularity_index = float((1 - np.mean(deviations)) * 100)

        # =================================================================
        # Data Quality Score
        # =================================================================
        # Composite score based on tracking rate and data consistency
        quality_components = [
            metrics.tracking_rate,
            min(100, metrics.regularity_index) if metrics.regularity_index > 0 else 50,
            100 - min(50, metrics.velocity_cv) if metrics.velocity_cv > 0 else 50,
        ]
        metrics.data_quality_score = float(np.mean(quality_components))

        # Store timeseries for further analysis
        metrics.velocity_timeseries = velocity
        metrics.acceleration_timeseries = acceleration

        return metrics

    def generate_report(self, metrics: ScientificGaitMetrics) -> str:
        """Generate formatted scientific report."""
        report = []
        report.append("=" * 70)
        report.append("SCIENTIFIC GAIT ANALYSIS REPORT")
        report.append("Mouse Locomotor Tracker - Stride Labs")
        report.append("=" * 70)
        report.append("")

        report.append("SPATIOTEMPORAL PARAMETERS")
        report.append("-" * 40)
        report.append(f"  Velocity:        {metrics.velocity_mean:6.2f} ± {metrics.velocity_sd:.2f} cm/s (CV: {metrics.velocity_cv:.1f}%)")
        report.append(f"  Stride Length:   {metrics.stride_length_mean:6.2f} ± {metrics.stride_length_sd:.2f} cm")
        report.append(f"  Cadence:         {metrics.cadence_mean:6.2f} ± {metrics.cadence_sd:.2f} steps/s")
        report.append(f"  Step Cycle:      {metrics.step_cycle_mean:6.3f} ± {metrics.step_cycle_sd:.3f} s")
        report.append(f"  Swing Time:      {metrics.swing_time_mean:6.3f} s ({metrics.swing_time_percent:.0f}%)")
        report.append(f"  Stance Time:     {metrics.stance_time_mean:6.3f} s ({metrics.stance_time_percent:.0f}%)")
        report.append(f"  Duty Factor:     {metrics.duty_factor:6.2f}")
        report.append("")

        report.append("KINEMATIC PARAMETERS")
        report.append("-" * 40)
        report.append(f"  Acceleration:    {metrics.acceleration_mean:6.2f} ± {metrics.acceleration_sd:.2f} cm/s²")
        report.append(f"  Peak Accel:      {metrics.acceleration_max:6.2f} cm/s²")
        report.append(f"  Jerk:            {metrics.jerk_mean:6.2f} ± {metrics.jerk_sd:.2f} cm/s³")
        report.append("")

        report.append("VARIABILITY & REGULARITY")
        report.append("-" * 40)
        report.append(f"  Regularity Index:    {metrics.regularity_index:6.1f}%")
        report.append(f"  Stride Variability:  {metrics.stride_variability:6.1f}% (CV)")
        report.append(f"  Velocity Stability:  {metrics.velocity_stability:6.1f}%")
        report.append("")

        report.append("DISTANCE & DURATION")
        report.append("-" * 40)
        report.append(f"  Total Distance:  {metrics.total_distance:8.2f} cm")
        report.append(f"  Total Duration:  {metrics.total_duration:8.2f} s")
        report.append(f"  Active Time:     {metrics.active_duration:8.2f} s")
        report.append(f"  Rest Time:       {metrics.rest_duration:8.2f} s")
        report.append("")

        report.append("ACTIVITY CLASSIFICATION")
        report.append("-" * 40)
        report.append(f"  Resting:     {metrics.time_resting:5.1f}%")
        report.append(f"  Walking:     {metrics.time_walking:5.1f}% ({metrics.n_bouts_walk} bouts)")
        report.append(f"  Trotting:    {metrics.time_trotting:5.1f}% ({metrics.n_bouts_trot} bouts)")
        report.append(f"  Galloping:   {metrics.time_galloping:5.1f}% ({metrics.n_bouts_gallop} bouts)")
        report.append(f"  Mean Bout:   {metrics.mean_bout_duration:.2f} s")
        report.append("")

        report.append("DATA QUALITY")
        report.append("-" * 40)
        report.append(f"  Tracking Rate:   {metrics.tracking_rate:6.1f}%")
        report.append(f"  Frames Total:    {metrics.n_frames_total:6d}")
        report.append(f"  Frames Tracked:  {metrics.n_frames_tracked:6d}")
        report.append(f"  Strides Detected:{metrics.n_strides_detected:6d}")
        report.append(f"  Quality Score:   {metrics.data_quality_score:6.1f}/100")
        report.append("")

        report.append("=" * 70)

        return "\n".join(report)

    def to_dict(self, metrics: ScientificGaitMetrics) -> Dict:
        """Convert metrics to dictionary for JSON export."""
        return {
            "spatiotemporal": {
                "velocity": {
                    "mean": metrics.velocity_mean,
                    "sd": metrics.velocity_sd,
                    "sem": metrics.velocity_sem,
                    "cv": metrics.velocity_cv,
                    "max": metrics.velocity_max,
                    "min": metrics.velocity_min,
                    "unit": "cm/s"
                },
                "stride_length": {
                    "mean": metrics.stride_length_mean,
                    "sd": metrics.stride_length_sd,
                    "sem": metrics.stride_length_sem,
                    "cv": metrics.stride_length_cv,
                    "unit": "cm"
                },
                "cadence": {
                    "mean": metrics.cadence_mean,
                    "sd": metrics.cadence_sd,
                    "unit": "steps/s"
                },
                "step_cycle": {
                    "mean": metrics.step_cycle_mean,
                    "sd": metrics.step_cycle_sd,
                    "unit": "s"
                },
                "swing_time": {
                    "mean": metrics.swing_time_mean,
                    "sd": metrics.swing_time_sd,
                    "percent": metrics.swing_time_percent,
                    "unit": "s"
                },
                "stance_time": {
                    "mean": metrics.stance_time_mean,
                    "sd": metrics.stance_time_sd,
                    "percent": metrics.stance_time_percent,
                    "unit": "s"
                },
                "duty_factor": metrics.duty_factor,
                "swing_speed": {
                    "mean": metrics.swing_speed_mean,
                    "sd": metrics.swing_speed_sd,
                    "unit": "cm/s"
                }
            },
            "kinematic": {
                "acceleration": {
                    "mean": metrics.acceleration_mean,
                    "sd": metrics.acceleration_sd,
                    "max": metrics.acceleration_max,
                    "unit": "cm/s²"
                },
                "jerk": {
                    "mean": metrics.jerk_mean,
                    "sd": metrics.jerk_sd,
                    "max": metrics.jerk_max,
                    "unit": "cm/s³"
                }
            },
            "variability": {
                "regularity_index": metrics.regularity_index,
                "stride_variability": metrics.stride_variability,
                "velocity_stability": metrics.velocity_stability
            },
            "distance_duration": {
                "total_distance_cm": metrics.total_distance,
                "total_duration_s": metrics.total_duration,
                "active_duration_s": metrics.active_duration,
                "rest_duration_s": metrics.rest_duration
            },
            "activity": {
                "time_resting_pct": metrics.time_resting,
                "time_walking_pct": metrics.time_walking,
                "time_trotting_pct": metrics.time_trotting,
                "time_galloping_pct": metrics.time_galloping,
                "n_bouts_walk": metrics.n_bouts_walk,
                "n_bouts_trot": metrics.n_bouts_trot,
                "n_bouts_gallop": metrics.n_bouts_gallop,
                "mean_bout_duration_s": metrics.mean_bout_duration
            },
            "quality": {
                "tracking_rate": metrics.tracking_rate,
                "n_frames_total": metrics.n_frames_total,
                "n_frames_tracked": metrics.n_frames_tracked,
                "n_strides_detected": metrics.n_strides_detected,
                "data_quality_score": metrics.data_quality_score
            }
        }


# =============================================================================
# Convenience Functions
# =============================================================================

def analyze_gait(
    positions: np.ndarray,
    fps: float = 30.0,
    cm_per_pixel: float = 0.03125,
) -> ScientificGaitMetrics:
    """
    Convenience function for quick gait analysis.

    Args:
        positions: Array of (x, y) positions in pixels
        fps: Frame rate
        cm_per_pixel: Spatial calibration

    Returns:
        ScientificGaitMetrics object
    """
    analyzer = ScientificGaitAnalyzer(fps=fps, cm_per_pixel=cm_per_pixel)
    return analyzer.analyze(positions)


if __name__ == "__main__":
    # Demo with synthetic data
    print("Scientific Gait Analyzer - Demo")
    print("=" * 50)

    # Generate synthetic mouse locomotion data
    np.random.seed(42)
    n_frames = 300
    fps = 30.0

    # Simulate mouse walking on treadmill
    t = np.linspace(0, n_frames/fps, n_frames)

    # Position with oscillation (simulating gait)
    x = 600 + np.cumsum(np.random.randn(n_frames) * 2 + 1)  # Forward motion
    y = 450 + 10 * np.sin(t * 10) + np.random.randn(n_frames) * 2  # Lateral oscillation

    positions = np.column_stack([x, y])

    # Analyze
    analyzer = ScientificGaitAnalyzer(fps=fps, cm_per_pixel=0.03125)
    metrics = analyzer.analyze(positions)

    # Print report
    print(analyzer.generate_report(metrics))
