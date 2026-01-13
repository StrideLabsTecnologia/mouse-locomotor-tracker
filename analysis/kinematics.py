"""
Joint Angle Kinematics Module
=============================

Calculates joint angles between anatomical segments using
vector mathematics (dot product method).

Based on Locomotor-Allodi2021 methodology with enhancements.

Author: Stride Labs - Mouse Locomotor Tracker
"""

import numpy as np
from scipy import signal
from scipy.ndimage import uniform_filter1d
from typing import Optional, Tuple, List, Dict, Any, Union
from dataclasses import dataclass
import warnings

from .metrics import JointType, JointAngleMetrics, KinematicsMetrics


# Standard joint marker triplets for mouse hindlimb
# Each tuple is (proximal_marker, joint_marker, distal_marker)
JOINT_MARKER_DEFINITIONS = {
    JointType.HIP: ('iliac_crest', 'hip', 'knee'),
    JointType.KNEE: ('hip', 'knee', 'ankle'),
    JointType.ANKLE: ('knee', 'ankle', 'mtp'),
    JointType.FOOT: ('ankle', 'mtp', 'toe_tip'),
    JointType.MTP: ('ankle', 'mtp', 'toe_tip'),
}


class JointAngleAnalyzer:
    """
    Analyzes joint angles from marker coordinate data.

    Joint angles are calculated using the dot product formula:
    angle = arccos((v1 . v2) / (|v1| * |v2|))

    where v1 and v2 are vectors formed by connecting the joint
    to proximal and distal markers respectively.

    Attributes:
        frame_rate: Video frame rate in Hz
        pixel_to_mm: Conversion factor from pixels to mm
        smoothing_window: Window size for angle smoothing
        angle_convention: 'anatomical' (flexion=smaller) or 'mathematical'

    Example:
        >>> analyzer = JointAngleAnalyzer(frame_rate=30)
        >>> angle = analyzer.calculate_angle(p1, p2, p3)
        >>> metrics = analyzer.analyze_joint(
        ...     hip_coords, knee_coords, ankle_coords,
        ...     joint_type=JointType.KNEE
        ... )
    """

    def __init__(
        self,
        frame_rate: float = 30.0,
        pixel_to_mm: float = 1.0,
        smoothing_window: int = 5,
        angle_convention: str = 'anatomical',
    ):
        """
        Initialize the JointAngleAnalyzer.

        Args:
            frame_rate: Video frame rate in Hz
            pixel_to_mm: Pixels to millimeters conversion
            smoothing_window: Window for moving average smoothing
            angle_convention: 'anatomical' or 'mathematical'
        """
        if frame_rate <= 0:
            raise ValueError("frame_rate must be positive")
        if smoothing_window < 1:
            raise ValueError("smoothing_window must be at least 1")
        if angle_convention not in ['anatomical', 'mathematical']:
            raise ValueError(
                "angle_convention must be 'anatomical' or 'mathematical'"
            )

        self.frame_rate = frame_rate
        self.pixel_to_mm = pixel_to_mm
        self.smoothing_window = smoothing_window
        self.angle_convention = angle_convention
        self._dt = 1.0 / frame_rate

    def _validate_point(
        self, point: np.ndarray, name: str = "point"
    ) -> np.ndarray:
        """
        Validate and format a point array.

        Args:
            point: Point coordinates, shape (2,) or (N, 2)
            name: Name for error messages

        Returns:
            Validated point array
        """
        point = np.asarray(point, dtype=np.float64)

        if point.ndim == 1:
            if len(point) != 2:
                raise ValueError(
                    f"{name} must have 2 coordinates (x, y), got {len(point)}"
                )
            point = point.reshape(1, 2)

        if point.ndim != 2 or point.shape[1] != 2:
            raise ValueError(
                f"{name} must have shape (N, 2), got {point.shape}"
            )

        return point

    def _handle_missing_data(
        self,
        p1: np.ndarray,
        p2: np.ndarray,
        p3: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Handle NaN values in coordinate data.

        Args:
            p1: Proximal marker coordinates
            p2: Joint marker coordinates
            p3: Distal marker coordinates

        Returns:
            Tuple of (p1_clean, p2_clean, p3_clean, valid_mask)
        """
        # Find frames where all three points are valid
        valid_mask = (
            ~np.isnan(p1).any(axis=1) &
            ~np.isnan(p2).any(axis=1) &
            ~np.isnan(p3).any(axis=1)
        )

        if not np.any(valid_mask):
            raise ValueError("All coordinate values are NaN")

        # For invalid frames, interpolate
        if not np.all(valid_mask):
            indices = np.arange(len(p1))
            valid_idx = indices[valid_mask]

            p1_clean = np.zeros_like(p1)
            p2_clean = np.zeros_like(p2)
            p3_clean = np.zeros_like(p3)

            for i in range(2):  # x and y
                p1_clean[:, i] = np.interp(indices, valid_idx, p1[valid_mask, i])
                p2_clean[:, i] = np.interp(indices, valid_idx, p2[valid_mask, i])
                p3_clean[:, i] = np.interp(indices, valid_idx, p3[valid_mask, i])

            return p1_clean, p2_clean, p3_clean, valid_mask

        return p1.copy(), p2.copy(), p3.copy(), valid_mask

    def calculate_angle(
        self,
        p1: np.ndarray,
        p2: np.ndarray,
        p3: np.ndarray,
        degrees: bool = True,
    ) -> Union[float, np.ndarray]:
        """
        Calculate angle at p2 formed by vectors p2->p1 and p2->p3.

        Uses the dot product formula:
        cos(angle) = (v1 . v2) / (|v1| * |v2|)

        Args:
            p1: Proximal point coordinates, shape (2,) or (N, 2)
            p2: Joint point coordinates (vertex), shape (2,) or (N, 2)
            p3: Distal point coordinates, shape (2,) or (N, 2)
            degrees: Return angle in degrees (True) or radians (False)

        Returns:
            Angle(s) at joint point(s)
        """
        # Validate and format inputs
        p1 = self._validate_point(p1, "p1 (proximal)")
        p2 = self._validate_point(p2, "p2 (joint)")
        p3 = self._validate_point(p3, "p3 (distal)")

        # Check shapes match
        if not (p1.shape == p2.shape == p3.shape):
            raise ValueError(
                f"All points must have same shape: "
                f"{p1.shape}, {p2.shape}, {p3.shape}"
            )

        # Compute vectors from joint to proximal and distal
        v1 = p1 - p2  # Joint to proximal
        v2 = p3 - p2  # Joint to distal

        # Compute dot product
        dot_product = np.sum(v1 * v2, axis=1)

        # Compute magnitudes
        mag_v1 = np.linalg.norm(v1, axis=1)
        mag_v2 = np.linalg.norm(v2, axis=1)

        # Handle zero-length vectors
        with np.errstate(divide='ignore', invalid='ignore'):
            cos_angle = dot_product / (mag_v1 * mag_v2)

        # Clamp to [-1, 1] to handle numerical errors
        cos_angle = np.clip(cos_angle, -1.0, 1.0)

        # Compute angle
        angle = np.arccos(cos_angle)

        if degrees:
            angle = np.degrees(angle)

        # Return scalar if input was single point
        if len(angle) == 1:
            return float(angle[0])

        return angle

    def calculate_angle_with_sign(
        self,
        p1: np.ndarray,
        p2: np.ndarray,
        p3: np.ndarray,
        degrees: bool = True,
    ) -> Union[float, np.ndarray]:
        """
        Calculate signed angle at p2 (useful for flexion/extension).

        Positive = counterclockwise rotation from v1 to v2
        Negative = clockwise rotation

        Args:
            p1: Proximal point coordinates
            p2: Joint point coordinates
            p3: Distal point coordinates
            degrees: Return in degrees

        Returns:
            Signed angle(s)
        """
        p1 = self._validate_point(p1, "p1")
        p2 = self._validate_point(p2, "p2")
        p3 = self._validate_point(p3, "p3")

        v1 = p1 - p2
        v2 = p3 - p2

        # Cross product z-component gives sign
        cross_z = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]

        # Magnitude of angle
        angle = self.calculate_angle(p1, p2, p3, degrees=False)

        # Apply sign
        if isinstance(angle, np.ndarray):
            signed_angle = np.where(cross_z >= 0, angle, -angle)
        else:
            signed_angle = angle if cross_z[0] >= 0 else -angle

        if degrees:
            signed_angle = np.degrees(signed_angle)

        return signed_angle

    def _smooth_angles(
        self, angles: np.ndarray, window: Optional[int] = None
    ) -> np.ndarray:
        """
        Apply smoothing to angle time series.

        Args:
            angles: Angle values
            window: Smoothing window (default: self.smoothing_window)

        Returns:
            Smoothed angles
        """
        if window is None:
            window = self.smoothing_window

        if window <= 1:
            return angles.copy()

        return uniform_filter1d(angles, size=window, mode='nearest')

    def _calculate_angular_velocity(self, angles: np.ndarray) -> np.ndarray:
        """
        Calculate angular velocity from angle time series.

        Args:
            angles: Angle time series in degrees

        Returns:
            Angular velocity in degrees/second
        """
        n = len(angles)
        velocity = np.zeros(n)

        if n < 2:
            return velocity

        # Central difference for interior
        if n > 2:
            velocity[1:-1] = (angles[2:] - angles[:-2]) / (2 * self._dt)

        # Forward/backward for boundaries
        velocity[0] = (angles[1] - angles[0]) / self._dt
        velocity[-1] = (angles[-1] - angles[-2]) / self._dt

        return velocity

    def analyze_joint(
        self,
        proximal_coords: np.ndarray,
        joint_coords: np.ndarray,
        distal_coords: np.ndarray,
        joint_type: JointType = JointType.KNEE,
        stance_frames: Optional[np.ndarray] = None,
        swing_frames: Optional[np.ndarray] = None,
    ) -> JointAngleMetrics:
        """
        Perform complete analysis for a single joint.

        Args:
            proximal_coords: Coordinates of proximal marker, shape (N, 2)
            joint_coords: Coordinates of joint marker, shape (N, 2)
            distal_coords: Coordinates of distal marker, shape (N, 2)
            joint_type: Type of joint being analyzed
            stance_frames: Boolean mask or indices of stance phase frames
            swing_frames: Boolean mask or indices of swing phase frames

        Returns:
            JointAngleMetrics containing all joint analysis results
        """
        # Validate inputs
        p1 = self._validate_point(proximal_coords, "proximal")
        p2 = self._validate_point(joint_coords, "joint")
        p3 = self._validate_point(distal_coords, "distal")

        # Handle missing data
        p1, p2, p3, valid_mask = self._handle_missing_data(p1, p2, p3)

        # Calculate angles
        angles = self.calculate_angle(p1, p2, p3, degrees=True)

        # Convert to anatomical convention if needed
        if self.angle_convention == 'anatomical':
            # For most joints, 180 deg = full extension
            # We keep as is (supplement is flexion)
            pass

        # Smooth angles
        smoothed_angles = self._smooth_angles(angles)

        # Calculate angular velocity
        angular_velocity = self._calculate_angular_velocity(smoothed_angles)

        # Compute statistics
        mean_angle = float(np.nanmean(smoothed_angles))
        max_angle = float(np.nanmax(smoothed_angles))
        min_angle = float(np.nanmin(smoothed_angles))
        rom = max_angle - min_angle

        # Phase-specific angles
        angle_at_stance = None
        angle_at_swing = None

        if stance_frames is not None:
            stance_mask = np.zeros(len(angles), dtype=bool)
            stance_mask[stance_frames] = True
            if np.any(stance_mask):
                angle_at_stance = float(np.nanmean(smoothed_angles[stance_mask]))

        if swing_frames is not None:
            swing_mask = np.zeros(len(angles), dtype=bool)
            swing_mask[swing_frames] = True
            if np.any(swing_mask):
                angle_at_swing = float(np.nanmean(smoothed_angles[swing_mask]))

        return JointAngleMetrics(
            joint_type=joint_type,
            mean_angle=mean_angle,
            max_angle=max_angle,
            min_angle=min_angle,
            range_of_motion=rom,
            angle_at_stance=angle_at_stance,
            angle_at_swing=angle_at_swing,
            angular_velocity=angular_velocity,
            angle_profile=smoothed_angles,
            cycle_angles=None,  # Set by get_cycle_angles
        )

    def get_cycle_angles(
        self,
        angles: np.ndarray,
        cycle_starts: np.ndarray,
        cycle_ends: np.ndarray,
        normalize_length: int = 100,
    ) -> List[np.ndarray]:
        """
        Extract and normalize angles for each gait cycle.

        Each cycle is interpolated to a standard length for comparison.

        Args:
            angles: Full angle time series
            cycle_starts: Frame indices of cycle starts
            cycle_ends: Frame indices of cycle ends
            normalize_length: Number of points for normalized cycle

        Returns:
            List of normalized angle arrays (one per cycle)
        """
        angles = np.asarray(angles)
        cycle_starts = np.asarray(cycle_starts)
        cycle_ends = np.asarray(cycle_ends)

        if len(cycle_starts) != len(cycle_ends):
            raise ValueError(
                f"cycle_starts and cycle_ends must have same length: "
                f"{len(cycle_starts)} vs {len(cycle_ends)}"
            )

        normalized_cycles = []

        for start, end in zip(cycle_starts, cycle_ends):
            if end <= start:
                continue

            cycle_angles = angles[start:end]

            if len(cycle_angles) < 2:
                continue

            # Interpolate to standard length
            x_original = np.linspace(0, 1, len(cycle_angles))
            x_normalized = np.linspace(0, 1, normalize_length)

            normalized = np.interp(x_normalized, x_original, cycle_angles)
            normalized_cycles.append(normalized)

        return normalized_cycles

    def compute_average_cycle(
        self, cycle_angles: List[np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute mean and standard deviation across cycles.

        Args:
            cycle_angles: List of normalized cycle angle arrays

        Returns:
            Tuple of (mean_cycle, std_cycle)
        """
        if not cycle_angles:
            return np.array([]), np.array([])

        # Stack all cycles
        stacked = np.vstack(cycle_angles)

        mean_cycle = np.nanmean(stacked, axis=0)
        std_cycle = np.nanstd(stacked, axis=0)

        return mean_cycle, std_cycle

    def analyze_all_joints(
        self,
        marker_coords: Dict[str, np.ndarray],
        joint_definitions: Optional[Dict[JointType, Tuple[str, str, str]]] = None,
        stance_frames: Optional[np.ndarray] = None,
        swing_frames: Optional[np.ndarray] = None,
    ) -> KinematicsMetrics:
        """
        Analyze all defined joints from marker data.

        Args:
            marker_coords: Dictionary mapping marker names to (N, 2) coordinates
            joint_definitions: Optional custom joint definitions
            stance_frames: Optional stance phase frame indices
            swing_frames: Optional swing phase frame indices

        Returns:
            KinematicsMetrics containing all joint analyses
        """
        if joint_definitions is None:
            joint_definitions = JOINT_MARKER_DEFINITIONS

        joint_metrics: Dict[JointType, JointAngleMetrics] = {}

        for joint_type, (prox, joint, dist) in joint_definitions.items():
            # Check if all markers are available
            if prox not in marker_coords:
                warnings.warn(f"Missing marker '{prox}' for {joint_type.value}")
                continue
            if joint not in marker_coords:
                warnings.warn(f"Missing marker '{joint}' for {joint_type.value}")
                continue
            if dist not in marker_coords:
                warnings.warn(f"Missing marker '{dist}' for {joint_type.value}")
                continue

            try:
                metrics = self.analyze_joint(
                    marker_coords[prox],
                    marker_coords[joint],
                    marker_coords[dist],
                    joint_type=joint_type,
                    stance_frames=stance_frames,
                    swing_frames=swing_frames,
                )
                joint_metrics[joint_type] = metrics
            except Exception as e:
                warnings.warn(f"Failed to analyze {joint_type.value}: {e}")

        # Calculate segment lengths
        limb_lengths = self._calculate_segment_lengths(marker_coords)

        # Calculate step height and toe clearance
        step_height = 0.0
        toe_clearance = 0.0

        if 'toe_tip' in marker_coords:
            toe_y = marker_coords['toe_tip'][:, 1]  # Assuming y is vertical
            # In image coordinates, smaller y = higher position
            step_height = float(np.nanmax(toe_y) - np.nanmin(toe_y))

            if swing_frames is not None and len(swing_frames) > 0:
                swing_toe_y = toe_y[swing_frames]
                toe_clearance = float(np.nanmin(swing_toe_y))

        # Calculate body angle (trunk inclination)
        body_angle = 0.0
        body_angle_var = 0.0

        if 'hip' in marker_coords and 'iliac_crest' in marker_coords:
            hip = marker_coords['hip']
            crest = marker_coords['iliac_crest']

            # Angle relative to horizontal
            dx = crest[:, 0] - hip[:, 0]
            dy = crest[:, 1] - hip[:, 1]
            trunk_angles = np.degrees(np.arctan2(dy, dx))

            body_angle = float(np.nanmean(trunk_angles))
            body_angle_var = float(np.nanstd(trunk_angles))

        return KinematicsMetrics(
            joint_metrics=joint_metrics,
            limb_length=limb_lengths,
            step_height=step_height * self.pixel_to_mm,
            toe_clearance=toe_clearance * self.pixel_to_mm,
            body_angle=body_angle,
            body_angle_variability=body_angle_var,
        )

    def _calculate_segment_lengths(
        self, marker_coords: Dict[str, np.ndarray]
    ) -> Dict[str, float]:
        """
        Calculate limb segment lengths from marker positions.

        Args:
            marker_coords: Dictionary of marker coordinates

        Returns:
            Dictionary of segment names to lengths in mm
        """
        lengths = {}

        # Define segments
        segments = [
            ('thigh', 'hip', 'knee'),
            ('shank', 'knee', 'ankle'),
            ('foot', 'ankle', 'mtp'),
            ('toe', 'mtp', 'toe_tip'),
        ]

        for name, start, end in segments:
            if start in marker_coords and end in marker_coords:
                p1 = marker_coords[start]
                p2 = marker_coords[end]

                # Calculate distances
                distances = np.linalg.norm(p2 - p1, axis=1)
                mean_length = float(np.nanmean(distances))
                lengths[name] = mean_length * self.pixel_to_mm

        return lengths

    def detect_joint_events(
        self,
        angles: np.ndarray,
        event_type: str = 'peaks',
        min_prominence: float = 5.0,
        min_distance_frames: int = 10,
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Detect joint angle events (peaks, valleys).

        Args:
            angles: Angle time series
            event_type: 'peaks', 'valleys', or 'both'
            min_prominence: Minimum event prominence in degrees
            min_distance_frames: Minimum frames between events

        Returns:
            Tuple of (event_indices, event_properties)
        """
        angles = np.asarray(angles)

        if event_type in ['peaks', 'both']:
            peaks, peak_props = signal.find_peaks(
                angles,
                prominence=min_prominence,
                distance=min_distance_frames,
            )
        else:
            peaks = np.array([])
            peak_props = {}

        if event_type in ['valleys', 'both']:
            valleys, valley_props = signal.find_peaks(
                -angles,
                prominence=min_prominence,
                distance=min_distance_frames,
            )
        else:
            valleys = np.array([])
            valley_props = {}

        if event_type == 'peaks':
            return peaks, peak_props
        elif event_type == 'valleys':
            return valleys, valley_props
        else:
            # Combine peaks and valleys
            all_events = np.concatenate([peaks, valleys])
            sort_idx = np.argsort(all_events)
            all_events = all_events[sort_idx]

            return all_events, {'peaks': peaks, 'valleys': valleys}


def create_joint_analyzer(
    frame_rate: float = 30.0,
    pixel_to_mm: float = 1.0,
    config: Optional[Dict[str, Any]] = None,
) -> JointAngleAnalyzer:
    """
    Factory function to create a JointAngleAnalyzer.

    Args:
        frame_rate: Video frame rate
        pixel_to_mm: Pixel to mm conversion
        config: Optional configuration dictionary

    Returns:
        Configured JointAngleAnalyzer instance
    """
    if config is None:
        config = {}

    return JointAngleAnalyzer(
        frame_rate=frame_rate,
        pixel_to_mm=pixel_to_mm,
        smoothing_window=config.get('smoothing_window', 5),
        angle_convention=config.get('angle_convention', 'anatomical'),
    )


def calculate_limb_endpoint_trajectory(
    joint_angles: Dict[JointType, np.ndarray],
    segment_lengths: Dict[str, float],
    base_position: np.ndarray,
) -> np.ndarray:
    """
    Calculate limb endpoint (toe) trajectory from joint angles.

    Uses forward kinematics to compute toe position from
    joint angles and segment lengths.

    Args:
        joint_angles: Dictionary of joint angles over time
        segment_lengths: Dictionary of segment lengths
        base_position: Hip/base position over time, shape (N, 2)

    Returns:
        Toe position trajectory, shape (N, 2)
    """
    n_frames = len(base_position)
    toe_position = np.zeros((n_frames, 2))

    # Get segment lengths (with defaults)
    l_thigh = segment_lengths.get('thigh', 10.0)
    l_shank = segment_lengths.get('shank', 10.0)
    l_foot = segment_lengths.get('foot', 5.0)

    # Get joint angles
    hip_angles = joint_angles.get(JointType.HIP, np.zeros(n_frames))
    knee_angles = joint_angles.get(JointType.KNEE, np.zeros(n_frames))
    ankle_angles = joint_angles.get(JointType.ANKLE, np.zeros(n_frames))

    for i in range(n_frames):
        # Convert to radians
        hip_rad = np.radians(hip_angles[i])
        knee_rad = np.radians(knee_angles[i])
        ankle_rad = np.radians(ankle_angles[i])

        # Cumulative angle (assuming angles are relative)
        theta1 = hip_rad
        theta2 = theta1 + (np.pi - knee_rad)  # Knee flexion
        theta3 = theta2 + (np.pi - ankle_rad)  # Ankle dorsiflexion

        # Forward kinematics
        x = base_position[i, 0]
        y = base_position[i, 1]

        # Add thigh contribution
        x += l_thigh * np.cos(theta1)
        y += l_thigh * np.sin(theta1)

        # Add shank contribution
        x += l_shank * np.cos(theta2)
        y += l_shank * np.sin(theta2)

        # Add foot contribution
        x += l_foot * np.cos(theta3)
        y += l_foot * np.sin(theta3)

        toe_position[i] = [x, y]

    return toe_position
