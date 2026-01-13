"""
Biomechanical Metrics Data Classes
==================================

Dataclasses for all locomotor analysis metrics.
Based on Locomotor-Allodi2021 methodology with enhancements.

Author: Stride Labs - Mouse Locomotor Tracker
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from enum import Enum
import numpy as np
from datetime import datetime


class LimbPair(Enum):
    """Enumeration of limb pairs for coordination analysis."""
    LHRH = "left_hind_right_hind"      # Left Hind - Right Hind
    LHLF = "left_hind_left_fore"        # Left Hind - Left Fore
    RHRF = "right_hind_right_fore"      # Right Hind - Right Fore
    LFRH = "left_fore_right_hind"       # Left Fore - Right Hind
    RFLH = "right_fore_left_hind"       # Right Fore - Left Hind
    LFRF = "left_fore_right_fore"       # Left Fore - Right Fore


class JointType(Enum):
    """Enumeration of joint types for kinematic analysis."""
    HIP = "hip"
    KNEE = "knee"
    ANKLE = "ankle"
    FOOT = "foot"
    MTP = "metatarsophalangeal"  # Toe joint


class LocomotionPhase(Enum):
    """Gait cycle phases."""
    STANCE = "stance"      # Foot on ground
    SWING = "swing"        # Foot in air
    DRAG = "drag"          # Toe dragging (pathological)
    RECOVERY = "recovery"  # Post-drag recovery


@dataclass
class VelocityMetrics:
    """
    Velocity and acceleration metrics from locomotion analysis.

    Attributes:
        mean_velocity: Average velocity in mm/s
        max_velocity: Peak velocity in mm/s
        min_velocity: Minimum velocity in mm/s
        velocity_std: Standard deviation of velocity
        mean_acceleration: Average acceleration in mm/s^2
        max_acceleration: Peak acceleration in mm/s^2
        velocity_profile: Time series of velocity values
        acceleration_profile: Time series of acceleration values
        smoothed_velocity: Velocity after moving average filter
        drag_events: List of detected drag event indices
        recovery_events: List of detected recovery event indices
        drag_count: Total number of drag events
        velocity_coefficient_of_variation: CV = std/mean * 100
    """
    mean_velocity: float
    max_velocity: float
    min_velocity: float
    velocity_std: float
    mean_acceleration: float
    max_acceleration: float
    velocity_profile: np.ndarray
    acceleration_profile: np.ndarray
    smoothed_velocity: np.ndarray
    drag_events: List[int] = field(default_factory=list)
    recovery_events: List[int] = field(default_factory=list)
    drag_count: int = 0
    velocity_coefficient_of_variation: float = 0.0

    def __post_init__(self):
        """Calculate derived metrics after initialization."""
        if self.mean_velocity > 0:
            self.velocity_coefficient_of_variation = (
                self.velocity_std / self.mean_velocity
            ) * 100

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "mean_velocity_mm_s": float(self.mean_velocity),
            "max_velocity_mm_s": float(self.max_velocity),
            "min_velocity_mm_s": float(self.min_velocity),
            "velocity_std": float(self.velocity_std),
            "velocity_cv_percent": float(self.velocity_coefficient_of_variation),
            "mean_acceleration_mm_s2": float(self.mean_acceleration),
            "max_acceleration_mm_s2": float(self.max_acceleration),
            "drag_count": self.drag_count,
            "drag_events": self.drag_events,
            "recovery_events": self.recovery_events,
        }


@dataclass
class CircularStatistics:
    """
    Circular statistics for a single limb pair.

    Based on Rayleigh test for circular uniformity.

    Attributes:
        mean_angle: Mean circular angle in radians
        mean_angle_degrees: Mean circular angle in degrees
        mean_vector_length: R value (0-1), measures concentration
        x_component: Mean of cos(phases)
        y_component: Mean of sin(phases)
        rayleigh_z: Rayleigh Z statistic
        rayleigh_p: P-value for Rayleigh test
        is_significant: Whether coordination is significantly non-uniform
        sample_size: Number of phase samples
    """
    mean_angle: float
    mean_angle_degrees: float
    mean_vector_length: float  # R value
    x_component: float
    y_component: float
    rayleigh_z: float
    rayleigh_p: float
    is_significant: bool
    sample_size: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "mean_angle_rad": float(self.mean_angle),
            "mean_angle_deg": float(self.mean_angle_degrees),
            "mean_vector_length_R": float(self.mean_vector_length),
            "x_component": float(self.x_component),
            "y_component": float(self.y_component),
            "rayleigh_z": float(self.rayleigh_z),
            "rayleigh_p": float(self.rayleigh_p),
            "is_significant": self.is_significant,
            "sample_size": self.sample_size,
        }


@dataclass
class CoordinationMetrics:
    """
    Interlimb coordination metrics using circular statistics.

    Attributes:
        pair_statistics: Dictionary mapping LimbPair to CircularStatistics
        overall_coordination_score: Aggregate coordination score (0-1)
        ipsilateral_coupling: Average R for same-side limbs
        contralateral_coupling: Average R for opposite-side limbs
        fore_hind_coupling: Average R for fore-hind pairs
        left_right_coupling: Average R for left-right pairs
        coordination_pattern: Detected gait pattern name
    """
    pair_statistics: Dict[LimbPair, CircularStatistics]
    overall_coordination_score: float
    ipsilateral_coupling: float
    contralateral_coupling: float
    fore_hind_coupling: float
    left_right_coupling: float
    coordination_pattern: str = "unknown"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "pair_statistics": {
                pair.value: stats.to_dict()
                for pair, stats in self.pair_statistics.items()
            },
            "overall_coordination_score": float(self.overall_coordination_score),
            "ipsilateral_coupling": float(self.ipsilateral_coupling),
            "contralateral_coupling": float(self.contralateral_coupling),
            "fore_hind_coupling": float(self.fore_hind_coupling),
            "left_right_coupling": float(self.left_right_coupling),
            "coordination_pattern": self.coordination_pattern,
        }


@dataclass
class JointAngleMetrics:
    """
    Metrics for a single joint's angular kinematics.

    Attributes:
        joint_type: Type of joint analyzed
        mean_angle: Mean angle in degrees
        max_angle: Maximum angle (extension)
        min_angle: Minimum angle (flexion)
        range_of_motion: ROM = max - min
        angle_at_stance: Angle during stance phase
        angle_at_swing: Angle during swing phase
        angular_velocity: Rate of angle change
        angle_profile: Time series of angles
        cycle_angles: Angles organized by gait cycle
    """
    joint_type: JointType
    mean_angle: float
    max_angle: float
    min_angle: float
    range_of_motion: float
    angle_at_stance: Optional[float] = None
    angle_at_swing: Optional[float] = None
    angular_velocity: Optional[np.ndarray] = None
    angle_profile: Optional[np.ndarray] = None
    cycle_angles: Optional[List[np.ndarray]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "joint_type": self.joint_type.value,
            "mean_angle_deg": float(self.mean_angle),
            "max_angle_deg": float(self.max_angle),
            "min_angle_deg": float(self.min_angle),
            "range_of_motion_deg": float(self.range_of_motion),
            "angle_at_stance_deg": (
                float(self.angle_at_stance) if self.angle_at_stance else None
            ),
            "angle_at_swing_deg": (
                float(self.angle_at_swing) if self.angle_at_swing else None
            ),
        }


@dataclass
class KinematicsMetrics:
    """
    Complete kinematic analysis metrics.

    Attributes:
        joint_metrics: Dictionary mapping JointType to JointAngleMetrics
        limb_length: Estimated limb segment lengths
        step_height: Maximum toe elevation during swing
        toe_clearance: Minimum toe height during swing
        body_angle: Trunk angle relative to horizontal
        body_angle_variability: Stability of body angle
    """
    joint_metrics: Dict[JointType, JointAngleMetrics]
    limb_length: Dict[str, float] = field(default_factory=dict)
    step_height: float = 0.0
    toe_clearance: float = 0.0
    body_angle: float = 0.0
    body_angle_variability: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "joint_metrics": {
                joint.value: metrics.to_dict()
                for joint, metrics in self.joint_metrics.items()
            },
            "limb_length_mm": self.limb_length,
            "step_height_mm": float(self.step_height),
            "toe_clearance_mm": float(self.toe_clearance),
            "body_angle_deg": float(self.body_angle),
            "body_angle_variability_deg": float(self.body_angle_variability),
        }


@dataclass
class GaitCycleInfo:
    """
    Information about a single gait cycle.

    Attributes:
        start_frame: Frame index where cycle starts
        end_frame: Frame index where cycle ends
        duration_frames: Duration in frames
        duration_seconds: Duration in seconds
        stance_duration: Duration of stance phase
        swing_duration: Duration of swing phase
        duty_factor: Stance duration / total duration
        stride_length: Distance traveled in one cycle
    """
    start_frame: int
    end_frame: int
    duration_frames: int
    duration_seconds: float
    stance_duration: float = 0.0
    swing_duration: float = 0.0
    duty_factor: float = 0.0
    stride_length: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "duration_frames": self.duration_frames,
            "duration_seconds": float(self.duration_seconds),
            "stance_duration_s": float(self.stance_duration),
            "swing_duration_s": float(self.swing_duration),
            "duty_factor": float(self.duty_factor),
            "stride_length_mm": float(self.stride_length),
        }


@dataclass
class GaitMetrics:
    """
    Gait cycle detection and temporal metrics.

    Attributes:
        cycles: List of detected gait cycles
        num_cycles: Total number of complete cycles
        cadence: Steps per second (Hz)
        mean_stride_length: Average stride length in mm
        stride_length_variability: CV of stride length
        mean_cycle_duration: Average cycle time in seconds
        cycle_duration_variability: CV of cycle duration
        mean_duty_factor: Average stance/total ratio
        step_frequency: Steps per minute
        symmetry_index: Left vs right symmetry (1.0 = perfect)
    """
    cycles: List[GaitCycleInfo]
    num_cycles: int
    cadence: float
    mean_stride_length: float
    stride_length_variability: float
    mean_cycle_duration: float
    cycle_duration_variability: float
    mean_duty_factor: float
    step_frequency: float
    symmetry_index: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "num_cycles": self.num_cycles,
            "cadence_hz": float(self.cadence),
            "step_frequency_per_min": float(self.step_frequency),
            "mean_stride_length_mm": float(self.mean_stride_length),
            "stride_length_cv_percent": float(self.stride_length_variability),
            "mean_cycle_duration_s": float(self.mean_cycle_duration),
            "cycle_duration_cv_percent": float(self.cycle_duration_variability),
            "mean_duty_factor": float(self.mean_duty_factor),
            "symmetry_index": float(self.symmetry_index),
            "cycles": [c.to_dict() for c in self.cycles],
        }


@dataclass
class LocomotorReport:
    """
    Complete locomotor analysis report combining all metrics.

    This is the main output structure containing all analysis results.

    Attributes:
        subject_id: Identifier for the animal/session
        recording_date: Date of the recording
        analysis_timestamp: When the analysis was performed
        duration_seconds: Total recording duration
        frame_rate: Video frame rate in Hz
        pixel_to_mm: Conversion factor from pixels to mm
        velocity_metrics: Velocity and acceleration analysis
        coordination_metrics: Interlimb coordination analysis
        kinematics_metrics: Joint angle analysis
        gait_metrics: Gait cycle analysis
        quality_score: Overall data quality (0-100)
        notes: Any analysis notes or warnings
        metadata: Additional metadata
    """
    subject_id: str
    recording_date: Optional[str] = None
    analysis_timestamp: str = field(
        default_factory=lambda: datetime.now().isoformat()
    )
    duration_seconds: float = 0.0
    frame_rate: float = 30.0
    pixel_to_mm: float = 1.0
    velocity_metrics: Optional[VelocityMetrics] = None
    coordination_metrics: Optional[CoordinationMetrics] = None
    kinematics_metrics: Optional[KinematicsMetrics] = None
    gait_metrics: Optional[GaitMetrics] = None
    quality_score: float = 0.0
    notes: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_complete(self) -> bool:
        """Check if all analysis components are present."""
        return all([
            self.velocity_metrics is not None,
            self.coordination_metrics is not None,
            self.kinematics_metrics is not None,
            self.gait_metrics is not None,
        ])

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of key metrics."""
        summary = {
            "subject_id": self.subject_id,
            "quality_score": self.quality_score,
            "duration_seconds": self.duration_seconds,
        }

        if self.velocity_metrics:
            summary["mean_velocity_mm_s"] = self.velocity_metrics.mean_velocity
            summary["drag_count"] = self.velocity_metrics.drag_count

        if self.coordination_metrics:
            summary["coordination_score"] = (
                self.coordination_metrics.overall_coordination_score
            )
            summary["gait_pattern"] = (
                self.coordination_metrics.coordination_pattern
            )

        if self.gait_metrics:
            summary["cadence_hz"] = self.gait_metrics.cadence
            summary["mean_stride_length_mm"] = self.gait_metrics.mean_stride_length
            summary["num_cycles"] = self.gait_metrics.num_cycles

        return summary

    def to_dict(self) -> Dict[str, Any]:
        """Convert complete report to dictionary."""
        return {
            "subject_id": self.subject_id,
            "recording_date": self.recording_date,
            "analysis_timestamp": self.analysis_timestamp,
            "duration_seconds": self.duration_seconds,
            "frame_rate_hz": self.frame_rate,
            "pixel_to_mm": self.pixel_to_mm,
            "quality_score": self.quality_score,
            "notes": self.notes,
            "metadata": self.metadata,
            "velocity": (
                self.velocity_metrics.to_dict()
                if self.velocity_metrics else None
            ),
            "coordination": (
                self.coordination_metrics.to_dict()
                if self.coordination_metrics else None
            ),
            "kinematics": (
                self.kinematics_metrics.to_dict()
                if self.kinematics_metrics else None
            ),
            "gait": (
                self.gait_metrics.to_dict()
                if self.gait_metrics else None
            ),
        }

    def add_note(self, note: str) -> None:
        """Add an analysis note."""
        self.notes.append(f"[{datetime.now().strftime('%H:%M:%S')}] {note}")
