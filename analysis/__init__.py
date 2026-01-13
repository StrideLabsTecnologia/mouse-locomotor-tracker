"""
Mouse Locomotor Tracker - Analysis Module
==========================================

Biomechanical analysis tools for mouse locomotion tracking data.

This module provides comprehensive analysis capabilities including:
- Velocity and acceleration analysis
- Interlimb coordination using circular statistics
- Joint angle kinematics
- Gait cycle detection and temporal-spatial parameters

Based on Locomotor-Allodi2021 methodology with enhancements.

Author: Stride Labs - Mouse Locomotor Tracker

Example Usage:
--------------
    >>> from analysis import (
    ...     VelocityAnalyzer,
    ...     CircularCoordinationAnalyzer,
    ...     JointAngleAnalyzer,
    ...     GaitCycleDetector,
    ...     LocomotorReport,
    ... )
    >>>
    >>> # Analyze velocity
    >>> velocity_analyzer = VelocityAnalyzer(frame_rate=30, pixel_to_mm=0.1)
    >>> velocity_metrics = velocity_analyzer.analyze(x_coords, y_coords)
    >>>
    >>> # Analyze coordination
    >>> coord_analyzer = CircularCoordinationAnalyzer()
    >>> coord_metrics = coord_analyzer.analyze_all_pairs(limb_phases)
    >>>
    >>> # Analyze kinematics
    >>> joint_analyzer = JointAngleAnalyzer(frame_rate=30)
    >>> joint_metrics = joint_analyzer.analyze_joint(p1, p2, p3)
    >>>
    >>> # Detect gait cycles
    >>> gait_detector = GaitCycleDetector(frame_rate=30)
    >>> gait_metrics = gait_detector.detect_cycles(toe_y_position)
    >>>
    >>> # Create comprehensive report
    >>> report = LocomotorReport(
    ...     subject_id="mouse_001",
    ...     velocity_metrics=velocity_metrics,
    ...     coordination_metrics=coord_metrics,
    ...     kinematics_metrics=kinematics_metrics,
    ...     gait_metrics=gait_metrics,
    ... )
"""

from .metrics import (
    # Enums
    LimbPair,
    JointType,
    LocomotionPhase,
    # Velocity metrics
    VelocityMetrics,
    # Coordination metrics
    CircularStatistics,
    CoordinationMetrics,
    # Kinematics metrics
    JointAngleMetrics,
    KinematicsMetrics,
    # Gait metrics
    GaitCycleInfo,
    GaitMetrics,
    # Complete report
    LocomotorReport,
)

from .velocity import (
    VelocityAnalyzer,
    create_velocity_analyzer,
)

from .coordination import (
    CircularCoordinationAnalyzer,
    create_coordination_analyzer,
    extract_phase_from_position,
    compute_phase_coherence,
    NORMAL_PHASE_RELATIONSHIPS,
)

from .kinematics import (
    JointAngleAnalyzer,
    create_joint_analyzer,
    calculate_limb_endpoint_trajectory,
    JOINT_MARKER_DEFINITIONS,
)

from .gait_cycles import (
    GaitCycleDetector,
    create_gait_detector,
    detect_footfall_events,
)


# Package metadata
__version__ = "1.0.0"
__author__ = "Stride Labs"
__email__ = "gvillegas@stridelabs.cl"


# Convenience function to create all analyzers with consistent parameters
def create_analysis_pipeline(
    frame_rate: float = 30.0,
    pixel_to_mm: float = 1.0,
    config: dict = None,
) -> dict:
    """
    Create a complete analysis pipeline with all analyzers.

    Args:
        frame_rate: Video frame rate in Hz
        pixel_to_mm: Pixel to millimeter conversion factor
        config: Optional configuration dictionary with keys:
            - velocity: VelocityAnalyzer config
            - coordination: CircularCoordinationAnalyzer config
            - kinematics: JointAngleAnalyzer config
            - gait: GaitCycleDetector config

    Returns:
        Dictionary with all analyzer instances:
            - velocity: VelocityAnalyzer
            - coordination: CircularCoordinationAnalyzer
            - kinematics: JointAngleAnalyzer
            - gait: GaitCycleDetector
    """
    if config is None:
        config = {}

    return {
        'velocity': create_velocity_analyzer(
            frame_rate=frame_rate,
            pixel_to_mm=pixel_to_mm,
            config=config.get('velocity', {}),
        ),
        'coordination': create_coordination_analyzer(
            config={
                'frame_rate': frame_rate,
                **config.get('coordination', {}),
            }
        ),
        'kinematics': create_joint_analyzer(
            frame_rate=frame_rate,
            pixel_to_mm=pixel_to_mm,
            config=config.get('kinematics', {}),
        ),
        'gait': create_gait_detector(
            frame_rate=frame_rate,
            pixel_to_mm=pixel_to_mm,
            config=config.get('gait', {}),
        ),
    }


def analyze_locomotion(
    tracking_data: dict,
    frame_rate: float = 30.0,
    pixel_to_mm: float = 1.0,
    subject_id: str = "unknown",
) -> LocomotorReport:
    """
    Perform complete locomotor analysis on tracking data.

    This is a high-level convenience function that runs all analyses
    and returns a comprehensive report.

    Args:
        tracking_data: Dictionary containing:
            - 'x': X coordinates of body center
            - 'y': Y coordinates of body center
            - 'markers': Dict of marker coordinates (optional)
            - 'limb_phases': Dict of limb phases (optional)
            - 'toe_y': Toe vertical position (optional)
        frame_rate: Video frame rate in Hz
        pixel_to_mm: Pixel to mm conversion
        subject_id: Subject identifier

    Returns:
        LocomotorReport with all analysis results

    Example:
        >>> tracking_data = {
        ...     'x': x_coords,
        ...     'y': y_coords,
        ...     'toe_y': toe_vertical,
        ...     'markers': {'hip': hip_coords, 'knee': knee_coords, ...}
        ... }
        >>> report = analyze_locomotion(tracking_data, frame_rate=30)
        >>> print(report.get_summary())
    """
    import numpy as np

    # Create analyzers
    pipeline = create_analysis_pipeline(frame_rate, pixel_to_mm)

    # Initialize report
    report = LocomotorReport(
        subject_id=subject_id,
        frame_rate=frame_rate,
        pixel_to_mm=pixel_to_mm,
    )

    # Calculate duration
    if 'x' in tracking_data:
        n_frames = len(tracking_data['x'])
        report.duration_seconds = n_frames / frame_rate

    # Velocity analysis
    if 'x' in tracking_data and 'y' in tracking_data:
        try:
            report.velocity_metrics = pipeline['velocity'].analyze(
                tracking_data['x'],
                tracking_data['y'],
            )
            report.add_note("Velocity analysis completed successfully")
        except Exception as e:
            report.add_note(f"Velocity analysis failed: {e}")

    # Coordination analysis
    if 'limb_phases' in tracking_data:
        try:
            report.coordination_metrics = pipeline['coordination'].analyze_all_pairs(
                tracking_data['limb_phases']
            )
            report.add_note("Coordination analysis completed successfully")
        except Exception as e:
            report.add_note(f"Coordination analysis failed: {e}")

    # Kinematics analysis
    if 'markers' in tracking_data:
        try:
            report.kinematics_metrics = pipeline['kinematics'].analyze_all_joints(
                tracking_data['markers']
            )
            report.add_note("Kinematics analysis completed successfully")
        except Exception as e:
            report.add_note(f"Kinematics analysis failed: {e}")

    # Gait cycle analysis
    if 'toe_y' in tracking_data:
        x_pos = tracking_data.get('x')
        try:
            report.gait_metrics = pipeline['gait'].detect_cycles(
                tracking_data['toe_y'],
                x_position=x_pos,
            )
            report.add_note("Gait analysis completed successfully")
        except Exception as e:
            report.add_note(f"Gait analysis failed: {e}")

    # Calculate quality score
    components_complete = sum([
        report.velocity_metrics is not None,
        report.coordination_metrics is not None,
        report.kinematics_metrics is not None,
        report.gait_metrics is not None,
    ])
    report.quality_score = (components_complete / 4.0) * 100

    return report


# Export all public symbols
__all__ = [
    # Version info
    "__version__",
    "__author__",
    # Enums
    "LimbPair",
    "JointType",
    "LocomotionPhase",
    # Metrics dataclasses
    "VelocityMetrics",
    "CircularStatistics",
    "CoordinationMetrics",
    "JointAngleMetrics",
    "KinematicsMetrics",
    "GaitCycleInfo",
    "GaitMetrics",
    "LocomotorReport",
    # Analyzers
    "VelocityAnalyzer",
    "CircularCoordinationAnalyzer",
    "JointAngleAnalyzer",
    "GaitCycleDetector",
    # Factory functions
    "create_velocity_analyzer",
    "create_coordination_analyzer",
    "create_joint_analyzer",
    "create_gait_detector",
    "create_analysis_pipeline",
    # Utility functions
    "extract_phase_from_position",
    "compute_phase_coherence",
    "calculate_limb_endpoint_trajectory",
    "detect_footfall_events",
    "analyze_locomotion",
    # Constants
    "NORMAL_PHASE_RELATIONSHIPS",
    "JOINT_MARKER_DEFINITIONS",
]
