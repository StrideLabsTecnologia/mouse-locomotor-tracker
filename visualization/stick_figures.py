"""
Stick Figure Generator
======================

Generates articulated stick figure visualizations for locomotion analysis.
Differentiates swing vs stance phases with color coding.

Author: Stride Labs
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass, field
from enum import Enum


class LimbPhase(Enum):
    """Gait phase enumeration."""
    SWING = "swing"
    STANCE = "stance"
    UNKNOWN = "unknown"


@dataclass
class JointStyle:
    """Style configuration for joint visualization."""
    radius: int = 6
    color: Tuple[int, int, int] = (255, 255, 255)
    thickness: int = -1  # -1 for filled
    glow: bool = True
    glow_radius: int = 12
    glow_color: Tuple[int, int, int] = (200, 200, 255)
    glow_intensity: float = 0.4


@dataclass
class BoneStyle:
    """Style configuration for bone (limb segment) visualization."""
    thickness: int = 4
    swing_color: Tuple[int, int, int] = (0, 0, 255)  # Red for swing
    stance_color: Tuple[int, int, int] = (128, 128, 128)  # Gray for stance
    unknown_color: Tuple[int, int, int] = (200, 200, 200)  # Light gray
    anti_aliased: bool = True


@dataclass
class SkeletonConfig:
    """Configuration for the entire skeleton."""
    joint_style: JointStyle = field(default_factory=JointStyle)
    bone_style: BoneStyle = field(default_factory=BoneStyle)

    # Skeleton structure: list of (joint1, joint2) connections
    connections: List[Tuple[str, str]] = field(default_factory=lambda: [
        # Front left leg
        ("hip_fl", "knee_fl"),
        ("knee_fl", "ankle_fl"),
        ("ankle_fl", "foot_fl"),
        # Front right leg
        ("hip_fr", "knee_fr"),
        ("knee_fr", "ankle_fr"),
        ("ankle_fr", "foot_fr"),
        # Hind left leg
        ("hip_hl", "knee_hl"),
        ("knee_hl", "ankle_hl"),
        ("ankle_hl", "foot_hl"),
        # Hind right leg
        ("hip_hr", "knee_hr"),
        ("knee_hr", "ankle_hr"),
        ("ankle_hr", "foot_hr"),
        # Body spine
        ("nose", "head"),
        ("head", "neck"),
        ("neck", "shoulder"),
        ("shoulder", "spine_mid"),
        ("spine_mid", "hip"),
        ("hip", "tail_base"),
        ("tail_base", "tail_mid"),
        ("tail_mid", "tail_tip"),
    ])

    # Limb groupings for phase detection
    limb_groups: Dict[str, List[str]] = field(default_factory=lambda: {
        "front_left": ["hip_fl", "knee_fl", "ankle_fl", "foot_fl"],
        "front_right": ["hip_fr", "knee_fr", "ankle_fr", "foot_fr"],
        "hind_left": ["hip_hl", "knee_hl", "ankle_hl", "foot_hl"],
        "hind_right": ["hip_hr", "knee_hr", "ankle_hr", "foot_hr"],
    })


class StickFigureGenerator:
    """
    Generates articulated stick figure visualizations.

    Creates professional stick figure animations showing limb positions
    with color-coded swing and stance phases, similar to biomechanical
    motion capture analysis.

    Attributes:
        config: Skeleton configuration
        frame_history: History of skeleton poses for animation
    """

    def __init__(self, config: Optional[SkeletonConfig] = None):
        """
        Initialize the stick figure generator.

        Args:
            config: Skeleton configuration
        """
        self.config = config or SkeletonConfig()
        self.frame_history: List[Dict[str, Tuple[float, float]]] = []
        self._glow_mask_cache: Dict[int, np.ndarray] = {}

    def _get_glow_mask(self, size: int) -> np.ndarray:
        """
        Get or create a cached glow mask.

        Args:
            size: Diameter of the glow mask

        Returns:
            2D glow mask array
        """
        if size not in self._glow_mask_cache:
            center = size // 2
            y, x = np.ogrid[:size, :size]
            distance = np.sqrt((x - center) ** 2 + (y - center) ** 2)
            mask = np.clip(1 - (distance / center), 0, 1)
            mask = (mask ** 2).astype(np.float32)
            self._glow_mask_cache[size] = mask
        return self._glow_mask_cache[size]

    def _draw_joint_glow(
        self,
        frame: np.ndarray,
        point: Tuple[int, int],
        style: JointStyle
    ) -> np.ndarray:
        """
        Draw a glowing joint marker.

        Args:
            frame: Input frame
            point: (x, y) joint position
            style: Joint style configuration

        Returns:
            Frame with glow effect
        """
        x, y = point
        glow_size = style.glow_radius * 2 + 1

        # Calculate ROI boundaries
        x1 = max(0, x - style.glow_radius)
        y1 = max(0, y - style.glow_radius)
        x2 = min(frame.shape[1], x + style.glow_radius + 1)
        y2 = min(frame.shape[0], y + style.glow_radius + 1)

        if x2 <= x1 or y2 <= y1:
            return frame

        # Get mask ROI
        mask = self._get_glow_mask(glow_size)
        mx1 = style.glow_radius - (x - x1)
        my1 = style.glow_radius - (y - y1)
        mx2 = mx1 + (x2 - x1)
        my2 = my1 + (y2 - y1)

        roi = frame[y1:y2, x1:x2].astype(np.float32)
        mask_roi = mask[my1:my2, mx1:mx2]

        # Apply glow
        glow_color = np.array(style.glow_color, dtype=np.float32)
        for c in range(3):
            roi[:, :, c] += mask_roi * glow_color[c] * style.glow_intensity

        frame[y1:y2, x1:x2] = np.clip(roi, 0, 255).astype(np.uint8)
        return frame

    def _get_limb_for_joint(self, joint_name: str) -> Optional[str]:
        """
        Get the limb group name for a joint.

        Args:
            joint_name: Name of the joint

        Returns:
            Limb group name or None
        """
        for limb_name, joints in self.config.limb_groups.items():
            if joint_name in joints:
                return limb_name
        return None

    def draw_skeleton(
        self,
        frame: np.ndarray,
        keypoints: Dict[str, Tuple[float, float]],
        limb_phases: Optional[Dict[str, LimbPhase]] = None,
        confidence: Optional[Dict[str, float]] = None
    ) -> np.ndarray:
        """
        Draw the skeleton on a frame.

        Args:
            frame: Input frame (BGR)
            keypoints: Dictionary mapping joint names to (x, y) positions
            limb_phases: Optional dictionary mapping limb names to phases
            confidence: Optional confidence scores for each joint

        Returns:
            Frame with skeleton drawn
        """
        output = frame.copy()
        limb_phases = limb_phases or {}
        confidence = confidence or {}

        # Draw bones first (underneath joints)
        for joint1, joint2 in self.config.connections:
            if joint1 not in keypoints or joint2 not in keypoints:
                continue

            # Get confidence threshold
            conf1 = confidence.get(joint1, 1.0)
            conf2 = confidence.get(joint2, 1.0)
            if conf1 < 0.3 or conf2 < 0.3:
                continue

            pt1 = (int(keypoints[joint1][0]), int(keypoints[joint1][1]))
            pt2 = (int(keypoints[joint2][0]), int(keypoints[joint2][1]))

            # Determine color based on limb phase
            limb1 = self._get_limb_for_joint(joint1)
            limb2 = self._get_limb_for_joint(joint2)

            # Use phase if both joints belong to same limb
            phase = LimbPhase.UNKNOWN
            if limb1 and limb1 == limb2:
                phase = limb_phases.get(limb1, LimbPhase.UNKNOWN)

            if phase == LimbPhase.SWING:
                color = self.config.bone_style.swing_color
            elif phase == LimbPhase.STANCE:
                color = self.config.bone_style.stance_color
            else:
                color = self.config.bone_style.unknown_color

            # Draw bone
            line_type = cv2.LINE_AA if self.config.bone_style.anti_aliased else cv2.LINE_8
            cv2.line(output, pt1, pt2, color, self.config.bone_style.thickness, line_type)

        # Draw joints on top
        for joint_name, (x, y) in keypoints.items():
            conf = confidence.get(joint_name, 1.0)
            if conf < 0.3:
                continue

            point = (int(x), int(y))

            # Check bounds
            if point[0] < 0 or point[1] < 0:
                continue
            if point[0] >= frame.shape[1] or point[1] >= frame.shape[0]:
                continue

            # Draw glow effect
            if self.config.joint_style.glow:
                output = self._draw_joint_glow(output, point, self.config.joint_style)

            # Draw joint
            cv2.circle(
                output, point,
                self.config.joint_style.radius,
                self.config.joint_style.color,
                self.config.joint_style.thickness,
                cv2.LINE_AA
            )

        return output

    def generate_frame(
        self,
        base_frame: np.ndarray,
        keypoints: Dict[str, Tuple[float, float]],
        limb_phases: Optional[Dict[str, LimbPhase]] = None,
        add_to_history: bool = True
    ) -> np.ndarray:
        """
        Generate a single frame with skeleton overlay.

        Args:
            base_frame: Input video frame
            keypoints: Joint positions
            limb_phases: Gait phase for each limb
            add_to_history: Whether to store in history

        Returns:
            Frame with skeleton overlay
        """
        if add_to_history:
            self.frame_history.append(keypoints.copy())
            # Limit history
            if len(self.frame_history) > 1000:
                self.frame_history = self.frame_history[-500:]

        return self.draw_skeleton(base_frame, keypoints, limb_phases)

    def animate_sequence(
        self,
        frame_size: Tuple[int, int],
        keypoints_sequence: List[Dict[str, Tuple[float, float]]],
        limb_phases_sequence: Optional[List[Dict[str, LimbPhase]]] = None,
        sync_mode: str = "temporal",
        num_sticks: int = 20,
        spacing: float = 0.05,
        background_color: Tuple[int, int, int] = (30, 30, 40)
    ) -> np.ndarray:
        """
        Create an animated sequence visualization (stick figure parade).

        Args:
            frame_size: (width, height) of output frame
            keypoints_sequence: List of keypoint dictionaries over time
            limb_phases_sequence: Optional list of phase dictionaries
            sync_mode: "temporal" for time-synced, "spatial" for position-synced
            num_sticks: Number of stick figures to display
            spacing: Horizontal spacing between figures
            background_color: Background color (BGR)

        Returns:
            Composite frame showing the animation sequence
        """
        width, height = frame_size
        output = np.zeros((height, width, 3), dtype=np.uint8)
        output[:] = background_color

        if len(keypoints_sequence) < 2:
            return output

        limb_phases_sequence = limb_phases_sequence or [{}] * len(keypoints_sequence)

        # Select frames to display
        n_frames = len(keypoints_sequence)
        indices = np.linspace(0, n_frames - 1, min(num_sticks, n_frames)).astype(int)

        # Calculate bounding box of all poses
        all_x = []
        all_y = []
        for kp in keypoints_sequence:
            for x, y in kp.values():
                all_x.append(x)
                all_y.append(y)

        if not all_x:
            return output

        min_x, max_x = min(all_x), max(all_x)
        min_y, max_y = min(all_y), max(all_y)
        pose_width = max_x - min_x if max_x > min_x else 100
        pose_height = max_y - min_y if max_y > min_y else 100

        # Scale to fit
        scale = min(height * 0.8 / pose_height, width / (num_sticks * pose_width * 1.5))
        scale = max(scale, 0.1)  # Minimum scale

        if sync_mode == "temporal":
            # Time-synchronized: evenly spaced horizontally
            x_positions = np.linspace(
                pose_width * scale * 0.5,
                width - pose_width * scale * 0.5,
                len(indices)
            )
        else:
            # Spatial-synchronized: based on foot position
            x_positions = []
            for idx in indices:
                kp = keypoints_sequence[idx]
                # Use average foot position
                foot_x = np.mean([
                    kp.get("foot_fl", (0, 0))[0],
                    kp.get("foot_fr", (0, 0))[0],
                    kp.get("foot_hl", (0, 0))[0],
                    kp.get("foot_hr", (0, 0))[0],
                ])
                x_positions.append(foot_x * scale)

        # Draw each stick figure
        for i, idx in enumerate(indices):
            kp = keypoints_sequence[idx]
            phases = limb_phases_sequence[idx]

            # Transform keypoints
            x_offset = x_positions[i] if sync_mode == "temporal" else 50 + i * pose_width * scale * spacing
            y_offset = height * 0.9  # Bottom anchor

            transformed_kp = {}
            for joint_name, (x, y) in kp.items():
                # Normalize and scale
                nx = (x - min_x) * scale
                ny = (y - min_y) * scale

                # Position in frame
                tx = nx + x_offset - (pose_width * scale / 2)
                ty = y_offset - ny

                transformed_kp[joint_name] = (tx, ty)

            # Draw skeleton with alpha based on temporal position
            t = i / max(len(indices) - 1, 1)
            alpha = 0.3 + 0.7 * t  # Fade from 0.3 to 1.0

            skeleton_frame = np.zeros_like(output)
            skeleton_frame = self.draw_skeleton(skeleton_frame, transformed_kp, phases)

            # Blend
            cv2.addWeighted(skeleton_frame, alpha, output, 1.0, 0, output)

        return output

    def create_overlay_sequence(
        self,
        base_frame: np.ndarray,
        keypoints_sequence: List[Dict[str, Tuple[float, float]]],
        limb_phases_sequence: Optional[List[Dict[str, LimbPhase]]] = None,
        max_overlays: int = 10,
        fade_factor: float = 0.7
    ) -> np.ndarray:
        """
        Create an overlay showing multiple time points on a single frame.

        Args:
            base_frame: Background frame
            keypoints_sequence: List of keypoint dictionaries
            limb_phases_sequence: Optional phase information
            max_overlays: Maximum number of poses to overlay
            fade_factor: Alpha reduction per historical frame

        Returns:
            Frame with overlaid poses
        """
        output = base_frame.copy()

        if len(keypoints_sequence) < 1:
            return output

        limb_phases_sequence = limb_phases_sequence or [{}] * len(keypoints_sequence)

        # Select frames
        n_frames = len(keypoints_sequence)
        step = max(1, n_frames // max_overlays)
        indices = list(range(0, n_frames, step))[-max_overlays:]

        # Draw from oldest to newest
        for i, idx in enumerate(indices):
            kp = keypoints_sequence[idx]
            phases = limb_phases_sequence[idx]

            # Calculate alpha
            t = i / max(len(indices) - 1, 1)
            alpha = 0.2 + 0.8 * t

            # Reduce bone thickness for older frames
            original_thickness = self.config.bone_style.thickness
            self.config.bone_style.thickness = max(1, int(original_thickness * (0.5 + 0.5 * t)))

            # Draw with reduced style for older frames
            skeleton_frame = np.zeros_like(output)
            skeleton_frame = self.draw_skeleton(skeleton_frame, kp, phases)

            # Blend
            cv2.addWeighted(skeleton_frame, alpha, output, 1.0, 0, output)

            self.config.bone_style.thickness = original_thickness

        return output

    def detect_limb_phases(
        self,
        keypoints: Dict[str, Tuple[float, float]],
        floor_y: Optional[float] = None,
        velocity: Optional[Dict[str, Tuple[float, float]]] = None
    ) -> Dict[str, LimbPhase]:
        """
        Detect swing vs stance phase for each limb.

        Args:
            keypoints: Current joint positions
            floor_y: Y-coordinate of the floor (higher value = lower on screen)
            velocity: Optional velocity vectors for each joint

        Returns:
            Dictionary mapping limb names to phases
        """
        phases = {}

        for limb_name, joints in self.config.limb_groups.items():
            # Get foot position (last joint in the group)
            foot_joint = joints[-1] if joints else None
            if foot_joint and foot_joint in keypoints:
                foot_x, foot_y = keypoints[foot_joint]

                # Simple heuristic: if foot is above floor threshold, it's swing
                if floor_y is not None:
                    threshold = floor_y - 20  # 20 pixels above floor
                    if foot_y < threshold:
                        phases[limb_name] = LimbPhase.SWING
                    else:
                        phases[limb_name] = LimbPhase.STANCE
                elif velocity and foot_joint in velocity:
                    # Use velocity: high vertical velocity = swing
                    vx, vy = velocity[foot_joint]
                    if abs(vy) > 5:  # Threshold for swing detection
                        phases[limb_name] = LimbPhase.SWING
                    else:
                        phases[limb_name] = LimbPhase.STANCE
                else:
                    phases[limb_name] = LimbPhase.UNKNOWN
            else:
                phases[limb_name] = LimbPhase.UNKNOWN

        return phases


def demo_stick_figure():
    """
    Demonstrate stick figure generation with synthetic data.
    """
    import math

    width, height = 800, 600
    generator = StickFigureGenerator()

    # Generate synthetic walking motion
    n_frames = 120
    keypoints_sequence = []
    phases_sequence = []

    for i in range(n_frames):
        t = i / n_frames
        cycle = t * 4 * math.pi  # Two full gait cycles

        # Base position
        base_x = 200 + t * 400
        base_y = height * 0.7

        # Simple oscillating limb positions
        keypoints = {
            # Body
            "nose": (base_x + 60, base_y - 40),
            "head": (base_x + 40, base_y - 35),
            "neck": (base_x + 20, base_y - 30),
            "shoulder": (base_x, base_y - 25),
            "spine_mid": (base_x - 30, base_y - 20),
            "hip": (base_x - 60, base_y - 15),
            "tail_base": (base_x - 80, base_y - 10),
            "tail_mid": (base_x - 100, base_y - 5 + 10 * math.sin(cycle)),
            "tail_tip": (base_x - 120, base_y + 10 * math.sin(cycle + 0.5)),

            # Front left (diagonal pair with hind right)
            "hip_fl": (base_x + 10, base_y - 20),
            "knee_fl": (base_x + 15, base_y + 10 + 15 * math.sin(cycle)),
            "ankle_fl": (base_x + 20, base_y + 35 + 20 * math.sin(cycle)),
            "foot_fl": (base_x + 25 + 10 * math.cos(cycle), base_y + 50 + max(0, 25 * math.sin(cycle))),

            # Front right
            "hip_fr": (base_x + 10, base_y - 20),
            "knee_fr": (base_x + 15, base_y + 10 + 15 * math.sin(cycle + math.pi)),
            "ankle_fr": (base_x + 20, base_y + 35 + 20 * math.sin(cycle + math.pi)),
            "foot_fr": (base_x + 25 + 10 * math.cos(cycle + math.pi), base_y + 50 + max(0, 25 * math.sin(cycle + math.pi))),

            # Hind left
            "hip_hl": (base_x - 50, base_y - 10),
            "knee_hl": (base_x - 45, base_y + 15 + 15 * math.sin(cycle + math.pi)),
            "ankle_hl": (base_x - 40, base_y + 40 + 20 * math.sin(cycle + math.pi)),
            "foot_hl": (base_x - 35 + 10 * math.cos(cycle + math.pi), base_y + 55 + max(0, 25 * math.sin(cycle + math.pi))),

            # Hind right (diagonal pair with front left)
            "hip_hr": (base_x - 50, base_y - 10),
            "knee_hr": (base_x - 45, base_y + 15 + 15 * math.sin(cycle)),
            "ankle_hr": (base_x - 40, base_y + 40 + 20 * math.sin(cycle)),
            "foot_hr": (base_x - 35 + 10 * math.cos(cycle), base_y + 55 + max(0, 25 * math.sin(cycle))),
        }

        # Determine phases
        phases = {
            "front_left": LimbPhase.SWING if math.sin(cycle) > 0.3 else LimbPhase.STANCE,
            "front_right": LimbPhase.SWING if math.sin(cycle + math.pi) > 0.3 else LimbPhase.STANCE,
            "hind_left": LimbPhase.SWING if math.sin(cycle + math.pi) > 0.3 else LimbPhase.STANCE,
            "hind_right": LimbPhase.SWING if math.sin(cycle) > 0.3 else LimbPhase.STANCE,
        }

        keypoints_sequence.append(keypoints)
        phases_sequence.append(phases)

    # Animate
    print("Generating stick figure demo...")
    for i in range(n_frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:] = (30, 30, 40)

        # Draw current skeleton
        result = generator.generate_frame(frame, keypoints_sequence[i], phases_sequence[i])

        # Add title
        cv2.putText(result, "Stick Figure Demo - Swing (Red) / Stance (Gray)",
                   (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

        cv2.imshow("Stick Figure Demo", result)
        key = cv2.waitKey(50)
        if key == 27:
            break

    # Show animation sequence
    print("Generating animation sequence...")
    sequence_frame = generator.animate_sequence(
        (width, height),
        keypoints_sequence,
        phases_sequence,
        sync_mode="temporal",
        num_sticks=15
    )

    cv2.putText(sequence_frame, "Temporal Stick Figure Sequence",
               (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

    cv2.imshow("Stick Figure Demo", sequence_frame)
    cv2.waitKey(0)

    cv2.destroyAllWindows()
    print("Demo complete!")


if __name__ == "__main__":
    demo_stick_figure()
