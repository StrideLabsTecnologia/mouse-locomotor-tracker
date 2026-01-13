"""
Trajectory Overlay Visualization
================================

Professional motion capture-style trajectory visualization with glow effects,
gradient trails, and customizable appearance.

Author: Stride Labs
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass, field
from enum import Enum


class ColorScheme(Enum):
    """Predefined color schemes for trajectories."""
    MOTION_CAPTURE = "motion_capture"  # Blue -> Green -> Yellow
    HEAT = "heat"  # Blue -> Red
    NEON = "neon"  # Cyan -> Magenta
    VELOCITY = "velocity"  # Green (slow) -> Red (fast)


@dataclass
class TrailConfig:
    """Configuration for trajectory trail appearance."""
    length: int = 30  # Number of points in the trail
    min_alpha: float = 0.1  # Minimum transparency at tail
    max_alpha: float = 1.0  # Maximum transparency at head
    base_thickness: int = 2  # Base line thickness
    max_thickness: int = 6  # Maximum line thickness at head
    color_scheme: ColorScheme = ColorScheme.MOTION_CAPTURE

    # Glow effect settings
    enable_glow: bool = True
    glow_radius: int = 15
    glow_intensity: float = 0.6
    glow_layers: int = 5


@dataclass
class MarkerConfig:
    """Configuration for keypoint markers."""
    radius: int = 8
    inner_radius: int = 4
    outer_color: Tuple[int, int, int] = (0, 255, 255)  # Cyan
    inner_color: Tuple[int, int, int] = (255, 255, 255)  # White
    enable_glow: bool = True
    glow_radius: int = 20
    glow_color: Tuple[int, int, int] = (0, 200, 255)  # Orange-yellow
    glow_intensity: float = 0.5


class TrajectoryVisualizer:
    """
    Professional motion capture-style trajectory visualizer.

    Creates visually stunning trajectory overlays with gradient trails,
    glow effects, and smooth animations suitable for scientific publications
    and presentations.

    Attributes:
        trail_config: Configuration for trail appearance
        marker_config: Configuration for keypoint markers
        history: Dictionary storing trajectory history for each keypoint
    """

    def __init__(
        self,
        trail_config: Optional[TrailConfig] = None,
        marker_config: Optional[MarkerConfig] = None
    ):
        """
        Initialize the trajectory visualizer.

        Args:
            trail_config: Trail appearance configuration
            marker_config: Marker appearance configuration
        """
        self.trail_config = trail_config or TrailConfig()
        self.marker_config = marker_config or MarkerConfig()
        self.history: Dict[str, List[Tuple[float, float]]] = {}

        # Precompute color gradients
        self._color_gradient = self._generate_color_gradient()

    def _generate_color_gradient(self) -> np.ndarray:
        """
        Generate color gradient based on the selected color scheme.

        Returns:
            Array of BGR colors for the trail gradient
        """
        n_colors = self.trail_config.length
        gradient = np.zeros((n_colors, 3), dtype=np.uint8)

        if self.trail_config.color_scheme == ColorScheme.MOTION_CAPTURE:
            # Blue -> Green -> Yellow (classic motion capture look)
            for i in range(n_colors):
                t = i / max(n_colors - 1, 1)
                if t < 0.5:
                    # Blue to Green
                    t2 = t * 2
                    gradient[i] = [
                        int(255 * (1 - t2)),  # B: 255 -> 0
                        int(255 * t2),         # G: 0 -> 255
                        0                       # R: 0
                    ]
                else:
                    # Green to Yellow
                    t2 = (t - 0.5) * 2
                    gradient[i] = [
                        0,                      # B: 0
                        255,                    # G: 255
                        int(255 * t2)           # R: 0 -> 255
                    ]

        elif self.trail_config.color_scheme == ColorScheme.HEAT:
            # Blue -> Purple -> Red
            for i in range(n_colors):
                t = i / max(n_colors - 1, 1)
                gradient[i] = [
                    int(255 * (1 - t)),  # B
                    0,                    # G
                    int(255 * t)          # R
                ]

        elif self.trail_config.color_scheme == ColorScheme.NEON:
            # Cyan -> Magenta
            for i in range(n_colors):
                t = i / max(n_colors - 1, 1)
                gradient[i] = [
                    int(255 * (1 - t * 0.5)),  # B
                    int(255 * (1 - t)),         # G
                    int(255 * t)                # R
                ]

        elif self.trail_config.color_scheme == ColorScheme.VELOCITY:
            # Green -> Yellow -> Red (velocity based)
            for i in range(n_colors):
                t = i / max(n_colors - 1, 1)
                if t < 0.5:
                    t2 = t * 2
                    gradient[i] = [0, 255, int(255 * t2)]
                else:
                    t2 = (t - 0.5) * 2
                    gradient[i] = [0, int(255 * (1 - t2)), 255]

        return gradient

    def _create_glow_mask(
        self,
        size: int,
        intensity: float = 1.0
    ) -> np.ndarray:
        """
        Create a circular glow mask for additive blending.

        Args:
            size: Diameter of the glow mask
            intensity: Glow intensity (0.0 to 1.0)

        Returns:
            2D array representing the glow mask
        """
        center = size // 2
        y, x = np.ogrid[:size, :size]
        distance = np.sqrt((x - center) ** 2 + (y - center) ** 2)
        mask = np.clip(1 - (distance / center), 0, 1)
        mask = (mask ** 2) * intensity  # Quadratic falloff for softer glow
        return mask.astype(np.float32)

    def draw_marker_with_glow(
        self,
        frame: np.ndarray,
        point: Tuple[float, float],
        config: Optional[MarkerConfig] = None
    ) -> np.ndarray:
        """
        Draw a keypoint marker with professional glow effect.

        Args:
            frame: Input frame (BGR)
            point: (x, y) coordinates of the marker
            config: Optional marker configuration override

        Returns:
            Frame with the marker drawn
        """
        config = config or self.marker_config
        x, y = int(point[0]), int(point[1])

        # Check bounds
        if x < 0 or y < 0 or x >= frame.shape[1] or y >= frame.shape[0]:
            return frame

        output = frame.copy()

        if config.enable_glow:
            # Create glow effect using additive blending
            glow_size = config.glow_radius * 2 + 1
            glow_mask = self._create_glow_mask(glow_size, config.glow_intensity)

            # Calculate ROI boundaries
            x1 = max(0, x - config.glow_radius)
            y1 = max(0, y - config.glow_radius)
            x2 = min(frame.shape[1], x + config.glow_radius + 1)
            y2 = min(frame.shape[0], y + config.glow_radius + 1)

            # Adjust mask boundaries
            mx1 = config.glow_radius - (x - x1)
            my1 = config.glow_radius - (y - y1)
            mx2 = mx1 + (x2 - x1)
            my2 = my1 + (y2 - y1)

            if mx2 > mx1 and my2 > my1:
                roi = output[y1:y2, x1:x2].astype(np.float32)
                mask_roi = glow_mask[my1:my2, mx1:mx2]

                # Additive blending for glow
                glow_color = np.array(config.glow_color, dtype=np.float32)
                for c in range(3):
                    roi[:, :, c] += mask_roi * glow_color[c]

                output[y1:y2, x1:x2] = np.clip(roi, 0, 255).astype(np.uint8)

        # Draw outer ring
        cv2.circle(output, (x, y), config.radius, config.outer_color, -1, cv2.LINE_AA)

        # Draw inner dot
        cv2.circle(output, (x, y), config.inner_radius, config.inner_color, -1, cv2.LINE_AA)

        return output

    def draw_trail(
        self,
        frame: np.ndarray,
        points: List[Tuple[float, float]],
        config: Optional[TrailConfig] = None
    ) -> np.ndarray:
        """
        Draw a gradient trail with optional glow effect.

        Args:
            frame: Input frame (BGR)
            points: List of (x, y) coordinates from oldest to newest
            config: Optional trail configuration override

        Returns:
            Frame with the trail drawn
        """
        config = config or self.trail_config

        if len(points) < 2:
            return frame

        output = frame.copy()
        n_points = len(points)

        # Limit to trail length
        if n_points > config.length:
            points = points[-config.length:]
            n_points = len(points)

        if config.enable_glow:
            # Create glow layer first (underneath the main trail)
            glow_layer = np.zeros_like(frame, dtype=np.float32)

            for i in range(n_points - 1):
                t = i / max(n_points - 2, 1)
                alpha = config.min_alpha + t * (config.max_alpha - config.min_alpha)

                pt1 = (int(points[i][0]), int(points[i][1]))
                pt2 = (int(points[i + 1][0]), int(points[i + 1][1]))

                # Get color from gradient
                color_idx = int(t * (len(self._color_gradient) - 1))
                color = self._color_gradient[color_idx].astype(np.float32)

                # Draw thick line for glow
                thickness = int(config.base_thickness +
                              t * (config.max_thickness - config.base_thickness))
                glow_thickness = thickness + config.glow_radius

                cv2.line(glow_layer, pt1, pt2,
                        color * alpha * config.glow_intensity,
                        glow_thickness, cv2.LINE_AA)

            # Apply Gaussian blur for soft glow
            glow_layer = cv2.GaussianBlur(glow_layer, (0, 0), config.glow_radius / 2)

            # Add glow to output
            output = np.clip(output.astype(np.float32) + glow_layer, 0, 255).astype(np.uint8)

        # Draw main trail segments
        for i in range(n_points - 1):
            t = i / max(n_points - 2, 1)
            alpha = config.min_alpha + t * (config.max_alpha - config.min_alpha)

            pt1 = (int(points[i][0]), int(points[i][1]))
            pt2 = (int(points[i + 1][0]), int(points[i + 1][1]))

            # Get color from gradient
            color_idx = int(t * (len(self._color_gradient) - 1))
            color = tuple(int(c) for c in self._color_gradient[color_idx])

            # Calculate thickness
            thickness = int(config.base_thickness +
                          t * (config.max_thickness - config.base_thickness))

            # Draw with alpha blending
            overlay = output.copy()
            cv2.line(overlay, pt1, pt2, color, thickness, cv2.LINE_AA)
            cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)

        return output

    def update_history(
        self,
        keypoint_id: str,
        point: Tuple[float, float]
    ) -> None:
        """
        Update the trajectory history for a keypoint.

        Args:
            keypoint_id: Unique identifier for the keypoint
            point: New (x, y) position
        """
        if keypoint_id not in self.history:
            self.history[keypoint_id] = []

        self.history[keypoint_id].append(point)

        # Limit history length
        if len(self.history[keypoint_id]) > self.trail_config.length:
            self.history[keypoint_id] = self.history[keypoint_id][-self.trail_config.length:]

    def clear_history(self, keypoint_id: Optional[str] = None) -> None:
        """
        Clear trajectory history.

        Args:
            keypoint_id: Specific keypoint to clear, or None for all
        """
        if keypoint_id:
            self.history.pop(keypoint_id, None)
        else:
            self.history.clear()

    def update_frame(
        self,
        frame: np.ndarray,
        keypoints: Dict[str, Tuple[float, float]],
        draw_markers: bool = True,
        draw_trails: bool = True
    ) -> np.ndarray:
        """
        Update frame with trajectory overlays for all keypoints.

        Args:
            frame: Input frame (BGR)
            keypoints: Dictionary mapping keypoint IDs to (x, y) positions
            draw_markers: Whether to draw markers at current positions
            draw_trails: Whether to draw trajectory trails

        Returns:
            Frame with all overlays applied
        """
        output = frame.copy()

        # Update history and draw trails
        for keypoint_id, point in keypoints.items():
            self.update_history(keypoint_id, point)

            if draw_trails:
                trail = self.history.get(keypoint_id, [])
                if len(trail) >= 2:
                    output = self.draw_trail(output, trail)

        # Draw markers on top
        if draw_markers:
            for keypoint_id, point in keypoints.items():
                output = self.draw_marker_with_glow(output, point)

        return output

    def create_composite_trail(
        self,
        frame: np.ndarray,
        all_points: Dict[str, List[Tuple[float, float]]],
        keypoint_colors: Optional[Dict[str, Tuple[int, int, int]]] = None
    ) -> np.ndarray:
        """
        Create a composite visualization with multiple colored trails.

        Args:
            frame: Input frame (BGR)
            all_points: Dictionary mapping keypoint IDs to lists of positions
            keypoint_colors: Optional custom colors for each keypoint

        Returns:
            Frame with all trails overlaid
        """
        output = frame.copy()

        # Default colors for different keypoints (motion capture style)
        default_colors = {
            "front_left": (255, 100, 100),   # Blue-ish
            "front_right": (100, 255, 100),  # Green-ish
            "hind_left": (100, 100, 255),    # Red-ish
            "hind_right": (255, 255, 100),   # Cyan-ish
            "nose": (255, 200, 100),         # Light blue
            "tail_base": (100, 200, 255),    # Orange
        }

        colors = keypoint_colors or default_colors

        for keypoint_id, points in all_points.items():
            if len(points) < 2:
                continue

            color = colors.get(keypoint_id, (200, 200, 200))

            # Create custom gradient for this keypoint
            custom_config = TrailConfig(
                length=self.trail_config.length,
                min_alpha=self.trail_config.min_alpha,
                max_alpha=self.trail_config.max_alpha,
                base_thickness=self.trail_config.base_thickness,
                max_thickness=self.trail_config.max_thickness,
                enable_glow=self.trail_config.enable_glow,
                glow_radius=self.trail_config.glow_radius,
                glow_intensity=self.trail_config.glow_intensity,
            )

            # Draw trail with custom color
            n_points = min(len(points), custom_config.length)
            trail_points = points[-n_points:]

            for i in range(len(trail_points) - 1):
                t = i / max(len(trail_points) - 2, 1)
                alpha = custom_config.min_alpha + t * (custom_config.max_alpha - custom_config.min_alpha)

                pt1 = (int(trail_points[i][0]), int(trail_points[i][1]))
                pt2 = (int(trail_points[i + 1][0]), int(trail_points[i + 1][1]))

                thickness = int(custom_config.base_thickness +
                              t * (custom_config.max_thickness - custom_config.base_thickness))

                overlay = output.copy()
                cv2.line(overlay, pt1, pt2, color, thickness, cv2.LINE_AA)
                cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)

        return output


def demo_trajectory_visualizer():
    """
    Demonstration of the TrajectoryVisualizer capabilities.
    Creates a synthetic animation showing the motion capture effect.
    """
    import math

    # Create a black frame
    width, height = 800, 600

    # Initialize visualizer with motion capture settings
    config = TrailConfig(
        length=50,
        min_alpha=0.2,
        max_alpha=1.0,
        base_thickness=2,
        max_thickness=8,
        color_scheme=ColorScheme.MOTION_CAPTURE,
        enable_glow=True,
        glow_radius=20,
        glow_intensity=0.7,
    )

    visualizer = TrajectoryVisualizer(trail_config=config)

    # Generate circular motion path
    n_frames = 200
    center_x, center_y = width // 2, height // 2
    radius = 150

    print("Generating motion capture demo...")

    for i in range(n_frames):
        # Create dark frame
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:] = (20, 20, 30)  # Dark blue-gray background

        # Calculate positions
        angle = 2 * math.pi * i / n_frames

        keypoints = {
            "main": (
                center_x + radius * math.cos(angle),
                center_y + radius * math.sin(angle)
            ),
            "secondary": (
                center_x + radius * 0.7 * math.cos(angle * 1.5 + math.pi),
                center_y + radius * 0.7 * math.sin(angle * 1.5 + math.pi)
            ),
        }

        # Update and render
        result = visualizer.update_frame(frame, keypoints)

        # Display
        cv2.imshow("Motion Capture Demo", result)
        key = cv2.waitKey(30)
        if key == 27:  # ESC
            break

    cv2.destroyAllWindows()
    print("Demo complete!")


if __name__ == "__main__":
    demo_trajectory_visualizer()
