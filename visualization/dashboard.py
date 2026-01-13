"""
Real-time Dashboard
===================

Creates overlay dashboards for real-time visualization of locomotion metrics
during video playback.

Author: Stride Labs
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict, Any, Union, Callable
from dataclasses import dataclass, field
from enum import Enum


class MetricType(Enum):
    """Types of metrics that can be displayed."""
    SPEED = "speed"
    CADENCE = "cadence"
    PHASE = "phase"
    STRIDE_LENGTH = "stride_length"
    COORDINATION = "coordination"
    CUSTOM = "custom"


@dataclass
class MetricConfig:
    """Configuration for a single metric display."""
    name: str
    unit: str = ""
    min_value: float = 0.0
    max_value: float = 100.0
    warning_threshold: Optional[float] = None
    critical_threshold: Optional[float] = None
    format_string: str = "{:.1f}"
    color: Tuple[int, int, int] = (0, 255, 255)  # Cyan (BGR)
    history_length: int = 50  # For sparkline


@dataclass
class DashboardLayout:
    """Layout configuration for the dashboard."""
    # Position relative to video frame
    position: str = "bottom"  # "top", "bottom", "left", "right", "overlay"
    width: int = 400  # Width for side panels
    height: int = 120  # Height for top/bottom panels

    # Colors
    background_color: Tuple[int, int, int] = (20, 20, 30)
    background_alpha: float = 0.85
    border_color: Tuple[int, int, int] = (60, 60, 80)
    border_width: int = 2

    # Text
    font_scale: float = 0.6
    font_thickness: int = 1
    font_color: Tuple[int, int, int] = (240, 240, 240)
    title_color: Tuple[int, int, int] = (100, 200, 255)

    # Spacing
    padding: int = 15
    metric_spacing: int = 10


@dataclass
class DashboardMetrics:
    """Container for current metric values."""
    speed: float = 0.0
    speed_unit: str = "cm/s"
    cadence: float = 0.0
    cadence_unit: str = "Hz"
    stride_length: float = 0.0
    stride_length_unit: str = "mm"
    coordination_phase: float = 0.0  # 0 to 2*pi
    coordination_r: float = 0.0  # 0 to 1

    # Limb phases (True = swing, False = stance)
    front_left_swing: bool = False
    front_right_swing: bool = False
    hind_left_swing: bool = False
    hind_right_swing: bool = False

    # Custom metrics
    custom_metrics: Dict[str, float] = field(default_factory=dict)


class RealtimeDashboard:
    """
    Creates real-time overlay dashboards for video visualization.

    Displays locomotion metrics with professional styling including
    gauges, sparklines, and phase indicators.

    Attributes:
        layout: Dashboard layout configuration
        metrics_config: Configuration for each metric
        history: History of metric values for sparklines
    """

    def __init__(
        self,
        layout: Optional[DashboardLayout] = None,
        metrics_config: Optional[Dict[str, MetricConfig]] = None
    ):
        """
        Initialize the real-time dashboard.

        Args:
            layout: Dashboard layout configuration
            metrics_config: Configuration for displayed metrics
        """
        self.layout = layout or DashboardLayout()
        self.metrics_config = metrics_config or self._default_metrics_config()
        self.history: Dict[str, List[float]] = {name: [] for name in self.metrics_config}

    def _default_metrics_config(self) -> Dict[str, MetricConfig]:
        """Create default metric configurations."""
        return {
            "speed": MetricConfig(
                name="Speed",
                unit="cm/s",
                min_value=0,
                max_value=30,
                warning_threshold=25,
                color=(255, 200, 100)  # Light blue
            ),
            "cadence": MetricConfig(
                name="Cadence",
                unit="Hz",
                min_value=0,
                max_value=10,
                color=(100, 255, 200)  # Light green
            ),
            "stride": MetricConfig(
                name="Stride",
                unit="mm",
                min_value=0,
                max_value=50,
                color=(200, 100, 255)  # Pink
            ),
            "coordination": MetricConfig(
                name="Coord R",
                unit="",
                min_value=0,
                max_value=1,
                format_string="{:.2f}",
                color=(100, 200, 255)  # Orange
            ),
        }

    def _draw_rounded_rect(
        self,
        img: np.ndarray,
        pt1: Tuple[int, int],
        pt2: Tuple[int, int],
        color: Tuple[int, int, int],
        radius: int = 10,
        thickness: int = -1
    ) -> np.ndarray:
        """
        Draw a rounded rectangle.

        Args:
            img: Image to draw on
            pt1: Top-left corner
            pt2: Bottom-right corner
            color: BGR color
            radius: Corner radius
            thickness: Line thickness (-1 for filled)

        Returns:
            Image with rounded rectangle
        """
        x1, y1 = pt1
        x2, y2 = pt2

        # Draw the main rectangle body
        if thickness == -1:
            # Filled rectangle
            cv2.rectangle(img, (x1 + radius, y1), (x2 - radius, y2), color, -1)
            cv2.rectangle(img, (x1, y1 + radius), (x2, y2 - radius), color, -1)

            # Draw corners
            cv2.ellipse(img, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90, color, -1)
            cv2.ellipse(img, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90, color, -1)
            cv2.ellipse(img, (x1 + radius, y2 - radius), (radius, radius), 90, 0, 90, color, -1)
            cv2.ellipse(img, (x2 - radius, y2 - radius), (radius, radius), 0, 0, 90, color, -1)
        else:
            # Outline only
            cv2.line(img, (x1 + radius, y1), (x2 - radius, y1), color, thickness)
            cv2.line(img, (x1 + radius, y2), (x2 - radius, y2), color, thickness)
            cv2.line(img, (x1, y1 + radius), (x1, y2 - radius), color, thickness)
            cv2.line(img, (x2, y1 + radius), (x2, y2 - radius), color, thickness)

            cv2.ellipse(img, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90, color, thickness)
            cv2.ellipse(img, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90, color, thickness)
            cv2.ellipse(img, (x1 + radius, y2 - radius), (radius, radius), 90, 0, 90, color, thickness)
            cv2.ellipse(img, (x2 - radius, y2 - radius), (radius, radius), 0, 0, 90, color, thickness)

        return img

    def _draw_gauge(
        self,
        img: np.ndarray,
        center: Tuple[int, int],
        radius: int,
        value: float,
        min_val: float,
        max_val: float,
        color: Tuple[int, int, int],
        label: str = "",
        show_value: bool = True,
        format_str: str = "{:.1f}"
    ) -> np.ndarray:
        """
        Draw a semicircular gauge.

        Args:
            img: Image to draw on
            center: Center point
            radius: Gauge radius
            value: Current value
            min_val: Minimum value
            max_val: Maximum value
            color: Gauge color
            label: Label text
            show_value: Whether to show numeric value
            format_str: Format string for value

        Returns:
            Image with gauge
        """
        cx, cy = center

        # Background arc
        cv2.ellipse(img, center, (radius, radius), 0, 180, 360,
                   self.layout.border_color, 3, cv2.LINE_AA)

        # Value arc
        normalized = np.clip((value - min_val) / (max_val - min_val), 0, 1)
        end_angle = 180 + normalized * 180

        cv2.ellipse(img, center, (radius, radius), 0, 180, int(end_angle),
                   color, 4, cv2.LINE_AA)

        # Draw tick marks
        for tick_val in [0.25, 0.5, 0.75]:
            tick_angle = np.radians(180 + tick_val * 180)
            inner_r = radius - 5
            outer_r = radius + 5
            x1 = int(cx + inner_r * np.cos(tick_angle))
            y1 = int(cy + inner_r * np.sin(tick_angle))
            x2 = int(cx + outer_r * np.cos(tick_angle))
            y2 = int(cy + outer_r * np.sin(tick_angle))
            cv2.line(img, (x1, y1), (x2, y2), self.layout.font_color, 1, cv2.LINE_AA)

        # Needle
        needle_angle = np.radians(180 + normalized * 180)
        needle_x = int(cx + (radius - 10) * np.cos(needle_angle))
        needle_y = int(cy + (radius - 10) * np.sin(needle_angle))
        cv2.line(img, center, (needle_x, needle_y), (255, 255, 255), 2, cv2.LINE_AA)
        cv2.circle(img, center, 4, (255, 255, 255), -1, cv2.LINE_AA)

        # Value text
        if show_value:
            text = format_str.format(value)
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            text_x = cx - text_size[0] // 2
            text_y = cy + radius // 2
            cv2.putText(img, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                       0.5, color, 1, cv2.LINE_AA)

        # Label
        if label:
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
            label_x = cx - label_size[0] // 2
            label_y = cy + radius // 2 + 18
            cv2.putText(img, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX,
                       0.4, self.layout.font_color, 1, cv2.LINE_AA)

        return img

    def _draw_sparkline(
        self,
        img: np.ndarray,
        rect: Tuple[int, int, int, int],
        values: List[float],
        color: Tuple[int, int, int],
        fill: bool = True
    ) -> np.ndarray:
        """
        Draw a sparkline graph.

        Args:
            img: Image to draw on
            rect: (x, y, width, height) rectangle for the sparkline
            values: List of values to plot
            color: Line color
            fill: Whether to fill under the line

        Returns:
            Image with sparkline
        """
        x, y, w, h = rect

        if len(values) < 2:
            return img

        # Normalize values
        min_val = min(values)
        max_val = max(values)
        if max_val == min_val:
            max_val = min_val + 1

        # Generate points
        points = []
        for i, val in enumerate(values):
            px = x + int(i * w / (len(values) - 1))
            py = y + h - int((val - min_val) / (max_val - min_val) * h)
            points.append((px, py))

        points = np.array(points, dtype=np.int32)

        # Fill area
        if fill:
            fill_points = np.vstack([
                points,
                [[x + w, y + h], [x, y + h]]
            ])
            overlay = img.copy()
            cv2.fillPoly(overlay, [fill_points], color)
            cv2.addWeighted(overlay, 0.2, img, 0.8, 0, img)

        # Draw line
        cv2.polylines(img, [points], False, color, 2, cv2.LINE_AA)

        # Draw current value marker
        if len(points) > 0:
            cv2.circle(img, tuple(points[-1]), 4, color, -1, cv2.LINE_AA)

        return img

    def _draw_phase_indicator(
        self,
        img: np.ndarray,
        position: Tuple[int, int],
        limb_name: str,
        is_swing: bool,
        size: int = 15
    ) -> np.ndarray:
        """
        Draw a limb phase indicator (swing/stance).

        Args:
            img: Image to draw on
            position: Center position
            limb_name: Short name for the limb
            is_swing: True if swing phase, False if stance
            size: Indicator size

        Returns:
            Image with indicator
        """
        x, y = position

        # Color based on phase
        if is_swing:
            color = (0, 0, 255)  # Red for swing
            label = "S"
        else:
            color = (128, 128, 128)  # Gray for stance
            label = "T"

        # Draw circle
        cv2.circle(img, (x, y), size, color, -1, cv2.LINE_AA)
        cv2.circle(img, (x, y), size, (255, 255, 255), 1, cv2.LINE_AA)

        # Draw label
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
        text_x = x - text_size[0] // 2
        text_y = y + text_size[1] // 2
        cv2.putText(img, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                   0.4, (255, 255, 255), 1, cv2.LINE_AA)

        # Draw limb name below
        name_size = cv2.getTextSize(limb_name, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)[0]
        name_x = x - name_size[0] // 2
        name_y = y + size + 12
        cv2.putText(img, limb_name, (name_x, name_y), cv2.FONT_HERSHEY_SIMPLEX,
                   0.35, self.layout.font_color, 1, cv2.LINE_AA)

        return img

    def _draw_coordination_indicator(
        self,
        img: np.ndarray,
        center: Tuple[int, int],
        radius: int,
        phase: float,
        r_value: float
    ) -> np.ndarray:
        """
        Draw a circular coordination indicator.

        Args:
            img: Image to draw on
            center: Center position
            radius: Circle radius
            phase: Phase angle in radians
            r_value: Rayleigh R-value (0-1)

        Returns:
            Image with indicator
        """
        cx, cy = center

        # Draw background circle
        cv2.circle(img, center, radius, self.layout.border_color, 2, cv2.LINE_AA)

        # Draw crosshairs
        cv2.line(img, (cx - radius, cy), (cx + radius, cy),
                self.layout.border_color, 1, cv2.LINE_AA)
        cv2.line(img, (cx, cy - radius), (cx, cy + radius),
                self.layout.border_color, 1, cv2.LINE_AA)

        # Draw vector
        # Note: phase=0 should point up, phase=pi should point down
        angle = -phase + np.pi / 2  # Adjust for screen coordinates
        arrow_length = int(r_value * radius * 0.9)
        arrow_x = int(cx + arrow_length * np.cos(angle))
        arrow_y = int(cy - arrow_length * np.sin(angle))

        cv2.arrowedLine(img, center, (arrow_x, arrow_y),
                       (0, 100, 255), 2, cv2.LINE_AA, tipLength=0.3)

        # R-value text
        r_text = f"R={r_value:.2f}"
        text_size = cv2.getTextSize(r_text, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)[0]
        text_x = cx - text_size[0] // 2
        text_y = cy + radius + 15
        cv2.putText(img, r_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                   0.35, (0, 100, 255), 1, cv2.LINE_AA)

        return img

    def update_metrics(self, metrics: DashboardMetrics) -> None:
        """
        Update metric values and history.

        Args:
            metrics: Current metric values
        """
        # Update history
        if "speed" in self.metrics_config:
            self.history["speed"].append(metrics.speed)
            if len(self.history["speed"]) > self.metrics_config["speed"].history_length:
                self.history["speed"] = self.history["speed"][-self.metrics_config["speed"].history_length:]

        if "cadence" in self.metrics_config:
            self.history["cadence"].append(metrics.cadence)
            if len(self.history["cadence"]) > self.metrics_config["cadence"].history_length:
                self.history["cadence"] = self.history["cadence"][-self.metrics_config["cadence"].history_length:]

        if "stride" in self.metrics_config:
            self.history["stride"].append(metrics.stride_length)
            if len(self.history["stride"]) > self.metrics_config["stride"].history_length:
                self.history["stride"] = self.history["stride"][-self.metrics_config["stride"].history_length:]

        if "coordination" in self.metrics_config:
            self.history["coordination"].append(metrics.coordination_r)
            if len(self.history["coordination"]) > self.metrics_config["coordination"].history_length:
                self.history["coordination"] = self.history["coordination"][-self.metrics_config["coordination"].history_length:]

    def create_dashboard(
        self,
        frame_size: Tuple[int, int]
    ) -> np.ndarray:
        """
        Create an empty dashboard panel.

        Args:
            frame_size: (width, height) of the video frame

        Returns:
            Dashboard panel image
        """
        width, height = frame_size

        if self.layout.position in ["top", "bottom"]:
            panel_w = width
            panel_h = self.layout.height
        else:
            panel_w = self.layout.width
            panel_h = height

        # Create panel with transparency
        panel = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
        panel[:] = self.layout.background_color

        # Draw border
        self._draw_rounded_rect(
            panel,
            (0, 0),
            (panel_w - 1, panel_h - 1),
            self.layout.border_color,
            radius=5,
            thickness=self.layout.border_width
        )

        return panel

    def render_overlay(
        self,
        frame: np.ndarray,
        metrics: DashboardMetrics
    ) -> np.ndarray:
        """
        Render the complete dashboard overlay on a frame.

        Args:
            frame: Input video frame
            metrics: Current metric values

        Returns:
            Frame with dashboard overlay
        """
        self.update_metrics(metrics)

        height, width = frame.shape[:2]
        output = frame.copy()

        # Create dashboard panel
        panel = self.create_dashboard((width, height))
        panel_h, panel_w = panel.shape[:2]

        # Layout metrics
        pad = self.layout.padding
        section_width = (panel_w - 3 * pad) // 4

        # Section 1: Speed gauge and sparkline
        if "speed" in self.metrics_config:
            config = self.metrics_config["speed"]

            # Gauge
            gauge_center = (pad + section_width // 2, panel_h // 2 - 10)
            panel = self._draw_gauge(
                panel, gauge_center, 35,
                metrics.speed, config.min_value, config.max_value,
                config.color, f"{config.name} ({config.unit})"
            )

            # Sparkline below
            if len(self.history["speed"]) > 1:
                sparkline_rect = (pad, panel_h - 25, section_width, 15)
                panel = self._draw_sparkline(panel, sparkline_rect,
                                            self.history["speed"], config.color)

        # Section 2: Cadence
        if "cadence" in self.metrics_config:
            config = self.metrics_config["cadence"]
            x_offset = pad + section_width + pad // 2

            gauge_center = (x_offset + section_width // 2, panel_h // 2 - 10)
            panel = self._draw_gauge(
                panel, gauge_center, 35,
                metrics.cadence, config.min_value, config.max_value,
                config.color, f"{config.name} ({config.unit})"
            )

        # Section 3: Limb phase indicators
        x_offset = pad + 2 * (section_width + pad // 2)
        phase_y = panel_h // 2

        # Draw title
        cv2.putText(panel, "Gait Phase", (x_offset + 10, 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.layout.title_color, 1, cv2.LINE_AA)

        # 4 limb indicators in a 2x2 grid
        indicators = [
            ("FL", metrics.front_left_swing, x_offset + 25, phase_y - 15),
            ("FR", metrics.front_right_swing, x_offset + 70, phase_y - 15),
            ("HL", metrics.hind_left_swing, x_offset + 25, phase_y + 25),
            ("HR", metrics.hind_right_swing, x_offset + 70, phase_y + 25),
        ]

        for name, is_swing, px, py in indicators:
            panel = self._draw_phase_indicator(panel, (px, py), name, is_swing, 12)

        # Section 4: Coordination indicator
        x_offset = pad + 3 * (section_width + pad // 2)

        cv2.putText(panel, "Coordination", (x_offset + 5, 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.layout.title_color, 1, cv2.LINE_AA)

        coord_center = (x_offset + section_width // 2, panel_h // 2 + 10)
        panel = self._draw_coordination_indicator(
            panel, coord_center, 30,
            metrics.coordination_phase, metrics.coordination_r
        )

        # Blend panel with frame
        if self.layout.position == "bottom":
            y_start = height - panel_h
            y_end = height
        elif self.layout.position == "top":
            y_start = 0
            y_end = panel_h
        else:
            y_start = (height - panel_h) // 2
            y_end = y_start + panel_h

        # Apply with alpha blending
        roi = output[y_start:y_end, 0:panel_w]
        blended = cv2.addWeighted(panel, self.layout.background_alpha,
                                 roi, 1 - self.layout.background_alpha, 0)
        output[y_start:y_end, 0:panel_w] = blended

        return output

    def render_minimal_overlay(
        self,
        frame: np.ndarray,
        metrics: DashboardMetrics,
        position: str = "top_right"
    ) -> np.ndarray:
        """
        Render a minimal metrics overlay.

        Args:
            frame: Input frame
            metrics: Current metrics
            position: Corner position ("top_left", "top_right", "bottom_left", "bottom_right")

        Returns:
            Frame with minimal overlay
        """
        output = frame.copy()
        height, width = frame.shape[:2]

        # Create text lines
        lines = [
            f"Speed: {metrics.speed:.1f} {metrics.speed_unit}",
            f"Cadence: {metrics.cadence:.1f} {metrics.cadence_unit}",
            f"Stride: {metrics.stride_length:.1f} {metrics.stride_length_unit}",
            f"R-value: {metrics.coordination_r:.2f}",
        ]

        # Calculate text box size
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        line_height = 20
        max_width = 0

        for line in lines:
            text_size = cv2.getTextSize(line, font, font_scale, thickness)[0]
            max_width = max(max_width, text_size[0])

        box_w = max_width + 20
        box_h = len(lines) * line_height + 15

        # Determine position
        margin = 10
        if "right" in position:
            x = width - box_w - margin
        else:
            x = margin

        if "bottom" in position:
            y = height - box_h - margin
        else:
            y = margin

        # Draw background
        overlay = output.copy()
        cv2.rectangle(overlay, (x, y), (x + box_w, y + box_h),
                     self.layout.background_color, -1)
        cv2.addWeighted(overlay, self.layout.background_alpha,
                       output, 1 - self.layout.background_alpha, 0, output)

        # Draw border
        cv2.rectangle(output, (x, y), (x + box_w, y + box_h),
                     self.layout.border_color, 1)

        # Draw text
        for i, line in enumerate(lines):
            text_y = y + 15 + i * line_height
            cv2.putText(output, line, (x + 10, text_y), font, font_scale,
                       self.layout.font_color, thickness, cv2.LINE_AA)

        return output


def demo_dashboard():
    """
    Demonstrate the real-time dashboard with synthetic data.
    """
    import math
    import time

    dashboard = RealtimeDashboard()

    # Create a synthetic video frame
    width, height = 800, 600
    n_frames = 200

    print("Generating dashboard demo...")

    for i in range(n_frames):
        # Create base frame
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:] = (40, 40, 50)

        # Add some visual elements
        cv2.putText(frame, "Video Frame", (width // 2 - 60, height // 2),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 100, 100), 2)

        # Simulate metrics
        t = i / n_frames * 4 * math.pi
        metrics = DashboardMetrics(
            speed=15 + 5 * math.sin(t) + np.random.normal(0, 0.5),
            cadence=4 + 1 * math.sin(t * 1.5) + np.random.normal(0, 0.1),
            stride_length=20 + 5 * math.sin(t * 0.8),
            coordination_phase=t % (2 * math.pi),
            coordination_r=0.7 + 0.2 * math.sin(t * 0.5),
            front_left_swing=math.sin(t) > 0.3,
            front_right_swing=math.sin(t + math.pi) > 0.3,
            hind_left_swing=math.sin(t + math.pi) > 0.3,
            hind_right_swing=math.sin(t) > 0.3,
        )

        # Render dashboard
        result = dashboard.render_overlay(frame, metrics)

        # Display
        cv2.imshow("Dashboard Demo", result)
        key = cv2.waitKey(30)
        if key == 27:
            break

    cv2.destroyAllWindows()
    print("Demo complete!")


if __name__ == "__main__":
    demo_dashboard()
