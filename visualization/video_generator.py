"""
Video Generator
===============

Combines video, trajectory overlays, and dashboard into final output videos
with professional H.264 encoding.

Author: Stride Labs
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any, Union, Callable, Generator
from dataclasses import dataclass, field
from enum import Enum
import time

from .trajectory_overlay import TrajectoryVisualizer, TrailConfig, MarkerConfig, ColorScheme
from .stick_figures import StickFigureGenerator, SkeletonConfig, LimbPhase
from .dashboard import RealtimeDashboard, DashboardLayout, DashboardMetrics


class VideoCodec(Enum):
    """Supported video codecs."""
    H264 = "avc1"  # H.264 codec (most compatible)
    H265 = "hvc1"  # H.265/HEVC (better compression)
    VP9 = "vp90"   # VP9 (web optimized)
    MJPEG = "mjpg"  # Motion JPEG (fast, large files)
    MP4V = "mp4v"  # MPEG-4 (fallback)


@dataclass
class VideoConfig:
    """Configuration for video output."""
    # Output settings
    codec: VideoCodec = VideoCodec.H264
    fps: float = 30.0
    resolution: Optional[Tuple[int, int]] = None  # None = same as input
    quality: int = 23  # CRF value (lower = better quality, 18-28 typical)

    # Overlay options
    enable_trajectory: bool = True
    enable_stick_figures: bool = False
    enable_dashboard: bool = True

    # Processing
    start_frame: int = 0
    end_frame: int = -1  # -1 for all frames
    skip_frames: int = 1  # Process every Nth frame

    # Progress callback
    progress_callback: Optional[Callable[[int, int], None]] = None


@dataclass
class OverlayConfig:
    """Configuration for overlay elements."""
    # Trajectory settings
    trajectory_config: Optional[TrailConfig] = None

    # Stick figure settings
    skeleton_config: Optional[SkeletonConfig] = None

    # Dashboard settings
    dashboard_layout: Optional[DashboardLayout] = None

    # Keypoint mapping (map your data keys to standard names)
    keypoint_mapping: Dict[str, str] = field(default_factory=dict)


class VideoGenerator:
    """
    Generates professional output videos with overlays.

    Combines original video footage with trajectory visualization,
    stick figures, and real-time metric dashboards.

    Attributes:
        video_config: Video output configuration
        overlay_config: Overlay element configuration
        trajectory_viz: Trajectory visualizer instance
        stick_figure_gen: Stick figure generator instance
        dashboard: Dashboard instance
    """

    def __init__(
        self,
        video_config: Optional[VideoConfig] = None,
        overlay_config: Optional[OverlayConfig] = None
    ):
        """
        Initialize the video generator.

        Args:
            video_config: Video output configuration
            overlay_config: Overlay element configuration
        """
        self.video_config = video_config or VideoConfig()
        self.overlay_config = overlay_config or OverlayConfig()

        # Initialize overlay components
        self._init_overlays()

    def _init_overlays(self) -> None:
        """Initialize overlay visualization components."""
        # Trajectory visualizer
        trail_config = self.overlay_config.trajectory_config or TrailConfig(
            length=40,
            min_alpha=0.2,
            max_alpha=1.0,
            base_thickness=2,
            max_thickness=6,
            color_scheme=ColorScheme.MOTION_CAPTURE,
            enable_glow=True,
            glow_radius=15,
            glow_intensity=0.5
        )
        self.trajectory_viz = TrajectoryVisualizer(trail_config=trail_config)

        # Stick figure generator
        skeleton_config = self.overlay_config.skeleton_config or SkeletonConfig()
        self.stick_figure_gen = StickFigureGenerator(config=skeleton_config)

        # Dashboard
        dashboard_layout = self.overlay_config.dashboard_layout or DashboardLayout(
            position="bottom",
            height=100,
            background_alpha=0.8
        )
        self.dashboard = RealtimeDashboard(layout=dashboard_layout)

    def _get_fourcc(self, codec: VideoCodec) -> int:
        """
        Get OpenCV FourCC code for codec.

        Args:
            codec: VideoCodec enum value

        Returns:
            FourCC integer code
        """
        codec_map = {
            VideoCodec.H264: cv2.VideoWriter_fourcc(*'avc1'),
            VideoCodec.H265: cv2.VideoWriter_fourcc(*'hvc1'),
            VideoCodec.VP9: cv2.VideoWriter_fourcc(*'vp90'),
            VideoCodec.MJPEG: cv2.VideoWriter_fourcc(*'MJPG'),
            VideoCodec.MP4V: cv2.VideoWriter_fourcc(*'mp4v'),
        }

        # Try requested codec, fall back to mp4v if not available
        fourcc = codec_map.get(codec, cv2.VideoWriter_fourcc(*'mp4v'))

        return fourcc

    def _map_keypoints(
        self,
        keypoints: Dict[str, Any]
    ) -> Dict[str, Tuple[float, float]]:
        """
        Map keypoint names using the configured mapping.

        Args:
            keypoints: Input keypoints dictionary

        Returns:
            Mapped keypoints dictionary
        """
        if not self.overlay_config.keypoint_mapping:
            # Return as-is if no mapping configured
            return {k: tuple(v[:2]) if hasattr(v, '__len__') else v
                   for k, v in keypoints.items()}

        mapped = {}
        for src_name, dst_name in self.overlay_config.keypoint_mapping.items():
            if src_name in keypoints:
                val = keypoints[src_name]
                if hasattr(val, '__len__'):
                    mapped[dst_name] = (float(val[0]), float(val[1]))
                else:
                    mapped[dst_name] = val

        return mapped

    def add_overlays(
        self,
        frame: np.ndarray,
        keypoints: Optional[Dict[str, Tuple[float, float]]] = None,
        limb_phases: Optional[Dict[str, LimbPhase]] = None,
        metrics: Optional[DashboardMetrics] = None
    ) -> np.ndarray:
        """
        Add all configured overlays to a frame.

        Args:
            frame: Input video frame
            keypoints: Keypoint positions
            limb_phases: Gait phase for each limb
            metrics: Dashboard metrics

        Returns:
            Frame with overlays applied
        """
        output = frame.copy()

        # Map keypoints if needed
        if keypoints:
            keypoints = self._map_keypoints(keypoints)

        # Add trajectory overlay
        if self.video_config.enable_trajectory and keypoints:
            output = self.trajectory_viz.update_frame(
                output, keypoints,
                draw_markers=True,
                draw_trails=True
            )

        # Add stick figures
        if self.video_config.enable_stick_figures and keypoints:
            output = self.stick_figure_gen.generate_frame(
                output, keypoints, limb_phases
            )

        # Add dashboard
        if self.video_config.enable_dashboard and metrics:
            output = self.dashboard.render_overlay(output, metrics)

        return output

    def process_video(
        self,
        input_path: Union[str, Path],
        output_path: Union[str, Path],
        keypoints_data: Optional[List[Dict[str, Tuple[float, float]]]] = None,
        limb_phases_data: Optional[List[Dict[str, LimbPhase]]] = None,
        metrics_data: Optional[List[DashboardMetrics]] = None
    ) -> bool:
        """
        Process a video file with overlays.

        Args:
            input_path: Path to input video
            output_path: Path for output video
            keypoints_data: List of keypoints for each frame
            limb_phases_data: List of limb phases for each frame
            metrics_data: List of metrics for each frame

        Returns:
            True if successful, False otherwise
        """
        input_path = Path(input_path)
        output_path = Path(output_path)

        # Open input video
        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            print(f"Error: Could not open video {input_path}")
            return False

        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        input_fps = cap.get(cv2.CAP_PROP_FPS)
        input_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        input_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Determine output properties
        output_fps = self.video_config.fps or input_fps
        if self.video_config.resolution:
            output_width, output_height = self.video_config.resolution
        else:
            output_width, output_height = input_width, input_height

        # Calculate frame range
        start_frame = max(0, self.video_config.start_frame)
        end_frame = self.video_config.end_frame
        if end_frame < 0:
            end_frame = total_frames

        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize video writer
        fourcc = self._get_fourcc(self.video_config.codec)
        writer = cv2.VideoWriter(
            str(output_path),
            fourcc,
            output_fps,
            (output_width, output_height)
        )

        if not writer.isOpened():
            print(f"Error: Could not create output video {output_path}")
            print("Trying fallback codec...")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(
                str(output_path),
                fourcc,
                output_fps,
                (output_width, output_height)
            )
            if not writer.isOpened():
                cap.release()
                return False

        # Process frames
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        frame_idx = start_frame
        processed_count = 0

        print(f"Processing video: {input_path.name}")
        print(f"Output: {output_path.name}")
        print(f"Frames: {start_frame} to {end_frame} (skip={self.video_config.skip_frames})")

        start_time = time.time()

        while frame_idx < end_frame:
            ret, frame = cap.read()
            if not ret:
                break

            # Skip frames if configured
            if (frame_idx - start_frame) % self.video_config.skip_frames != 0:
                frame_idx += 1
                continue

            # Resize if needed
            if (output_width, output_height) != (input_width, input_height):
                frame = cv2.resize(frame, (output_width, output_height))

            # Get data for this frame
            data_idx = (frame_idx - start_frame) // self.video_config.skip_frames

            keypoints = None
            if keypoints_data and data_idx < len(keypoints_data):
                keypoints = keypoints_data[data_idx]

            limb_phases = None
            if limb_phases_data and data_idx < len(limb_phases_data):
                limb_phases = limb_phases_data[data_idx]

            metrics = None
            if metrics_data and data_idx < len(metrics_data):
                metrics = metrics_data[data_idx]

            # Add overlays
            output_frame = self.add_overlays(frame, keypoints, limb_phases, metrics)

            # Write frame
            writer.write(output_frame)
            processed_count += 1

            # Progress callback
            if self.video_config.progress_callback:
                self.video_config.progress_callback(frame_idx - start_frame, end_frame - start_frame)

            # Progress output
            if processed_count % 100 == 0:
                elapsed = time.time() - start_time
                fps_actual = processed_count / elapsed if elapsed > 0 else 0
                remaining = (end_frame - frame_idx) / fps_actual if fps_actual > 0 else 0
                print(f"  Frame {frame_idx}/{end_frame} ({processed_count} processed, "
                      f"{fps_actual:.1f} fps, ~{remaining:.0f}s remaining)")

            frame_idx += 1

        # Cleanup
        cap.release()
        writer.release()

        elapsed = time.time() - start_time
        print(f"Completed! Processed {processed_count} frames in {elapsed:.1f}s")
        print(f"Output saved to: {output_path}")

        return True

    def export(
        self,
        frames: List[np.ndarray],
        output_path: Union[str, Path],
        keypoints_data: Optional[List[Dict[str, Tuple[float, float]]]] = None,
        limb_phases_data: Optional[List[Dict[str, LimbPhase]]] = None,
        metrics_data: Optional[List[DashboardMetrics]] = None
    ) -> bool:
        """
        Export a list of frames as a video.

        Args:
            frames: List of frame images
            output_path: Output video path
            keypoints_data: Keypoints for each frame
            limb_phases_data: Limb phases for each frame
            metrics_data: Metrics for each frame

        Returns:
            True if successful
        """
        if not frames:
            print("Error: No frames to export")
            return False

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Get dimensions from first frame
        height, width = frames[0].shape[:2]

        if self.video_config.resolution:
            out_width, out_height = self.video_config.resolution
        else:
            out_width, out_height = width, height

        # Initialize writer
        fourcc = self._get_fourcc(self.video_config.codec)
        writer = cv2.VideoWriter(
            str(output_path),
            fourcc,
            self.video_config.fps,
            (out_width, out_height)
        )

        if not writer.isOpened():
            print(f"Error: Could not create output video")
            return False

        print(f"Exporting {len(frames)} frames to {output_path.name}")

        for i, frame in enumerate(frames):
            # Resize if needed
            if (out_width, out_height) != (width, height):
                frame = cv2.resize(frame, (out_width, out_height))

            # Get overlay data
            keypoints = keypoints_data[i] if keypoints_data and i < len(keypoints_data) else None
            limb_phases = limb_phases_data[i] if limb_phases_data and i < len(limb_phases_data) else None
            metrics = metrics_data[i] if metrics_data and i < len(metrics_data) else None

            # Add overlays
            output_frame = self.add_overlays(frame, keypoints, limb_phases, metrics)

            writer.write(output_frame)

            if self.video_config.progress_callback:
                self.video_config.progress_callback(i, len(frames))

        writer.release()
        print(f"Export complete: {output_path}")

        return True

    def generate_preview(
        self,
        input_path: Union[str, Path],
        keypoints_data: Optional[List[Dict[str, Tuple[float, float]]]] = None,
        metrics_data: Optional[List[DashboardMetrics]] = None,
        preview_frames: int = 100,
        start_frame: int = 0
    ) -> Generator[np.ndarray, None, None]:
        """
        Generate preview frames as a generator.

        Args:
            input_path: Input video path
            keypoints_data: Keypoints for each frame
            metrics_data: Metrics for each frame
            preview_frames: Number of frames to preview
            start_frame: Starting frame

        Yields:
            Processed frames
        """
        input_path = Path(input_path)

        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            print(f"Error: Could not open video {input_path}")
            return

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        end_frame = min(start_frame + preview_frames, total_frames)

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        for frame_idx in range(start_frame, end_frame):
            ret, frame = cap.read()
            if not ret:
                break

            data_idx = frame_idx - start_frame

            keypoints = None
            if keypoints_data and data_idx < len(keypoints_data):
                keypoints = keypoints_data[data_idx]

            metrics = None
            if metrics_data and data_idx < len(metrics_data):
                metrics = metrics_data[data_idx]

            output_frame = self.add_overlays(frame, keypoints, metrics=metrics)

            yield output_frame

        cap.release()

    def create_comparison_video(
        self,
        video_paths: List[Union[str, Path]],
        output_path: Union[str, Path],
        labels: Optional[List[str]] = None,
        layout: str = "horizontal"  # "horizontal", "vertical", "grid"
    ) -> bool:
        """
        Create a side-by-side comparison video.

        Args:
            video_paths: List of input video paths
            output_path: Output video path
            labels: Optional labels for each video
            layout: Layout arrangement

        Returns:
            True if successful
        """
        if not video_paths:
            return False

        # Open all videos
        caps = []
        for path in video_paths:
            cap = cv2.VideoCapture(str(path))
            if not cap.isOpened():
                print(f"Error: Could not open {path}")
                for c in caps:
                    c.release()
                return False
            caps.append(cap)

        # Get properties from first video
        width = int(caps[0].get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(caps[0].get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = caps[0].get(cv2.CAP_PROP_FPS)
        total_frames = min(int(c.get(cv2.CAP_PROP_FRAME_COUNT)) for c in caps)

        n_videos = len(caps)
        labels = labels or [f"Video {i+1}" for i in range(n_videos)]

        # Calculate output dimensions
        if layout == "horizontal":
            out_width = width * n_videos
            out_height = height
            grid_cols = n_videos
            grid_rows = 1
        elif layout == "vertical":
            out_width = width
            out_height = height * n_videos
            grid_cols = 1
            grid_rows = n_videos
        else:  # grid
            grid_cols = int(np.ceil(np.sqrt(n_videos)))
            grid_rows = int(np.ceil(n_videos / grid_cols))
            out_width = width * grid_cols
            out_height = height * grid_rows

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        fourcc = self._get_fourcc(self.video_config.codec)
        writer = cv2.VideoWriter(
            str(output_path),
            fourcc,
            fps,
            (out_width, out_height)
        )

        print(f"Creating comparison video: {output_path.name}")
        print(f"Layout: {layout} ({grid_cols}x{grid_rows})")

        for frame_idx in range(total_frames):
            # Read frames from all videos
            frames = []
            for cap in caps:
                ret, frame = cap.read()
                if not ret:
                    frames.append(np.zeros((height, width, 3), dtype=np.uint8))
                else:
                    frames.append(frame)

            # Create output frame
            output = np.zeros((out_height, out_width, 3), dtype=np.uint8)

            for i, (frame, label) in enumerate(zip(frames, labels)):
                row = i // grid_cols
                col = i % grid_cols
                y1 = row * height
                y2 = y1 + height
                x1 = col * width
                x2 = x1 + width

                # Add label
                cv2.putText(frame, label, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                output[y1:y2, x1:x2] = frame

            writer.write(output)

            if frame_idx % 100 == 0:
                print(f"  Frame {frame_idx}/{total_frames}")

        # Cleanup
        for cap in caps:
            cap.release()
        writer.release()

        print(f"Comparison video saved: {output_path}")
        return True


def demo_video_generator():
    """
    Demonstrate video generation with synthetic data.
    """
    import math

    # Create synthetic frames and data
    width, height = 640, 480
    n_frames = 150
    fps = 30

    print("Generating synthetic demo video...")

    frames = []
    keypoints_data = []
    metrics_data = []

    for i in range(n_frames):
        # Create frame with gradient background
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        for y in range(height):
            intensity = int(30 + 20 * y / height)
            frame[y, :] = (intensity, intensity, intensity + 10)

        # Add some visual elements
        cv2.putText(frame, f"Frame {i}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 2)

        frames.append(frame)

        # Generate keypoints (circular motion)
        t = i / n_frames * 2 * math.pi
        cx, cy = width // 2, height // 2 - 50
        radius = 100

        keypoints = {
            "nose": (cx + radius * math.cos(t), cy + radius * math.sin(t)),
            "tail": (cx + radius * 0.5 * math.cos(t + math.pi),
                    cy + radius * 0.5 * math.sin(t + math.pi)),
            "front_left": (cx + radius * 0.8 * math.cos(t + 0.5),
                          cy + radius * 0.8 * math.sin(t + 0.5)),
            "hind_right": (cx + radius * 0.8 * math.cos(t + 0.5 + math.pi),
                          cy + radius * 0.8 * math.sin(t + 0.5 + math.pi)),
        }
        keypoints_data.append(keypoints)

        # Generate metrics
        metrics = DashboardMetrics(
            speed=15 + 5 * math.sin(t * 2),
            cadence=4 + math.sin(t * 3),
            stride_length=20 + 5 * math.sin(t),
            coordination_phase=t,
            coordination_r=0.7 + 0.2 * math.sin(t),
            front_left_swing=math.sin(t) > 0,
            front_right_swing=math.sin(t + math.pi) > 0,
            hind_left_swing=math.sin(t + math.pi) > 0,
            hind_right_swing=math.sin(t) > 0,
        )
        metrics_data.append(metrics)

    # Configure and create generator
    video_config = VideoConfig(
        codec=VideoCodec.MP4V,
        fps=fps,
        enable_trajectory=True,
        enable_stick_figures=False,
        enable_dashboard=True,
    )

    generator = VideoGenerator(video_config=video_config)

    # Export video
    output_path = Path("/tmp/demo_output.mp4")
    success = generator.export(
        frames,
        output_path,
        keypoints_data=keypoints_data,
        metrics_data=metrics_data
    )

    if success:
        print(f"\nDemo video saved to: {output_path}")
        print("You can play it with: ffplay /tmp/demo_output.mp4")

    # Also show a preview
    print("\nShowing preview (press ESC to exit)...")
    for i, frame in enumerate(frames[:100]):
        keypoints = keypoints_data[i] if i < len(keypoints_data) else None
        metrics = metrics_data[i] if i < len(metrics_data) else None

        output = generator.add_overlays(frame, keypoints, metrics=metrics)

        cv2.imshow("Video Generator Demo", output)
        key = cv2.waitKey(33)
        if key == 27:
            break

    cv2.destroyAllWindows()
    print("Demo complete!")


if __name__ == "__main__":
    demo_video_generator()
