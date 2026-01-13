"""
Video Metadata Module

Provides utilities for extracting metadata from video files including
frame rate, duration, resolution, and pixel-to-millimeter conversion.

Author: Stride Labs
License: MIT
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# Supported video formats
SUPPORTED_FORMATS = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm'}


@dataclass
class VideoMetadata:
    """
    Container for video file metadata.

    Attributes:
        filepath: Path to the video file
        fps: Frames per second
        width: Video width in pixels
        height: Video height in pixels
        frame_count: Total number of frames
        duration: Duration in seconds
        codec: Video codec name
        pixel_to_mm: Conversion ratio from pixels to millimeters (optional)

    Example:
        >>> metadata = VideoMetadata.from_video("mouse_recording.mp4")
        >>> print(f"Duration: {metadata.duration:.2f}s at {metadata.fps} FPS")
    """

    filepath: str
    fps: float
    width: int
    height: int
    frame_count: int
    duration: float
    codec: str = "unknown"
    pixel_to_mm: Optional[float] = None

    @classmethod
    def from_video(
        cls,
        video_path: str,
        pixel_to_mm: Optional[float] = None,
        calibration_length_mm: Optional[float] = None,
        calibration_length_px: Optional[float] = None
    ) -> "VideoMetadata":
        """
        Extract metadata from a video file.

        Args:
            video_path: Path to the video file
            pixel_to_mm: Direct pixel-to-mm conversion ratio
            calibration_length_mm: Known length in mm for calibration
            calibration_length_px: Same length in pixels for calibration

        Returns:
            VideoMetadata instance with extracted information

        Raises:
            FileNotFoundError: If video file doesn't exist
            ValueError: If video cannot be opened or format unsupported
            ImportError: If OpenCV is not installed
        """
        try:
            import cv2
        except ImportError:
            raise ImportError(
                "OpenCV is required for video metadata extraction. "
                "Install with: pip install opencv-python"
            )

        path = Path(video_path)

        # Validate file exists
        if not path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        # Check format
        suffix = path.suffix.lower()
        if suffix not in SUPPORTED_FORMATS:
            logger.warning(
                f"Video format '{suffix}' may not be fully supported. "
                f"Supported formats: {SUPPORTED_FORMATS}"
            )

        # Open video
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        try:
            # Extract metadata
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))

            # Decode codec
            codec = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
            codec = codec.strip() if codec.isprintable() else "unknown"

            # Calculate duration
            duration = frame_count / fps if fps > 0 else 0.0

            # Handle calibration
            px_to_mm = pixel_to_mm
            if px_to_mm is None and calibration_length_mm and calibration_length_px:
                px_to_mm = calibration_length_mm / calibration_length_px
                logger.info(
                    f"Calculated pixel_to_mm ratio: {px_to_mm:.6f} "
                    f"({calibration_length_mm}mm / {calibration_length_px}px)"
                )

        finally:
            cap.release()

        metadata = cls(
            filepath=str(path.absolute()),
            fps=fps,
            width=width,
            height=height,
            frame_count=frame_count,
            duration=duration,
            codec=codec,
            pixel_to_mm=px_to_mm
        )

        logger.info(
            f"Loaded video metadata: {path.name} - "
            f"{width}x{height} @ {fps:.2f}fps, {duration:.2f}s ({frame_count} frames)"
        )

        return metadata

    def get_frame_count(self) -> int:
        """
        Get the total number of frames in the video.

        Returns:
            Total frame count
        """
        return self.frame_count

    def get_scale(self) -> Optional[float]:
        """
        Get the pixel-to-mm conversion scale.

        Returns:
            Pixel to millimeter ratio, or None if not calibrated
        """
        return self.pixel_to_mm

    def set_scale(
        self,
        pixel_to_mm: Optional[float] = None,
        calibration_length_mm: Optional[float] = None,
        calibration_length_px: Optional[float] = None
    ) -> None:
        """
        Set or update the pixel-to-mm scale.

        Args:
            pixel_to_mm: Direct pixel-to-mm ratio
            calibration_length_mm: Known length in mm
            calibration_length_px: Same length in pixels
        """
        if pixel_to_mm is not None:
            self.pixel_to_mm = pixel_to_mm
        elif calibration_length_mm and calibration_length_px:
            self.pixel_to_mm = calibration_length_mm / calibration_length_px
        else:
            raise ValueError(
                "Must provide either pixel_to_mm or both calibration_length_mm and calibration_length_px"
            )

    def pixels_to_mm(self, pixels: float) -> float:
        """
        Convert a distance from pixels to millimeters.

        Args:
            pixels: Distance in pixels

        Returns:
            Distance in millimeters

        Raises:
            ValueError: If pixel_to_mm scale not set
        """
        if self.pixel_to_mm is None:
            raise ValueError(
                "Pixel-to-mm scale not set. "
                "Use set_scale() or provide calibration during initialization."
            )
        return pixels * self.pixel_to_mm

    def mm_to_pixels(self, mm: float) -> float:
        """
        Convert a distance from millimeters to pixels.

        Args:
            mm: Distance in millimeters

        Returns:
            Distance in pixels

        Raises:
            ValueError: If pixel_to_mm scale not set
        """
        if self.pixel_to_mm is None:
            raise ValueError("Pixel-to-mm scale not set.")
        return mm / self.pixel_to_mm

    def frame_to_time(self, frame_number: int) -> float:
        """
        Convert frame number to time in seconds.

        Args:
            frame_number: Frame index (0-based)

        Returns:
            Time in seconds
        """
        return frame_number / self.fps if self.fps > 0 else 0.0

    def time_to_frame(self, time_seconds: float) -> int:
        """
        Convert time in seconds to frame number.

        Args:
            time_seconds: Time in seconds

        Returns:
            Frame number (0-based, rounded)
        """
        return int(round(time_seconds * self.fps))

    def get_resolution(self) -> Tuple[int, int]:
        """
        Get video resolution as (width, height) tuple.

        Returns:
            Tuple of (width, height) in pixels
        """
        return (self.width, self.height)

    def get_aspect_ratio(self) -> float:
        """
        Get video aspect ratio.

        Returns:
            Width divided by height
        """
        return self.width / self.height if self.height > 0 else 0.0

    def is_calibrated(self) -> bool:
        """
        Check if pixel-to-mm calibration is set.

        Returns:
            True if calibration is available
        """
        return self.pixel_to_mm is not None

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert metadata to dictionary.

        Returns:
            Dictionary with all metadata fields
        """
        return {
            'filepath': self.filepath,
            'fps': self.fps,
            'width': self.width,
            'height': self.height,
            'frame_count': self.frame_count,
            'duration': self.duration,
            'codec': self.codec,
            'pixel_to_mm': self.pixel_to_mm,
            'resolution': f"{self.width}x{self.height}",
            'aspect_ratio': self.get_aspect_ratio(),
            'is_calibrated': self.is_calibrated()
        }

    def __repr__(self) -> str:
        return (
            f"VideoMetadata('{Path(self.filepath).name}', "
            f"{self.width}x{self.height}, "
            f"{self.fps:.2f}fps, "
            f"{self.duration:.2f}s, "
            f"calibrated={self.is_calibrated()})"
        )


def probe_video(video_path: str) -> Dict[str, Any]:
    """
    Quick probe of video file for basic information without full metadata extraction.

    Args:
        video_path: Path to video file

    Returns:
        Dictionary with basic video information

    Raises:
        FileNotFoundError: If file doesn't exist
    """
    path = Path(video_path)

    if not path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    info = {
        'filename': path.name,
        'format': path.suffix.lower(),
        'size_bytes': path.stat().st_size,
        'size_mb': path.stat().st_size / (1024 * 1024),
        'supported': path.suffix.lower() in SUPPORTED_FORMATS
    }

    # Try to get more details if cv2 available
    try:
        import cv2
        cap = cv2.VideoCapture(str(path))
        if cap.isOpened():
            info['fps'] = cap.get(cv2.CAP_PROP_FPS)
            info['width'] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            info['height'] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            info['frame_count'] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
    except ImportError:
        pass

    return info


def estimate_memory_usage(metadata: VideoMetadata, dtype_bytes: int = 4) -> Dict[str, float]:
    """
    Estimate memory usage for loading video frames.

    Args:
        metadata: Video metadata
        dtype_bytes: Bytes per pixel value (4 for float32, 8 for float64)

    Returns:
        Dictionary with memory estimates in MB
    """
    pixels_per_frame = metadata.width * metadata.height
    channels = 3  # Assuming RGB

    frame_size_mb = (pixels_per_frame * channels * dtype_bytes) / (1024 * 1024)
    total_size_mb = frame_size_mb * metadata.frame_count

    return {
        'frame_size_mb': frame_size_mb,
        'total_size_mb': total_size_mb,
        'total_size_gb': total_size_mb / 1024,
        'frames': metadata.frame_count,
        'dtype_bytes': dtype_bytes
    }
