"""
Track Processor Module

Post-processing utilities for DeepLabCut tracking data including
filtering, interpolation, smoothing, and coordinate conversion.

Author: Stride Labs
License: MIT
"""

from typing import Optional, List, Tuple, Dict, Any, Union
from dataclasses import dataclass, field
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ProcessingConfig:
    """
    Configuration for track processing parameters.

    Attributes:
        confidence_threshold: Minimum confidence for valid points (0-1)
        max_gap_frames: Maximum gap size for interpolation
        smooth_window: Window size for Savitzky-Golay filter (must be odd)
        smooth_polyorder: Polynomial order for smoothing filter
        pixel_to_mm: Conversion ratio from pixels to millimeters
        remove_outliers: Whether to detect and remove outliers
        outlier_std_threshold: Number of standard deviations for outlier detection
    """

    confidence_threshold: float = 0.6
    max_gap_frames: int = 5
    smooth_window: int = 5
    smooth_polyorder: int = 2
    pixel_to_mm: Optional[float] = None
    remove_outliers: bool = True
    outlier_std_threshold: float = 3.0

    def validate(self) -> bool:
        """Validate configuration parameters."""
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be between 0 and 1")
        if self.max_gap_frames < 1:
            raise ValueError("max_gap_frames must be at least 1")
        if self.smooth_window < 3 or self.smooth_window % 2 == 0:
            raise ValueError("smooth_window must be odd and >= 3")
        if self.smooth_polyorder >= self.smooth_window:
            raise ValueError("smooth_polyorder must be less than smooth_window")
        if self.pixel_to_mm is not None and self.pixel_to_mm <= 0:
            raise ValueError("pixel_to_mm must be positive")
        return True


class TrackProcessor:
    """
    Processor for DeepLabCut tracking data.

    Handles post-processing of raw tracking output including:
    - Confidence-based filtering
    - Gap interpolation
    - Trajectory smoothing
    - Coordinate conversion (pixels to mm)
    - Outlier detection and removal

    Example:
        >>> processor = TrackProcessor(confidence_threshold=0.7, pixel_to_mm=0.1)
        >>> processed_df = processor.process(raw_tracking_df)
    """

    def __init__(
        self,
        confidence_threshold: float = 0.6,
        max_gap_frames: int = 5,
        smooth_window: int = 5,
        smooth_polyorder: int = 2,
        pixel_to_mm: Optional[float] = None,
        remove_outliers: bool = True,
        outlier_std_threshold: float = 3.0
    ):
        """
        Initialize the track processor.

        Args:
            confidence_threshold: Minimum confidence for valid points (0-1)
            max_gap_frames: Maximum gap size for linear interpolation
            smooth_window: Window size for Savitzky-Golay smoothing (must be odd)
            smooth_polyorder: Polynomial order for smoothing
            pixel_to_mm: Conversion ratio from pixels to mm (None to skip conversion)
            remove_outliers: Whether to detect and remove outliers
            outlier_std_threshold: Standard deviations for outlier detection
        """
        self.config = ProcessingConfig(
            confidence_threshold=confidence_threshold,
            max_gap_frames=max_gap_frames,
            smooth_window=smooth_window,
            smooth_polyorder=smooth_polyorder,
            pixel_to_mm=pixel_to_mm,
            remove_outliers=remove_outliers,
            outlier_std_threshold=outlier_std_threshold
        )
        self.config.validate()

        self._processing_stats: Dict[str, Any] = {}

    def process(
        self,
        tracks: pd.DataFrame,
        markers: Optional[List[str]] = None,
        apply_filter: bool = True,
        apply_interpolation: bool = True,
        apply_smoothing: bool = True,
        apply_conversion: bool = True
    ) -> pd.DataFrame:
        """
        Apply full processing pipeline to tracking data.

        Args:
            tracks: DataFrame with DeepLabCut output format
                   (MultiIndex columns: scorer, bodypart, coords)
            markers: List of markers to process (None for all)
            apply_filter: Apply confidence filtering
            apply_interpolation: Apply gap interpolation
            apply_smoothing: Apply trajectory smoothing
            apply_conversion: Apply pixel-to-mm conversion

        Returns:
            Processed DataFrame with same structure

        Raises:
            ValueError: If input format is invalid
        """
        logger.info("Starting track processing pipeline")

        # Validate input
        if tracks.empty:
            logger.warning("Empty DataFrame provided")
            return tracks.copy()

        # Make a copy to avoid modifying original
        df = tracks.copy()

        # Detect and extract markers
        available_markers = self._get_markers(df)
        if markers is not None:
            # Filter to requested markers
            markers_to_process = [m for m in markers if m in available_markers]
            if not markers_to_process:
                raise ValueError(f"None of the requested markers found: {markers}")
        else:
            markers_to_process = available_markers

        logger.info(f"Processing {len(markers_to_process)} markers: {markers_to_process}")

        # Initialize stats
        self._processing_stats = {
            'markers_processed': len(markers_to_process),
            'total_frames': len(df),
            'points_filtered': 0,
            'gaps_interpolated': 0,
            'outliers_removed': 0
        }

        # Process each marker
        for marker in markers_to_process:
            df = self._process_marker(
                df, marker,
                apply_filter=apply_filter,
                apply_interpolation=apply_interpolation,
                apply_smoothing=apply_smoothing
            )

        # Apply coordinate conversion
        if apply_conversion and self.config.pixel_to_mm is not None:
            df = self._convert_coordinates(df, markers_to_process)

        logger.info(f"Processing complete. Stats: {self._processing_stats}")
        return df

    def filter_low_confidence(
        self,
        tracks: pd.DataFrame,
        markers: Optional[List[str]] = None,
        threshold: Optional[float] = None
    ) -> pd.DataFrame:
        """
        Filter out points with low confidence scores.

        Sets x and y coordinates to NaN for points below threshold.

        Args:
            tracks: DataFrame with tracking data
            markers: Markers to filter (None for all)
            threshold: Confidence threshold (uses config default if None)

        Returns:
            DataFrame with low-confidence points set to NaN
        """
        df = tracks.copy()
        threshold = threshold or self.config.confidence_threshold

        available_markers = self._get_markers(df)
        markers_to_filter = markers if markers else available_markers

        total_filtered = 0

        for marker in markers_to_filter:
            if marker not in available_markers:
                continue

            # Get likelihood column
            likelihood = self._get_column(df, marker, 'likelihood')
            if likelihood is None:
                continue

            # Create mask for low confidence
            low_conf_mask = likelihood < threshold
            n_filtered = low_conf_mask.sum()
            total_filtered += n_filtered

            if n_filtered > 0:
                # Set x and y to NaN
                x_col = self._get_column_name(df, marker, 'x')
                y_col = self._get_column_name(df, marker, 'y')

                if x_col:
                    df.loc[low_conf_mask, x_col] = np.nan
                if y_col:
                    df.loc[low_conf_mask, y_col] = np.nan

                logger.debug(f"Filtered {n_filtered} points for marker '{marker}'")

        self._processing_stats['points_filtered'] = total_filtered
        logger.info(f"Filtered {total_filtered} low-confidence points (threshold={threshold})")

        return df

    def interpolate_gaps(
        self,
        tracks: pd.DataFrame,
        markers: Optional[List[str]] = None,
        max_gap: Optional[int] = None,
        method: str = 'linear'
    ) -> pd.DataFrame:
        """
        Interpolate small gaps in tracking data.

        Args:
            tracks: DataFrame with tracking data (may contain NaN)
            markers: Markers to interpolate (None for all)
            max_gap: Maximum gap size to interpolate (uses config default if None)
            method: Interpolation method ('linear', 'spline', 'polynomial')

        Returns:
            DataFrame with interpolated values
        """
        df = tracks.copy()
        max_gap = max_gap or self.config.max_gap_frames

        available_markers = self._get_markers(df)
        markers_to_interpolate = markers if markers else available_markers

        total_interpolated = 0

        for marker in markers_to_interpolate:
            if marker not in available_markers:
                continue

            for coord in ['x', 'y']:
                col_name = self._get_column_name(df, marker, coord)
                if col_name is None:
                    continue

                series = df[col_name].copy()
                n_missing_before = series.isna().sum()

                # Find gap sizes
                gap_groups = self._identify_gaps(series)

                # Interpolate only small gaps
                for start, end in gap_groups:
                    gap_size = end - start
                    if gap_size <= max_gap:
                        # Perform interpolation
                        if method == 'linear':
                            series[start:end] = series[start:end].interpolate(method='linear')
                        elif method == 'spline':
                            # Use index-based interpolation for spline
                            series[start:end] = series.interpolate(method='spline', order=3)[start:end]
                        elif method == 'polynomial':
                            series[start:end] = series.interpolate(method='polynomial', order=2)[start:end]

                df[col_name] = series
                n_missing_after = df[col_name].isna().sum()
                total_interpolated += (n_missing_before - n_missing_after)

        self._processing_stats['gaps_interpolated'] = total_interpolated
        logger.info(f"Interpolated {total_interpolated} missing values (max_gap={max_gap})")

        return df

    def smooth(
        self,
        tracks: pd.DataFrame,
        markers: Optional[List[str]] = None,
        window: Optional[int] = None,
        polyorder: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Apply Savitzky-Golay smoothing filter to trajectories.

        Args:
            tracks: DataFrame with tracking data
            markers: Markers to smooth (None for all)
            window: Smoothing window size (uses config default if None)
            polyorder: Polynomial order (uses config default if None)

        Returns:
            DataFrame with smoothed trajectories
        """
        try:
            from scipy.signal import savgol_filter
        except ImportError:
            raise ImportError(
                "scipy is required for smoothing. Install with: pip install scipy"
            )

        df = tracks.copy()
        window = window or self.config.smooth_window
        polyorder = polyorder or self.config.smooth_polyorder

        # Validate parameters
        if window % 2 == 0:
            window += 1  # Must be odd
        if polyorder >= window:
            polyorder = window - 1

        available_markers = self._get_markers(df)
        markers_to_smooth = markers if markers else available_markers

        for marker in markers_to_smooth:
            if marker not in available_markers:
                continue

            for coord in ['x', 'y']:
                col_name = self._get_column_name(df, marker, coord)
                if col_name is None:
                    continue

                series = df[col_name]

                # Only smooth if we have enough valid points
                valid_mask = ~series.isna()
                if valid_mask.sum() < window:
                    logger.debug(f"Not enough points to smooth {marker}.{coord}")
                    continue

                # Apply filter to valid segments
                smoothed = series.copy()
                valid_indices = np.where(valid_mask)[0]

                # Find continuous segments
                segments = self._find_continuous_segments(valid_indices)

                for seg_start, seg_end in segments:
                    seg_length = seg_end - seg_start
                    if seg_length >= window:
                        seg_data = series.iloc[seg_start:seg_end].values
                        # Adjust window if segment is small
                        seg_window = min(window, seg_length if seg_length % 2 == 1 else seg_length - 1)
                        seg_polyorder = min(polyorder, seg_window - 1)

                        if seg_window >= 3:
                            smoothed_seg = savgol_filter(seg_data, seg_window, seg_polyorder)
                            smoothed.iloc[seg_start:seg_end] = smoothed_seg

                df[col_name] = smoothed

        logger.info(f"Applied Savitzky-Golay smoothing (window={window}, polyorder={polyorder})")
        return df

    def remove_outliers(
        self,
        tracks: pd.DataFrame,
        markers: Optional[List[str]] = None,
        std_threshold: Optional[float] = None
    ) -> pd.DataFrame:
        """
        Detect and remove outlier points based on velocity.

        Uses velocity-based detection to find physically implausible jumps.

        Args:
            tracks: DataFrame with tracking data
            markers: Markers to check (None for all)
            std_threshold: Number of standard deviations for outlier detection

        Returns:
            DataFrame with outliers set to NaN
        """
        df = tracks.copy()
        threshold = std_threshold or self.config.outlier_std_threshold

        available_markers = self._get_markers(df)
        markers_to_check = markers if markers else available_markers

        total_outliers = 0

        for marker in markers_to_check:
            if marker not in available_markers:
                continue

            x_col = self._get_column_name(df, marker, 'x')
            y_col = self._get_column_name(df, marker, 'y')

            if x_col is None or y_col is None:
                continue

            x = df[x_col].values
            y = df[y_col].values

            # Calculate velocity
            dx = np.diff(x, prepend=x[0])
            dy = np.diff(y, prepend=y[0])
            velocity = np.sqrt(dx**2 + dy**2)

            # Find outliers based on velocity
            valid_vel = velocity[~np.isnan(velocity)]
            if len(valid_vel) > 10:
                vel_mean = np.nanmean(velocity)
                vel_std = np.nanstd(velocity)
                outlier_mask = velocity > (vel_mean + threshold * vel_std)

                n_outliers = outlier_mask.sum()
                if n_outliers > 0:
                    df.loc[outlier_mask, x_col] = np.nan
                    df.loc[outlier_mask, y_col] = np.nan
                    total_outliers += n_outliers
                    logger.debug(f"Removed {n_outliers} outliers for marker '{marker}'")

        self._processing_stats['outliers_removed'] = total_outliers
        logger.info(f"Removed {total_outliers} outlier points")

        return df

    def convert_to_mm(
        self,
        tracks: pd.DataFrame,
        pixel_to_mm: Optional[float] = None,
        markers: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Convert coordinates from pixels to millimeters.

        Args:
            tracks: DataFrame with pixel coordinates
            pixel_to_mm: Conversion ratio (uses config default if None)
            markers: Markers to convert (None for all)

        Returns:
            DataFrame with mm coordinates
        """
        scale = pixel_to_mm or self.config.pixel_to_mm

        if scale is None:
            raise ValueError("pixel_to_mm scale not provided")

        return self._convert_coordinates(tracks, markers, scale)

    def get_processing_stats(self) -> Dict[str, Any]:
        """
        Get statistics from the last processing run.

        Returns:
            Dictionary with processing statistics
        """
        return self._processing_stats.copy()

    # =========================================================================
    # Private helper methods
    # =========================================================================

    def _process_marker(
        self,
        df: pd.DataFrame,
        marker: str,
        apply_filter: bool,
        apply_interpolation: bool,
        apply_smoothing: bool
    ) -> pd.DataFrame:
        """Process a single marker through the pipeline."""
        if apply_filter:
            df = self.filter_low_confidence(df, markers=[marker])

        if self.config.remove_outliers:
            df = self.remove_outliers(df, markers=[marker])

        if apply_interpolation:
            df = self.interpolate_gaps(df, markers=[marker])

        if apply_smoothing:
            df = self.smooth(df, markers=[marker])

        return df

    def _get_markers(self, df: pd.DataFrame) -> List[str]:
        """Extract list of markers from DataFrame columns."""
        if isinstance(df.columns, pd.MultiIndex):
            # DeepLabCut format: (scorer, bodypart, coords)
            if df.columns.nlevels >= 2:
                # Get unique bodyparts from second level
                return list(df.columns.get_level_values(1).unique())
        else:
            # Flat column format: try to parse marker_x, marker_y pattern
            markers = set()
            for col in df.columns:
                if '_x' in col:
                    markers.add(col.replace('_x', ''))
                elif '_y' in col:
                    markers.add(col.replace('_y', ''))
            return list(markers)
        return []

    def _get_column(
        self,
        df: pd.DataFrame,
        marker: str,
        coord: str
    ) -> Optional[pd.Series]:
        """Get a column series by marker and coordinate name."""
        col_name = self._get_column_name(df, marker, coord)
        if col_name is not None:
            return df[col_name]
        return None

    def _get_column_name(
        self,
        df: pd.DataFrame,
        marker: str,
        coord: str
    ) -> Optional[Union[str, Tuple]]:
        """Get the column name/index for a marker coordinate."""
        if isinstance(df.columns, pd.MultiIndex):
            # DeepLabCut MultiIndex format
            for col in df.columns:
                if len(col) >= 3 and col[1] == marker and col[2] == coord:
                    return col
                elif len(col) >= 2 and col[0] == marker and col[1] == coord:
                    return col
        else:
            # Flat format: marker_coord
            flat_name = f"{marker}_{coord}"
            if flat_name in df.columns:
                return flat_name

        return None

    def _convert_coordinates(
        self,
        df: pd.DataFrame,
        markers: Optional[List[str]] = None,
        scale: Optional[float] = None
    ) -> pd.DataFrame:
        """Convert pixel coordinates to mm."""
        df = df.copy()
        scale = scale or self.config.pixel_to_mm

        if scale is None:
            return df

        available_markers = self._get_markers(df)
        markers_to_convert = markers if markers else available_markers

        for marker in markers_to_convert:
            if marker not in available_markers:
                continue

            for coord in ['x', 'y']:
                col_name = self._get_column_name(df, marker, coord)
                if col_name is not None:
                    df[col_name] = df[col_name] * scale

        logger.info(f"Converted coordinates to mm (scale={scale})")
        return df

    def _identify_gaps(self, series: pd.Series) -> List[Tuple[int, int]]:
        """Identify contiguous NaN gaps in a series."""
        is_nan = series.isna()
        gaps = []

        in_gap = False
        gap_start = 0

        for i, val in enumerate(is_nan):
            if val and not in_gap:
                in_gap = True
                gap_start = i
            elif not val and in_gap:
                gaps.append((gap_start, i))
                in_gap = False

        if in_gap:
            gaps.append((gap_start, len(series)))

        return gaps

    def _find_continuous_segments(
        self,
        indices: np.ndarray
    ) -> List[Tuple[int, int]]:
        """Find continuous segments in array of indices."""
        if len(indices) == 0:
            return []

        segments = []
        start = indices[0]

        for i in range(1, len(indices)):
            if indices[i] - indices[i-1] > 1:
                segments.append((start, indices[i-1] + 1))
                start = indices[i]

        segments.append((start, indices[-1] + 1))
        return segments


def create_processor_from_config(config: Dict[str, Any]) -> TrackProcessor:
    """
    Create a TrackProcessor from a configuration dictionary.

    Args:
        config: Dictionary with processing parameters

    Returns:
        Configured TrackProcessor instance
    """
    return TrackProcessor(
        confidence_threshold=config.get('confidence_threshold', 0.6),
        max_gap_frames=config.get('max_gap_frames', 5),
        smooth_window=config.get('smooth_window', 5),
        smooth_polyorder=config.get('smooth_polyorder', 2),
        pixel_to_mm=config.get('pixel_to_mm'),
        remove_outliers=config.get('remove_outliers', True),
        outlier_std_threshold=config.get('outlier_std_threshold', 3.0)
    )
