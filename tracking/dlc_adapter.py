"""
DeepLabCut Adapter Module

Provides a wrapper around the DeepLabCut API for analyzing mouse locomotion
videos using either pre-trained SuperAnimal models or custom trained models.

Author: Stride Labs
License: MIT
"""

from typing import Optional, List, Dict, Any, Union, Tuple
from pathlib import Path
import logging
import warnings

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# Attempt to import DeepLabCut
try:
    import deeplabcut
    DLC_AVAILABLE = True
    DLC_VERSION = deeplabcut.__version__
except ImportError:
    DLC_AVAILABLE = False
    DLC_VERSION = None
    deeplabcut = None


class DeepLabCutNotAvailableError(ImportError):
    """Raised when DeepLabCut is not installed."""
    pass


class DeepLabCutAdapter:
    """
    Adapter class for DeepLabCut pose estimation.

    Provides a unified interface for running pose estimation using either
    pre-trained SuperAnimal models (TopViewMouse, QuadrupedHorse, etc.)
    or custom trained DeepLabCut models.

    Attributes:
        model_type: Type of model ('superanimal' or 'custom')
        model_path: Path to custom model config (for custom models)
        superanimal_name: Name of SuperAnimal model (for superanimal models)
        device: Computing device ('cuda', 'cpu', 'auto')

    Example:
        >>> # Using SuperAnimal pre-trained model
        >>> adapter = DeepLabCutAdapter.from_superanimal("TopViewMouse")
        >>> tracks = adapter.analyze_video("mouse_video.mp4")

        >>> # Using custom trained model
        >>> adapter = DeepLabCutAdapter.from_custom("/path/to/config.yaml")
        >>> tracks = adapter.analyze_video("mouse_video.mp4")
    """

    # Available SuperAnimal models
    SUPERANIMAL_MODELS = {
        'TopViewMouse': 'superanimal_topviewmouse',
        'QuadrupedHorse': 'superanimal_quadruped',
        'Quadruped': 'superanimal_quadruped',
    }

    def __init__(
        self,
        model_type: str = 'superanimal',
        model_path: Optional[str] = None,
        superanimal_name: str = 'TopViewMouse',
        device: str = 'auto',
        video_type: str = 'mp4'
    ):
        """
        Initialize the DeepLabCut adapter.

        Args:
            model_type: Type of model to use ('superanimal' or 'custom')
            model_path: Path to config.yaml for custom models
            superanimal_name: Name of SuperAnimal model to use
            device: Computing device ('cuda', 'cpu', 'auto')
            video_type: Default video format

        Raises:
            DeepLabCutNotAvailableError: If DeepLabCut is not installed
            ValueError: If invalid model configuration provided
        """
        if not DLC_AVAILABLE:
            raise DeepLabCutNotAvailableError(
                "DeepLabCut is not installed. Install with:\n"
                "  pip install deeplabcut\n"
                "or for GPU support:\n"
                "  pip install 'deeplabcut[tf]'"
            )

        self.model_type = model_type
        self.model_path = model_path
        self.superanimal_name = superanimal_name
        self.device = device
        self.video_type = video_type

        # Validate configuration
        self._validate_config()

        # Analysis results cache
        self._last_analysis: Optional[Dict[str, Any]] = None

        logger.info(
            f"Initialized DeepLabCutAdapter (version={DLC_VERSION}, "
            f"model_type={model_type}, device={device})"
        )

    @classmethod
    def from_superanimal(
        cls,
        model_name: str = 'TopViewMouse',
        device: str = 'auto'
    ) -> 'DeepLabCutAdapter':
        """
        Create adapter using a pre-trained SuperAnimal model.

        Args:
            model_name: Name of SuperAnimal model
                - 'TopViewMouse': For ventral/top view recordings
                - 'Quadruped' or 'QuadrupedHorse': For lateral view recordings
            device: Computing device ('cuda', 'cpu', 'auto')

        Returns:
            Configured DeepLabCutAdapter instance

        Example:
            >>> adapter = DeepLabCutAdapter.from_superanimal("TopViewMouse")
        """
        return cls(
            model_type='superanimal',
            superanimal_name=model_name,
            device=device
        )

    @classmethod
    def from_custom(
        cls,
        config_path: str,
        device: str = 'auto'
    ) -> 'DeepLabCutAdapter':
        """
        Create adapter using a custom trained DeepLabCut model.

        Args:
            config_path: Path to the project's config.yaml file
            device: Computing device ('cuda', 'cpu', 'auto')

        Returns:
            Configured DeepLabCutAdapter instance

        Raises:
            FileNotFoundError: If config file doesn't exist

        Example:
            >>> adapter = DeepLabCutAdapter.from_custom("/projects/dlc/config.yaml")
        """
        config = Path(config_path)
        if not config.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        return cls(
            model_type='custom',
            model_path=str(config),
            device=device
        )

    def analyze_video(
        self,
        video_path: str,
        output_dir: Optional[str] = None,
        save_as_csv: bool = True,
        destfolder: Optional[str] = None,
        shuffle: int = 1,
        trainingsetindex: int = 0,
        gputouse: Optional[int] = None,
        save_video: bool = False,
        **kwargs
    ) -> pd.DataFrame:
        """
        Analyze a video to extract pose estimations.

        Args:
            video_path: Path to the video file
            output_dir: Directory to save results (uses video directory if None)
            save_as_csv: Whether to save results as CSV
            destfolder: Destination folder for output files
            shuffle: Shuffle index for trained model
            trainingsetindex: Training set index
            gputouse: GPU ID to use (None for auto)
            save_video: Whether to create labeled video
            **kwargs: Additional arguments passed to DLC analyze function

        Returns:
            DataFrame with pose estimation results
            Columns are MultiIndex: (scorer, bodypart, coordinate)
            where coordinate is 'x', 'y', or 'likelihood'

        Raises:
            FileNotFoundError: If video file doesn't exist
            RuntimeError: If analysis fails

        Example:
            >>> tracks = adapter.analyze_video("mouse.mp4")
            >>> print(tracks.columns.levels)  # [scorer], [bodyparts], [x, y, likelihood]
        """
        video = Path(video_path)
        if not video.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        dest = destfolder or output_dir or str(video.parent)

        logger.info(f"Analyzing video: {video.name}")
        logger.info(f"Model: {self.model_type} ({self.superanimal_name if self.model_type == 'superanimal' else self.model_path})")

        try:
            if self.model_type == 'superanimal':
                # Use SuperAnimal model
                result = self._analyze_with_superanimal(
                    video_path=str(video),
                    destfolder=dest,
                    **kwargs
                )
            else:
                # Use custom model
                result = self._analyze_with_custom(
                    video_path=str(video),
                    destfolder=dest,
                    shuffle=shuffle,
                    trainingsetindex=trainingsetindex,
                    gputouse=gputouse,
                    save_as_csv=save_as_csv,
                    **kwargs
                )

            # Load results
            tracks = self.load_tracks(video_path, search_dir=dest)

            # Cache analysis info
            self._last_analysis = {
                'video_path': str(video),
                'output_dir': dest,
                'model_type': self.model_type,
                'n_frames': len(tracks),
                'bodyparts': list(tracks.columns.get_level_values(1).unique())
            }

            # Create labeled video if requested
            if save_video:
                self.create_labeled_video(video_path, destfolder=dest)

            logger.info(f"Analysis complete: {len(tracks)} frames processed")
            return tracks

        except Exception as e:
            logger.error(f"Video analysis failed: {str(e)}")
            raise RuntimeError(f"DeepLabCut analysis failed: {str(e)}") from e

    def _analyze_with_superanimal(
        self,
        video_path: str,
        destfolder: str,
        **kwargs
    ) -> Any:
        """Run analysis using SuperAnimal model."""
        superanimal_id = self.SUPERANIMAL_MODELS.get(
            self.superanimal_name,
            self.superanimal_name
        )

        # SuperAnimal analysis
        # Note: API may vary by DLC version
        try:
            # Try newer API (DLC 3.x)
            result = deeplabcut.video_inference_superanimal(
                videos=[video_path],
                superanimal_name=superanimal_id,
                dest_folder=destfolder,
                **kwargs
            )
        except AttributeError:
            # Fall back to older API
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = deeplabcut.analyze_videos(
                    config=superanimal_id,
                    videos=[video_path],
                    destfolder=destfolder,
                    **kwargs
                )

        return result

    def _analyze_with_custom(
        self,
        video_path: str,
        destfolder: str,
        shuffle: int,
        trainingsetindex: int,
        gputouse: Optional[int],
        save_as_csv: bool,
        **kwargs
    ) -> Any:
        """Run analysis using custom trained model."""
        return deeplabcut.analyze_videos(
            config=self.model_path,
            videos=[video_path],
            shuffle=shuffle,
            trainingsetindex=trainingsetindex,
            gputouse=gputouse,
            destfolder=destfolder,
            save_as_csv=save_as_csv,
            **kwargs
        )

    def create_labeled_video(
        self,
        video_path: str,
        destfolder: Optional[str] = None,
        output_path: Optional[str] = None,
        draw_skeleton: bool = True,
        pcutoff: float = 0.6,
        dotsize: int = 8,
        colormap: str = 'rainbow',
        skeleton_color: str = 'white',
        trailpoints: int = 0,
        **kwargs
    ) -> str:
        """
        Create a labeled video with pose overlays.

        Args:
            video_path: Path to original video
            destfolder: Destination folder for output
            output_path: Full path for output video (overrides destfolder)
            draw_skeleton: Whether to draw skeleton connections
            pcutoff: Minimum confidence to draw point
            dotsize: Size of marker dots
            colormap: Matplotlib colormap for markers
            skeleton_color: Color for skeleton lines
            trailpoints: Number of trail points to draw (0 = no trail)
            **kwargs: Additional arguments for DLC create_labeled_video

        Returns:
            Path to created labeled video

        Example:
            >>> labeled = adapter.create_labeled_video("mouse.mp4", draw_skeleton=True)
        """
        video = Path(video_path)
        dest = destfolder or str(video.parent)

        logger.info(f"Creating labeled video for: {video.name}")

        try:
            if self.model_type == 'superanimal':
                # SuperAnimal labeled video
                try:
                    deeplabcut.create_video_with_all_detections(
                        videos=[str(video)],
                        superanimal_name=self.SUPERANIMAL_MODELS.get(
                            self.superanimal_name, self.superanimal_name
                        ),
                        destfolder=dest,
                        pcutoff=pcutoff,
                        dotsize=dotsize,
                        colormap=colormap,
                        **kwargs
                    )
                except (AttributeError, TypeError):
                    # Alternative method
                    deeplabcut.create_labeled_video(
                        config=self.SUPERANIMAL_MODELS.get(
                            self.superanimal_name, self.superanimal_name
                        ),
                        videos=[str(video)],
                        destfolder=dest,
                        draw_skeleton=draw_skeleton,
                        pcutoff=pcutoff,
                        dotsize=dotsize,
                        colormap=colormap,
                        trailpoints=trailpoints,
                        **kwargs
                    )
            else:
                # Custom model labeled video
                deeplabcut.create_labeled_video(
                    config=self.model_path,
                    videos=[str(video)],
                    destfolder=dest,
                    draw_skeleton=draw_skeleton,
                    pcutoff=pcutoff,
                    dotsize=dotsize,
                    colormap=colormap,
                    skeleton_color=skeleton_color,
                    trailpoints=trailpoints,
                    **kwargs
                )

            # Find output file
            labeled_video = self._find_labeled_video(video_path, dest)
            logger.info(f"Labeled video created: {labeled_video}")

            return labeled_video

        except Exception as e:
            logger.error(f"Failed to create labeled video: {str(e)}")
            raise RuntimeError(f"Failed to create labeled video: {str(e)}") from e

    def load_tracks(
        self,
        video_path: str,
        search_dir: Optional[str] = None,
        h5_path: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Load tracking results from analysis output files.

        Args:
            video_path: Original video path (used to find output file)
            search_dir: Directory to search for results
            h5_path: Direct path to H5 file (skips search if provided)

        Returns:
            DataFrame with pose estimation data

        Raises:
            FileNotFoundError: If results file not found
        """
        if h5_path:
            h5_file = Path(h5_path)
            if not h5_file.exists():
                raise FileNotFoundError(f"H5 file not found: {h5_path}")
        else:
            h5_file = self._find_h5_file(video_path, search_dir)

        logger.info(f"Loading tracks from: {h5_file}")

        # Read H5 file
        df = pd.read_hdf(str(h5_file))

        # Ensure proper MultiIndex format
        if not isinstance(df.columns, pd.MultiIndex):
            logger.warning("Unexpected column format in H5 file")

        return df

    def get_available_models(self) -> List[str]:
        """
        Get list of available SuperAnimal models.

        Returns:
            List of model names
        """
        return list(self.SUPERANIMAL_MODELS.keys())

    def get_last_analysis_info(self) -> Optional[Dict[str, Any]]:
        """
        Get information about the last analysis run.

        Returns:
            Dictionary with analysis info, or None if no analysis performed
        """
        return self._last_analysis.copy() if self._last_analysis else None

    # =========================================================================
    # Private helper methods
    # =========================================================================

    def _validate_config(self) -> None:
        """Validate adapter configuration."""
        if self.model_type not in ['superanimal', 'custom']:
            raise ValueError(f"Invalid model_type: {self.model_type}")

        if self.model_type == 'custom' and not self.model_path:
            raise ValueError("model_path required for custom models")

        if self.model_type == 'superanimal':
            if self.superanimal_name not in self.SUPERANIMAL_MODELS:
                available = list(self.SUPERANIMAL_MODELS.keys())
                logger.warning(
                    f"Unknown SuperAnimal model '{self.superanimal_name}'. "
                    f"Available: {available}. Proceeding anyway."
                )

    def _find_h5_file(
        self,
        video_path: str,
        search_dir: Optional[str]
    ) -> Path:
        """Find H5 results file for a video."""
        video = Path(video_path)
        video_stem = video.stem
        search_path = Path(search_dir) if search_dir else video.parent

        # Search patterns
        patterns = [
            f"{video_stem}*.h5",
            f"*{video_stem}*.h5",
            "*.h5"
        ]

        for pattern in patterns:
            matches = list(search_path.glob(pattern))
            if matches:
                # Return most recent if multiple
                return max(matches, key=lambda p: p.stat().st_mtime)

        raise FileNotFoundError(
            f"No H5 results file found for video '{video.name}' in {search_path}"
        )

    def _find_labeled_video(
        self,
        video_path: str,
        search_dir: str
    ) -> str:
        """Find labeled video output file."""
        video = Path(video_path)
        video_stem = video.stem
        search_path = Path(search_dir)

        # Search patterns for labeled videos
        patterns = [
            f"{video_stem}*labeled*.mp4",
            f"{video_stem}*labeled*.avi",
            f"*{video_stem}*labeled*"
        ]

        for pattern in patterns:
            matches = list(search_path.glob(pattern))
            if matches:
                return str(max(matches, key=lambda p: p.stat().st_mtime))

        return str(search_path / f"{video_stem}_labeled.mp4")


def check_dlc_installation() -> Dict[str, Any]:
    """
    Check DeepLabCut installation status and capabilities.

    Returns:
        Dictionary with installation info
    """
    info = {
        'installed': DLC_AVAILABLE,
        'version': DLC_VERSION,
        'gpu_available': False,
        'tensorflow_version': None,
        'pytorch_version': None
    }

    if DLC_AVAILABLE:
        # Check TensorFlow
        try:
            import tensorflow as tf
            info['tensorflow_version'] = tf.__version__
            info['gpu_available'] = len(tf.config.list_physical_devices('GPU')) > 0
        except ImportError:
            pass

        # Check PyTorch (for newer DLC versions)
        try:
            import torch
            info['pytorch_version'] = torch.__version__
            if not info['gpu_available']:
                info['gpu_available'] = torch.cuda.is_available()
        except ImportError:
            pass

    return info


def get_superanimal_bodyparts(model_name: str) -> List[str]:
    """
    Get the body parts for a SuperAnimal model.

    Args:
        model_name: Name of the SuperAnimal model

    Returns:
        List of body part names
    """
    # These are approximate - actual list depends on DLC version
    bodyparts = {
        'TopViewMouse': [
            'snout', 'leftear', 'rightear', 'neck',
            'leftforepaw', 'rightforepaw', 'spine1', 'spine2', 'spine3',
            'lefthindpaw', 'righthindpaw', 'tailbase', 'tailmid', 'tailtip'
        ],
        'Quadruped': [
            'nose', 'lefteye', 'righteye', 'leftear', 'rightear',
            'neck', 'withers', 'spine', 'hip', 'tailbase',
            'leftfront', 'rightfront', 'leftback', 'rightback'
        ]
    }

    return bodyparts.get(model_name, [])
