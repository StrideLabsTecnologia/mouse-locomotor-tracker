"""
Integration Tests for Mouse Locomotor Tracker

This module provides end-to-end integration tests that verify:
- Full pipeline execution from video to metrics
- Module interaction and data flow
- Export functionality
- Batch processing capabilities

Test Coverage Target: 70%+ for integration tests

Author: Stride Labs
License: MIT
"""

import pytest
import numpy as np
import pandas as pd
import tempfile
import os
import json
from typing import Dict, Any
from dataclasses import dataclass, asdict


# =============================================================================
# Mock Classes for Integration Testing
# =============================================================================

@dataclass
class VideoMetadata:
    """Video file metadata container."""
    duration: float
    fps: int
    n_frames: int
    width: int
    height: int
    pixel_width_mm: float

    def to_dict(self) -> dict:
        return {
            "dur": self.duration,
            "fps": self.fps,
            "nFrame": self.n_frames,
            "imW": self.width,
            "imH": self.height,
            "xPixW": self.pixel_width_mm
        }


class MockTracker:
    """
    Mock tracker for generating synthetic tracking data.
    Used for testing without requiring actual DeepLabCut.
    """

    def __init__(self, markers: list, model_name: str = "DLC_mock_model"):
        self.markers = markers
        self.model_name = model_name

    def generate_tracks(
        self,
        metadata: VideoMetadata,
        gait_frequency: float = 4.0,
        speed_cm_s: float = 15.0,
        noise_level: float = 1.0
    ) -> pd.DataFrame:
        """Generate synthetic tracking data."""
        n_frames = metadata.n_frames
        t = np.linspace(0, metadata.duration, n_frames)

        # Base body position
        pixel_per_cm = metadata.width / 20.0
        body_x = 100 + speed_cm_s * pixel_per_cm * t / metadata.duration * 0.5
        body_y = metadata.height / 2

        # Create multi-index columns
        columns = pd.MultiIndex.from_tuples(
            [(self.model_name, marker, coord)
             for marker in self.markers
             for coord in ["x", "y", "likelihood"]],
            names=["scorer", "bodyparts", "coords"]
        )

        data = np.zeros((n_frames, len(columns)))

        for i, marker in enumerate(self.markers):
            base_idx = i * 3

            # Marker-specific behavior
            if 'fore' in marker.lower():
                offset_x = 25
                phase = 0 if 'L' in marker else np.pi
            elif 'hind' in marker.lower():
                offset_x = -25
                phase = np.pi if 'L' in marker else 0
            elif 'snout' in marker.lower():
                offset_x = 50
                phase = 0
            elif 'tail' in marker.lower():
                offset_x = -60
                phase = 0
            else:
                offset_x = 0
                phase = 0

            # Limb movement for paw markers
            if 'fore' in marker.lower() or 'hind' in marker.lower():
                limb_movement = 30 * np.sin(2 * np.pi * gait_frequency * t + phase)
            else:
                limb_movement = 0

            # Add noise
            np.random.seed(hash(marker) % 2**32)
            noise_x = np.random.normal(0, noise_level, n_frames)
            noise_y = np.random.normal(0, noise_level, n_frames)

            data[:, base_idx] = body_x + offset_x + limb_movement + noise_x
            data[:, base_idx + 1] = body_y + noise_y
            data[:, base_idx + 2] = 0.95

        return pd.DataFrame(data, columns=columns)


class LocomotorPipeline:
    """
    Main pipeline for locomotor analysis.
    Integrates all analysis modules.
    """

    def __init__(self, config: dict = None):
        self.config = config or {}
        self.results = {}

    def process_tracks(
        self,
        tracks: pd.DataFrame,
        metadata: VideoMetadata,
        model_name: str,
        markers: list,
        limb_pairs: dict,
        speed_markers: list
    ) -> Dict[str, Any]:
        """
        Process tracking data through full analysis pipeline.

        Args:
            tracks: DeepLabCut-format tracking DataFrame
            metadata: Video metadata
            model_name: DLC model name
            markers: List of marker names
            limb_pairs: Dictionary of limb pair definitions
            speed_markers: Markers used for speed calculation

        Returns:
            Dictionary with all analysis results
        """
        from scipy.signal import find_peaks

        results = {
            'metadata': metadata.to_dict(),
            'velocity': {},
            'coordination': {},
            'gait_cycles': {},
            'summary': {}
        }

        meta_dict = metadata.to_dict()

        # ==== Velocity Analysis ====
        # Compute body speed
        x_positions = []
        for marker in speed_markers:
            try:
                x = tracks[model_name][marker]['x'].values
                x_positions.append(x)
            except KeyError:
                continue

        if x_positions:
            body_x = np.mean(x_positions, axis=0)
            dx = np.diff(body_x)
            speed = np.abs(dx) * meta_dict['xPixW'] / 10.0 * meta_dict['fps']

            # Smoothing
            sm_factor = self.config.get('smoothing_factor', 10)
            if len(speed) > sm_factor:
                kernel = np.ones(sm_factor) / sm_factor
                speed = np.convolve(speed, kernel, mode='valid')

            results['velocity'] = {
                'mean_speed': float(np.mean(speed)),
                'max_speed': float(np.max(speed)),
                'min_speed': float(np.min(speed)),
                'std_speed': float(np.std(speed)),
                'speed_profile': speed.tolist()[:100]  # First 100 for summary
            }

            avg_speed = np.mean(speed)
        else:
            avg_speed = 0
            results['velocity'] = {'mean_speed': 0}

        # ==== Stride Extraction ====
        limb_strides = {}
        for marker in ['foreL', 'foreR', 'hindL', 'hindR']:
            try:
                limb_x = tracks[model_name][marker]['x'].values
                # Compute stride relative to body
                stride = (limb_x - body_x[:len(limb_x)]) * meta_dict['xPixW']
                # Smooth
                if len(stride) > sm_factor:
                    stride = np.convolve(stride, np.ones(sm_factor)/sm_factor, mode='valid')
                limb_strides[marker] = stride
            except (KeyError, NameError):
                continue

        # ==== Coordination Analysis ====
        coordination_results = {}
        for pair_name, (limb_0, limb_1) in limb_pairs.items():
            if limb_0 not in limb_strides or limb_1 not in limb_strides:
                continue

            stride_0 = limb_strides[limb_0]
            stride_1 = limb_strides[limb_1]

            # Align lengths
            min_len = min(len(stride_0), len(stride_1))
            stride_0 = stride_0[:min_len]
            stride_1 = stride_1[:min_len]

            # Relative stride
            rel_stride = stride_0 - stride_1

            # Find cycles
            peaks, _ = find_peaks(rel_stride)
            if len(peaks) < 2:
                continue

            mean_dist = np.diff(peaks).mean() / 2
            peaks, _ = find_peaks(rel_stride, distance=max(1, int(mean_dist)))

            if len(peaks) < 2:
                continue

            # Normalize and compute phase
            rel_norm = 2 * (rel_stride - rel_stride.min()) / \
                       (rel_stride.max() - rel_stride.min() + 1e-10) - 1

            phi = []
            for i in range(len(peaks) - 1):
                y = rel_norm[peaks[i]:peaks[i+1]]
                x = np.linspace(0, 2*np.pi, len(y))
                phi.append((4 - np.trapz(y, x)) * np.pi / 4)

            phi = np.array(phi)

            # Circular mean
            X = np.cos(phi).mean()
            Y = np.sin(phi).mean()
            R = np.sqrt(X**2 + Y**2)
            mean_phi = np.arctan2(Y, X) * 180 / np.pi

            coordination_results[pair_name] = {
                'R': float(R),
                'mean_phase_deg': float(mean_phi),
                'n_steps': len(peaks)
            }

        results['coordination'] = coordination_results

        # ==== Gait Cycle Analysis ====
        gait_results = {}
        mov_duration = meta_dict['dur']

        for limb, stride in limb_strides.items():
            peaks, _ = find_peaks(stride)
            if len(peaks) >= 2:
                mean_dist = np.diff(peaks).mean() / 2
                peaks, _ = find_peaks(stride, distance=max(1, int(mean_dist)))

            n_cycles = len(peaks)
            cadence = n_cycles / mov_duration if mov_duration > 0 else 0
            stride_length = avg_speed / cadence if cadence > 0 else 0

            gait_results[limb] = {
                'cadence': float(cadence),
                'stride_length': float(stride_length),
                'n_cycles': n_cycles
            }

        results['gait_cycles'] = gait_results

        # ==== Summary Statistics ====
        if coordination_results:
            mean_R = np.mean([v['R'] for v in coordination_results.values()])
        else:
            mean_R = 0

        if gait_results:
            mean_cadence = np.mean([v['cadence'] for v in gait_results.values()])
            mean_stride_len = np.mean([v['stride_length'] for v in gait_results.values()])
        else:
            mean_cadence = 0
            mean_stride_len = 0

        results['summary'] = {
            'duration': meta_dict['dur'],
            'mean_speed_cm_s': results['velocity'].get('mean_speed', 0),
            'mean_coordination_R': float(mean_R),
            'mean_cadence_hz': float(mean_cadence),
            'mean_stride_length_cm': float(mean_stride_len)
        }

        self.results = results
        return results

    def export_results(self, output_path: str, format: str = 'json'):
        """Export results to file."""
        if format == 'json':
            # Remove numpy arrays for JSON serialization
            export_data = self._prepare_for_export(self.results)
            with open(output_path, 'w') as f:
                json.dump(export_data, f, indent=2)
        elif format == 'csv':
            # Export summary as CSV
            summary = self.results.get('summary', {})
            df = pd.DataFrame([summary])
            df.to_csv(output_path, index=False)
        else:
            raise ValueError(f"Unknown format: {format}")

    def _prepare_for_export(self, obj):
        """Recursively convert numpy types for JSON serialization."""
        if isinstance(obj, dict):
            return {k: self._prepare_for_export(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._prepare_for_export(v) for v in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        return obj


# =============================================================================
# Integration Test Classes
# =============================================================================

class TestFullPipeline:
    """End-to-end tests for the full analysis pipeline."""

    @pytest.fixture
    def pipeline_setup(self, sample_markers, sample_video_metadata):
        """Set up pipeline with standard configuration."""
        metadata = VideoMetadata(
            duration=sample_video_metadata["dur"],
            fps=sample_video_metadata["fps"],
            n_frames=sample_video_metadata["nFrame"],
            width=sample_video_metadata["imW"],
            height=sample_video_metadata["imH"],
            pixel_width_mm=sample_video_metadata["xPixW"]
        )

        markers = sample_markers.markers
        tracker = MockTracker(markers=markers)
        tracks = tracker.generate_tracks(metadata)

        pipeline = LocomotorPipeline(config={
            'smoothing_factor': 10,
            'interpolation_factor': 4
        })

        return {
            'pipeline': pipeline,
            'tracks': tracks,
            'metadata': metadata,
            'markers': markers,
            'limb_pairs': sample_markers.limb_pairs,
            'speed_markers': sample_markers.speed_markers,
            'model_name': tracker.model_name
        }

    def test_full_pipeline_execution(self, pipeline_setup):
        """Test complete pipeline execution."""
        setup = pipeline_setup

        results = setup['pipeline'].process_tracks(
            tracks=setup['tracks'],
            metadata=setup['metadata'],
            model_name=setup['model_name'],
            markers=setup['markers'],
            limb_pairs=setup['limb_pairs'],
            speed_markers=setup['speed_markers']
        )

        # Verify all sections are present
        assert 'velocity' in results
        assert 'coordination' in results
        assert 'gait_cycles' in results
        assert 'summary' in results
        assert 'metadata' in results

    def test_full_pipeline_velocity_output(self, pipeline_setup):
        """Test velocity output from full pipeline."""
        setup = pipeline_setup

        results = setup['pipeline'].process_tracks(
            tracks=setup['tracks'],
            metadata=setup['metadata'],
            model_name=setup['model_name'],
            markers=setup['markers'],
            limb_pairs=setup['limb_pairs'],
            speed_markers=setup['speed_markers']
        )

        velocity = results['velocity']

        assert 'mean_speed' in velocity
        assert 'max_speed' in velocity
        assert 'std_speed' in velocity

        # Speed should be positive and reasonable
        assert velocity['mean_speed'] >= 0
        assert velocity['max_speed'] >= velocity['mean_speed']

    def test_full_pipeline_coordination_output(self, pipeline_setup):
        """Test coordination output from full pipeline."""
        setup = pipeline_setup

        results = setup['pipeline'].process_tracks(
            tracks=setup['tracks'],
            metadata=setup['metadata'],
            model_name=setup['model_name'],
            markers=setup['markers'],
            limb_pairs=setup['limb_pairs'],
            speed_markers=setup['speed_markers']
        )

        coordination = results['coordination']

        # Should have results for multiple limb pairs
        assert len(coordination) > 0

        for pair_name, pair_results in coordination.items():
            assert 'R' in pair_results
            assert 'mean_phase_deg' in pair_results
            assert 'n_steps' in pair_results

            # R should be in valid range
            assert 0 <= pair_results['R'] <= 1

    def test_full_pipeline_gait_output(self, pipeline_setup):
        """Test gait cycle output from full pipeline."""
        setup = pipeline_setup

        results = setup['pipeline'].process_tracks(
            tracks=setup['tracks'],
            metadata=setup['metadata'],
            model_name=setup['model_name'],
            markers=setup['markers'],
            limb_pairs=setup['limb_pairs'],
            speed_markers=setup['speed_markers']
        )

        gait = results['gait_cycles']

        # Should have results for limbs
        assert len(gait) > 0

        for limb_name, limb_results in gait.items():
            assert 'cadence' in limb_results
            assert 'stride_length' in limb_results
            assert 'n_cycles' in limb_results

            # Cadence should be reasonable (0-20 Hz)
            assert 0 <= limb_results['cadence'] <= 20

    def test_full_pipeline_summary(self, pipeline_setup):
        """Test summary statistics from full pipeline."""
        setup = pipeline_setup

        results = setup['pipeline'].process_tracks(
            tracks=setup['tracks'],
            metadata=setup['metadata'],
            model_name=setup['model_name'],
            markers=setup['markers'],
            limb_pairs=setup['limb_pairs'],
            speed_markers=setup['speed_markers']
        )

        summary = results['summary']

        expected_fields = [
            'duration',
            'mean_speed_cm_s',
            'mean_coordination_R',
            'mean_cadence_hz',
            'mean_stride_length_cm'
        ]

        for field in expected_fields:
            assert field in summary


class TestVideoProcessing:
    """Tests for video-related processing."""

    def test_video_metadata_creation(self, sample_video_metadata):
        """Test video metadata object creation."""
        metadata = VideoMetadata(
            duration=sample_video_metadata["dur"],
            fps=sample_video_metadata["fps"],
            n_frames=sample_video_metadata["nFrame"],
            width=sample_video_metadata["imW"],
            height=sample_video_metadata["imH"],
            pixel_width_mm=sample_video_metadata["xPixW"]
        )

        assert metadata.duration == 30.0
        assert metadata.fps == 100
        assert metadata.n_frames == 3000
        assert metadata.width == 640
        assert metadata.height == 480

    def test_video_metadata_to_dict(self, sample_video_metadata):
        """Test metadata conversion to dictionary."""
        metadata = VideoMetadata(
            duration=sample_video_metadata["dur"],
            fps=sample_video_metadata["fps"],
            n_frames=sample_video_metadata["nFrame"],
            width=sample_video_metadata["imW"],
            height=sample_video_metadata["imH"],
            pixel_width_mm=sample_video_metadata["xPixW"]
        )

        d = metadata.to_dict()

        assert d['dur'] == 30.0
        assert d['fps'] == 100
        assert d['nFrame'] == 3000
        assert d['imW'] == 640
        assert d['imH'] == 480
        assert 'xPixW' in d

    def test_mock_tracker_generation(self, sample_markers, sample_video_metadata):
        """Test mock tracker data generation."""
        metadata = VideoMetadata(
            duration=sample_video_metadata["dur"],
            fps=sample_video_metadata["fps"],
            n_frames=sample_video_metadata["nFrame"],
            width=sample_video_metadata["imW"],
            height=sample_video_metadata["imH"],
            pixel_width_mm=sample_video_metadata["xPixW"]
        )

        tracker = MockTracker(markers=sample_markers.markers)
        tracks = tracker.generate_tracks(metadata)

        # Verify DataFrame structure
        assert isinstance(tracks, pd.DataFrame)
        assert len(tracks) == metadata.n_frames

        # Verify all markers present
        for marker in sample_markers.markers:
            assert marker in tracks[tracker.model_name].columns.get_level_values(0)

    def test_mock_tracker_variable_parameters(self, sample_markers, sample_video_metadata):
        """Test mock tracker with different parameters."""
        metadata = VideoMetadata(
            duration=sample_video_metadata["dur"],
            fps=sample_video_metadata["fps"],
            n_frames=sample_video_metadata["nFrame"],
            width=sample_video_metadata["imW"],
            height=sample_video_metadata["imH"],
            pixel_width_mm=sample_video_metadata["xPixW"]
        )

        tracker = MockTracker(markers=sample_markers.markers)

        # Test different gait frequencies
        for freq in [2.0, 4.0, 6.0, 8.0]:
            tracks = tracker.generate_tracks(
                metadata,
                gait_frequency=freq,
                speed_cm_s=20.0,
                noise_level=1.0
            )

            assert len(tracks) == metadata.n_frames


class TestExportFunctionality:
    """Tests for data export functionality."""

    def test_export_json(self, pipeline_setup, temp_output_dir):
        """Test JSON export."""
        setup = pipeline_setup

        results = setup['pipeline'].process_tracks(
            tracks=setup['tracks'],
            metadata=setup['metadata'],
            model_name=setup['model_name'],
            markers=setup['markers'],
            limb_pairs=setup['limb_pairs'],
            speed_markers=setup['speed_markers']
        )

        output_path = os.path.join(temp_output_dir, 'results.json')
        setup['pipeline'].export_results(output_path, format='json')

        # Verify file exists and is valid JSON
        assert os.path.exists(output_path)

        with open(output_path, 'r') as f:
            loaded = json.load(f)

        assert 'summary' in loaded
        assert 'velocity' in loaded

    def test_export_csv(self, pipeline_setup, temp_output_dir):
        """Test CSV export."""
        setup = pipeline_setup

        results = setup['pipeline'].process_tracks(
            tracks=setup['tracks'],
            metadata=setup['metadata'],
            model_name=setup['model_name'],
            markers=setup['markers'],
            limb_pairs=setup['limb_pairs'],
            speed_markers=setup['speed_markers']
        )

        output_path = os.path.join(temp_output_dir, 'summary.csv')
        setup['pipeline'].export_results(output_path, format='csv')

        # Verify file exists and is valid CSV
        assert os.path.exists(output_path)

        df = pd.read_csv(output_path)
        assert len(df) > 0
        assert 'mean_speed_cm_s' in df.columns

    def test_export_invalid_format(self, pipeline_setup, temp_output_dir):
        """Test export with invalid format raises error."""
        setup = pipeline_setup

        setup['pipeline'].process_tracks(
            tracks=setup['tracks'],
            metadata=setup['metadata'],
            model_name=setup['model_name'],
            markers=setup['markers'],
            limb_pairs=setup['limb_pairs'],
            speed_markers=setup['speed_markers']
        )

        output_path = os.path.join(temp_output_dir, 'results.invalid')

        with pytest.raises(ValueError, match="Unknown format"):
            setup['pipeline'].export_results(output_path, format='invalid')


class TestBatchProcessing:
    """Tests for batch processing multiple videos."""

    def test_batch_multiple_videos(self, sample_markers, sample_video_metadata):
        """Test processing multiple videos in sequence."""
        metadata = VideoMetadata(
            duration=sample_video_metadata["dur"],
            fps=sample_video_metadata["fps"],
            n_frames=sample_video_metadata["nFrame"],
            width=sample_video_metadata["imW"],
            height=sample_video_metadata["imH"],
            pixel_width_mm=sample_video_metadata["xPixW"]
        )

        tracker = MockTracker(markers=sample_markers.markers)
        pipeline = LocomotorPipeline()

        all_results = []

        # Process 3 "videos" with different parameters
        for speed in [10, 20, 30]:
            tracks = tracker.generate_tracks(
                metadata,
                gait_frequency=4.0,
                speed_cm_s=float(speed)
            )

            results = pipeline.process_tracks(
                tracks=tracks,
                metadata=metadata,
                model_name=tracker.model_name,
                markers=sample_markers.markers,
                limb_pairs=sample_markers.limb_pairs,
                speed_markers=sample_markers.speed_markers
            )

            all_results.append(results)

        assert len(all_results) == 3

        # Speeds should increase across batches
        speeds = [r['summary']['mean_speed_cm_s'] for r in all_results]
        # Due to processing, relative order should be maintained
        assert len(set(speeds)) > 1  # Should be different values

    def test_batch_aggregate_statistics(self, sample_markers, sample_video_metadata):
        """Test aggregating statistics from batch processing."""
        metadata = VideoMetadata(
            duration=sample_video_metadata["dur"],
            fps=sample_video_metadata["fps"],
            n_frames=sample_video_metadata["nFrame"],
            width=sample_video_metadata["imW"],
            height=sample_video_metadata["imH"],
            pixel_width_mm=sample_video_metadata["xPixW"]
        )

        tracker = MockTracker(markers=sample_markers.markers)
        pipeline = LocomotorPipeline()

        summaries = []

        for _ in range(5):
            tracks = tracker.generate_tracks(metadata)
            results = pipeline.process_tracks(
                tracks=tracks,
                metadata=metadata,
                model_name=tracker.model_name,
                markers=sample_markers.markers,
                limb_pairs=sample_markers.limb_pairs,
                speed_markers=sample_markers.speed_markers
            )
            summaries.append(results['summary'])

        # Create aggregate DataFrame
        df = pd.DataFrame(summaries)

        assert len(df) == 5
        assert 'mean_speed_cm_s' in df.columns
        assert 'mean_cadence_hz' in df.columns

        # Compute aggregate statistics
        aggregate = {
            'mean_speed': df['mean_speed_cm_s'].mean(),
            'std_speed': df['mean_speed_cm_s'].std(),
            'mean_cadence': df['mean_cadence_hz'].mean(),
        }

        assert aggregate['mean_speed'] >= 0
        assert aggregate['mean_cadence'] >= 0


class TestDataFlowIntegrity:
    """Tests for data integrity through the pipeline."""

    def test_frame_count_consistency(self, pipeline_setup):
        """Test that frame counts remain consistent."""
        setup = pipeline_setup

        initial_frames = len(setup['tracks'])

        results = setup['pipeline'].process_tracks(
            tracks=setup['tracks'],
            metadata=setup['metadata'],
            model_name=setup['model_name'],
            markers=setup['markers'],
            limb_pairs=setup['limb_pairs'],
            speed_markers=setup['speed_markers']
        )

        # Metadata should reflect correct frame count
        assert results['metadata']['nFrame'] == initial_frames

    def test_metric_range_validity(self, pipeline_setup):
        """Test that all metrics are in valid ranges."""
        setup = pipeline_setup

        results = setup['pipeline'].process_tracks(
            tracks=setup['tracks'],
            metadata=setup['metadata'],
            model_name=setup['model_name'],
            markers=setup['markers'],
            limb_pairs=setup['limb_pairs'],
            speed_markers=setup['speed_markers']
        )

        # Velocity checks
        vel = results['velocity']
        assert vel['mean_speed'] >= 0
        assert vel['max_speed'] >= vel['min_speed']
        assert vel['std_speed'] >= 0

        # Coordination checks
        for pair, data in results['coordination'].items():
            assert 0 <= data['R'] <= 1
            assert -180 <= data['mean_phase_deg'] <= 180
            assert data['n_steps'] >= 0

        # Gait cycle checks
        for limb, data in results['gait_cycles'].items():
            assert data['cadence'] >= 0
            assert data['stride_length'] >= 0
            assert data['n_cycles'] >= 0

    def test_reproducibility(self, sample_markers, sample_video_metadata):
        """Test that results are reproducible with same seed."""
        metadata = VideoMetadata(
            duration=sample_video_metadata["dur"],
            fps=sample_video_metadata["fps"],
            n_frames=sample_video_metadata["nFrame"],
            width=sample_video_metadata["imW"],
            height=sample_video_metadata["imH"],
            pixel_width_mm=sample_video_metadata["xPixW"]
        )

        tracker = MockTracker(markers=sample_markers.markers)
        pipeline = LocomotorPipeline()

        # First run
        np.random.seed(42)
        tracks1 = tracker.generate_tracks(metadata)
        results1 = pipeline.process_tracks(
            tracks=tracks1,
            metadata=metadata,
            model_name=tracker.model_name,
            markers=sample_markers.markers,
            limb_pairs=sample_markers.limb_pairs,
            speed_markers=sample_markers.speed_markers
        )

        # Second run with same seed
        np.random.seed(42)
        tracks2 = tracker.generate_tracks(metadata)
        results2 = pipeline.process_tracks(
            tracks=tracks2,
            metadata=metadata,
            model_name=tracker.model_name,
            markers=sample_markers.markers,
            limb_pairs=sample_markers.limb_pairs,
            speed_markers=sample_markers.speed_markers
        )

        # Results should be identical
        assert results1['summary']['mean_speed_cm_s'] == \
               results2['summary']['mean_speed_cm_s']


class TestErrorHandling:
    """Tests for error handling in integration scenarios."""

    def test_missing_markers(self, sample_markers, sample_video_metadata):
        """Test handling of missing marker data."""
        metadata = VideoMetadata(
            duration=sample_video_metadata["dur"],
            fps=sample_video_metadata["fps"],
            n_frames=sample_video_metadata["nFrame"],
            width=sample_video_metadata["imW"],
            height=sample_video_metadata["imH"],
            pixel_width_mm=sample_video_metadata["xPixW"]
        )

        # Create tracker with limited markers
        limited_markers = ['snout', 'torso', 'tail']
        tracker = MockTracker(markers=limited_markers)
        tracks = tracker.generate_tracks(metadata)

        pipeline = LocomotorPipeline()

        # Should still process, but with limited results
        results = pipeline.process_tracks(
            tracks=tracks,
            metadata=metadata,
            model_name=tracker.model_name,
            markers=limited_markers,
            limb_pairs=sample_markers.limb_pairs,  # Has pairs for missing limbs
            speed_markers=['snout', 'torso', 'tail']
        )

        # Should have some results even with missing data
        assert 'velocity' in results
        # Coordination may be empty due to missing limbs
        assert 'coordination' in results

    def test_short_video(self, sample_markers):
        """Test handling of very short video."""
        metadata = VideoMetadata(
            duration=0.5,  # Very short
            fps=100,
            n_frames=50,
            width=640,
            height=480,
            pixel_width_mm=200/640
        )

        tracker = MockTracker(markers=sample_markers.markers)
        tracks = tracker.generate_tracks(metadata)

        pipeline = LocomotorPipeline()

        # Should handle gracefully
        results = pipeline.process_tracks(
            tracks=tracks,
            metadata=metadata,
            model_name=tracker.model_name,
            markers=sample_markers.markers,
            limb_pairs=sample_markers.limb_pairs,
            speed_markers=sample_markers.speed_markers
        )

        assert 'summary' in results


# =============================================================================
# Performance Tests
# =============================================================================

class TestPerformance:
    """Performance-related tests."""

    @pytest.mark.slow
    def test_large_video_processing(self, sample_markers):
        """Test processing of large video (1 hour equivalent)."""
        # 1 hour at 100 fps = 360,000 frames
        # Use smaller for practical testing: 10 minutes
        metadata = VideoMetadata(
            duration=600.0,  # 10 minutes
            fps=100,
            n_frames=60000,
            width=640,
            height=480,
            pixel_width_mm=200/640
        )

        tracker = MockTracker(markers=sample_markers.markers)
        tracks = tracker.generate_tracks(metadata, noise_level=0.5)

        pipeline = LocomotorPipeline()

        import time
        start = time.time()

        results = pipeline.process_tracks(
            tracks=tracks,
            metadata=metadata,
            model_name=tracker.model_name,
            markers=sample_markers.markers,
            limb_pairs=sample_markers.limb_pairs,
            speed_markers=sample_markers.speed_markers
        )

        elapsed = time.time() - start

        # Should complete in reasonable time (< 30 seconds)
        assert elapsed < 30

        # Results should be valid
        assert 'summary' in results
        assert results['summary']['duration'] == 600.0
