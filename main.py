#!/usr/bin/env python3
"""
Mouse Locomotor Tracker - TOP 0.1% Analysis System
==================================================

A comprehensive locomotor analysis system combining the best of:
- EstimAI_ (DeepLabCut integration)
- Locomotor-Allodi2021 (Scientific analysis methodology)

Features:
- DeepLabCut-based marker tracking
- Velocity and acceleration analysis
- Circular statistics for limb coordination
- Joint angle kinematics
- Hollywood-style motion capture visualization

Author: Stride Labs (gvillegas@stridelabs.cl)
License: MIT
Version: 1.0.0
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional
import yaml
import warnings

warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# ASCII Art Banner
BANNER = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║   ███╗   ███╗ ██████╗ ██╗   ██╗███████╗███████╗                              ║
║   ████╗ ████║██╔═══██╗██║   ██║██╔════╝██╔════╝                              ║
║   ██╔████╔██║██║   ██║██║   ██║███████╗█████╗                                ║
║   ██║╚██╔╝██║██║   ██║██║   ██║╚════██║██╔══╝                                ║
║   ██║ ╚═╝ ██║╚██████╔╝╚██████╔╝███████║███████╗                              ║
║   ╚═╝     ╚═╝ ╚═════╝  ╚═════╝ ╚══════╝╚══════╝                              ║
║                                                                              ║
║   ██╗      ██████╗  ██████╗ ██████╗ ███╗   ███╗ ██████╗ ████████╗ ██████╗ ██████╗  ║
║   ██║     ██╔═══██╗██╔════╝██╔═══██╗████╗ ████║██╔═══██╗╚══██╔══╝██╔═══██╗██╔══██╗ ║
║   ██║     ██║   ██║██║     ██║   ██║██╔████╔██║██║   ██║   ██║   ██║   ██║██████╔╝ ║
║   ██║     ██║   ██║██║     ██║   ██║██║╚██╔╝██║██║   ██║   ██║   ██║   ██║██╔══██╗ ║
║   ███████╗╚██████╔╝╚██████╗╚██████╔╝██║ ╚═╝ ██║╚██████╔╝   ██║   ╚██████╔╝██║  ██║ ║
║   ╚══════╝ ╚═════╝  ╚═════╝ ╚═════╝ ╚═╝     ╚═╝ ╚═════╝    ╚═╝    ╚═════╝ ╚═╝  ╚═╝ ║
║                                                                              ║
║   ████████╗██████╗  █████╗  ██████╗██╗  ██╗███████╗██████╗                   ║
║   ╚══██╔══╝██╔══██╗██╔══██╗██╔════╝██║ ██╔╝██╔════╝██╔══██╗                  ║
║      ██║   ██████╔╝███████║██║     █████╔╝ █████╗  ██████╔╝                  ║
║      ██║   ██╔══██╗██╔══██║██║     ██╔═██╗ ██╔══╝  ██╔══██╗                  ║
║      ██║   ██║  ██║██║  ██║╚██████╗██║  ██╗███████╗██║  ██║                  ║
║      ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝                  ║
║                                                                              ║
║                    TOP 0.1% Locomotor Analysis System                        ║
║                         Stride Labs - 2026                                   ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def setup_output_directory(output_dir: Path) -> dict:
    """Create output directory structure."""
    subdirs = {
        'tracks': output_dir / 'tracks',
        'analysis': output_dir / 'analysis',
        'plots': output_dir / 'plots',
        'videos': output_dir / 'videos',
        'reports': output_dir / 'reports'
    }

    for name, path in subdirs.items():
        path.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Created directory: {path}")

    return subdirs


def run_tracking(
    video_path: Path,
    config: dict,
    output_dirs: dict,
    use_pretrained: bool = True
) -> 'pd.DataFrame':
    """
    Step 1: Run DeepLabCut tracking on video.
    """
    logger.info("=" * 60)
    logger.info("STEP 1: MARKER TRACKING (DeepLabCut)")
    logger.info("=" * 60)

    try:
        from tracking.dlc_adapter import DeepLabCutAdapter
        from tracking.track_processor import TrackProcessor
        from tracking.video_metadata import VideoMetadata
    except ImportError:
        logger.warning("Tracking module not fully available. Using mock data.")
        from tracking.mock_tracker import MockTracker
        mock = MockTracker()
        return mock.generate_tracks(duration=10.0, fps=30)

    # Get video metadata
    metadata = VideoMetadata.from_video(video_path)
    logger.info(f"Video: {video_path.name}")
    logger.info(f"Duration: {metadata.duration:.2f}s | FPS: {metadata.fps} | Resolution: {metadata.width}x{metadata.height}")

    # Initialize tracker
    tracker = DeepLabCutAdapter(
        model_name="superanimal-topviewmouse" if use_pretrained else "custom",
        confidence_threshold=config['tracking']['confidence_threshold']
    )

    # Analyze video
    logger.info("Analyzing video with DeepLabCut...")
    raw_tracks = tracker.analyze_video(video_path)

    # Process tracks
    processor = TrackProcessor(
        confidence_threshold=config['tracking']['confidence_threshold'],
        smoothing_config=config['tracking']['smoothing']
    )
    tracks = processor.process(raw_tracks)

    # Save tracks
    tracks_file = output_dirs['tracks'] / f"{video_path.stem}_tracks.csv"
    tracks.to_csv(tracks_file)
    logger.info(f"Tracks saved to: {tracks_file}")

    return tracks, metadata


def run_analysis(
    tracks: 'pd.DataFrame',
    metadata: 'VideoMetadata',
    config: dict,
    output_dirs: dict
) -> dict:
    """
    Step 2: Run biomechanical analysis.
    """
    logger.info("")
    logger.info("=" * 60)
    logger.info("STEP 2: BIOMECHANICAL ANALYSIS")
    logger.info("=" * 60)

    try:
        from analysis.velocity import VelocityAnalyzer
        from analysis.coordination import CircularCoordinationAnalyzer
        from analysis.gait_cycles import GaitCycleDetector
        from analysis.metrics import LocomotorReport
    except ImportError as e:
        logger.error(f"Analysis module import error: {e}")
        return {}

    results = {}

    # 2.1 Velocity Analysis
    logger.info("\n--- 2.1 Velocity & Acceleration Analysis ---")
    velocity_analyzer = VelocityAnalyzer(
        fps=metadata.fps,
        pixel_to_mm=config['video'].get('pixel_to_mm', 0.5),
        smoothing_factor=config['velocity']['smoothing_factor'],
        speed_smoothing_factor=config['velocity']['speed_smoothing_factor']
    )
    velocity_metrics = velocity_analyzer.analyze(tracks)
    results['velocity'] = velocity_metrics
    logger.info(f"  Average Speed: {velocity_metrics.avg_speed:.2f} cm/s")
    logger.info(f"  Peak Acceleration: {velocity_metrics.peak_acceleration:.2f} cm/s²")

    # 2.2 Gait Cycle Detection
    logger.info("\n--- 2.2 Gait Cycle Detection ---")
    gait_detector = GaitCycleDetector(
        interpolation_factor=config['gait_cycle']['interpolation_factor'],
        speed_threshold=config['gait_cycle']['speed_threshold']
    )
    gait_metrics = gait_detector.detect(tracks, velocity_metrics.speed_profile)
    results['gait'] = gait_metrics
    logger.info(f"  Number of Steps: {gait_metrics.num_steps}")
    logger.info(f"  Movement Duration: {gait_metrics.movement_duration:.2f}s")
    logger.info(f"  Cadence (LH): {gait_metrics.cadence['LH']:.2f} Hz")
    logger.info(f"  Cadence (RH): {gait_metrics.cadence['RH']:.2f} Hz")

    # 2.3 Limb Coordination (Circular Statistics)
    logger.info("\n--- 2.3 Limb Coordination (Circular Statistics) ---")
    coordination_analyzer = CircularCoordinationAnalyzer()
    coordination_metrics = coordination_analyzer.analyze_all_pairs(
        tracks,
        gait_metrics.strides,
        gait_metrics.movement_duration,
        pairs=config['coordination']['pairs']
    )
    results['coordination'] = coordination_metrics

    for pair_name, metrics in coordination_metrics.items():
        logger.info(f"  {pair_name}: Phase={metrics.mean_phase:.1f}°, R={metrics.r_value:.3f}")

    # 2.4 Generate Report
    logger.info("\n--- 2.4 Generating Report ---")
    report = LocomotorReport(
        velocity=velocity_metrics,
        gait=gait_metrics,
        coordination=coordination_metrics
    )
    results['report'] = report

    return results


def run_visualization(
    video_path: Path,
    tracks: 'pd.DataFrame',
    analysis_results: dict,
    config: dict,
    output_dirs: dict
) -> Path:
    """
    Step 3: Generate visualizations.
    """
    logger.info("")
    logger.info("=" * 60)
    logger.info("STEP 3: VISUALIZATION (Hollywood Style)")
    logger.info("=" * 60)

    try:
        from visualization.trajectory_overlay import TrajectoryVisualizer
        from visualization.circular_plots import CoordinationPlotter
        from visualization.speed_plots import SpeedProfilePlotter
        from visualization.dashboard import RealtimeDashboard
        from visualization.video_generator import VideoGenerator
    except ImportError as e:
        logger.error(f"Visualization module import error: {e}")
        return None

    vis_config = config['visualization']

    # 3.1 Generate Circular Coordination Plots
    logger.info("\n--- 3.1 Circular Coordination Plots ---")
    coord_plotter = CoordinationPlotter()
    coord_plot_path = output_dirs['plots'] / f"{video_path.stem}_coordination.pdf"
    coord_plotter.plot_all_pairs(
        analysis_results['coordination'],
        output_path=coord_plot_path
    )
    logger.info(f"  Saved: {coord_plot_path}")

    # 3.2 Generate Speed Profile Plot
    logger.info("\n--- 3.2 Speed Profile Plot ---")
    speed_plotter = SpeedProfilePlotter()
    speed_plot_path = output_dirs['plots'] / f"{video_path.stem}_speed_profile.pdf"
    speed_plotter.plot_profile(
        analysis_results['velocity'],
        output_path=speed_plot_path
    )
    logger.info(f"  Saved: {speed_plot_path}")

    # 3.3 Generate Tracked Video with Overlays
    logger.info("\n--- 3.3 Generating Tracked Video ---")

    trajectory_viz = TrajectoryVisualizer(
        trail_length=vis_config['trajectory']['trail_length'],
        trail_color_gradient=vis_config['trajectory']['color_gradient'],
        glow_effect=vis_config['trajectory']['glow_effect'],
        marker_size=vis_config['trajectory']['marker_size']
    )

    dashboard = RealtimeDashboard(
        enabled=vis_config['dashboard']['enabled'],
        position=vis_config['dashboard']['position'],
        opacity=vis_config['dashboard']['opacity']
    )

    video_generator = VideoGenerator(
        trajectory_visualizer=trajectory_viz,
        dashboard=dashboard
    )

    output_video_path = output_dirs['videos'] / f"{video_path.stem}_tracked.mp4"
    video_generator.process_video(
        input_path=video_path,
        tracks=tracks,
        analysis_results=analysis_results,
        output_path=output_video_path
    )
    logger.info(f"  Saved: {output_video_path}")

    return output_video_path


def export_results(
    analysis_results: dict,
    output_dirs: dict,
    video_name: str,
    formats: list = ['csv', 'json']
) -> None:
    """
    Step 4: Export analysis results.
    """
    logger.info("")
    logger.info("=" * 60)
    logger.info("STEP 4: EXPORT RESULTS")
    logger.info("=" * 60)

    try:
        from export.csv_exporter import CSVExporter
        from export.json_exporter import JSONExporter
        from export.report_generator import ReportGenerator
    except ImportError as e:
        logger.error(f"Export module import error: {e}")
        return

    # Export to CSV
    if 'csv' in formats:
        csv_exporter = CSVExporter()
        csv_path = output_dirs['analysis'] / f"{video_name}_statistics.csv"
        csv_exporter.export(analysis_results, csv_path)
        logger.info(f"  CSV: {csv_path}")

    # Export to JSON
    if 'json' in formats:
        json_exporter = JSONExporter()
        json_path = output_dirs['analysis'] / f"{video_name}_statistics.json"
        json_exporter.export(analysis_results, json_path)
        logger.info(f"  JSON: {json_path}")

    # Generate PDF Report
    report_gen = ReportGenerator()
    report_path = output_dirs['reports'] / f"{video_name}_report.pdf"
    report_gen.generate(analysis_results, output_dirs['plots'], report_path)
    logger.info(f"  Report: {report_path}")


def print_summary(analysis_results: dict, output_video: Optional[Path]) -> None:
    """Print final summary."""
    logger.info("")
    logger.info("=" * 60)
    logger.info("ANALYSIS COMPLETE!")
    logger.info("=" * 60)

    if 'velocity' in analysis_results:
        v = analysis_results['velocity']
        logger.info(f"\n📊 VELOCITY METRICS:")
        logger.info(f"   Average Speed: {v.avg_speed:.2f} cm/s")
        logger.info(f"   Peak Speed: {v.peak_speed:.2f} cm/s")
        logger.info(f"   Peak Acceleration: {v.peak_acceleration:.2f} cm/s²")

    if 'gait' in analysis_results:
        g = analysis_results['gait']
        logger.info(f"\n🦿 GAIT METRICS:")
        logger.info(f"   Total Steps: {g.num_steps}")
        logger.info(f"   Movement Duration: {g.movement_duration:.2f}s")
        logger.info(f"   LH Stride Length: {g.stride_length.get('LH', 0):.2f} cm")
        logger.info(f"   RH Stride Length: {g.stride_length.get('RH', 0):.2f} cm")

    if 'coordination' in analysis_results:
        logger.info(f"\n🔄 COORDINATION METRICS:")
        for pair, metrics in analysis_results['coordination'].items():
            status = "✅" if metrics.r_value > 0.7 else "⚠️"
            logger.info(f"   {status} {pair}: R={metrics.r_value:.3f}, Phase={metrics.mean_phase:.1f}°")

    if output_video:
        logger.info(f"\n🎬 OUTPUT VIDEO: {output_video}")

    logger.info("\n" + "=" * 60)


def main():
    """Main entry point."""
    print(BANNER)

    parser = argparse.ArgumentParser(
        description='Mouse Locomotor Tracker - TOP 0.1% Analysis System',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '--video', '-v',
        type=str,
        required=True,
        help='Path to input video file'
    )

    parser.add_argument(
        '--config', '-c',
        type=str,
        default='config/analysis_params.yaml',
        help='Path to analysis configuration file'
    )

    parser.add_argument(
        '--output', '-o',
        type=str,
        default='output',
        help='Output directory for results'
    )

    parser.add_argument(
        '--view',
        choices=['ventral', 'lateral', 'both'],
        default='ventral',
        help='Camera view type'
    )

    parser.add_argument(
        '--skip-tracking',
        action='store_true',
        help='Skip tracking step (use existing tracks)'
    )

    parser.add_argument(
        '--skip-visualization',
        action='store_true',
        help='Skip video visualization generation'
    )

    parser.add_argument(
        '--use-mock',
        action='store_true',
        help='Use mock tracker for testing without DeepLabCut'
    )

    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug logging'
    )

    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    # Validate inputs
    video_path = Path(args.video)
    if not video_path.exists():
        logger.error(f"Video file not found: {video_path}")
        sys.exit(1)

    config_path = Path(args.config)
    if not config_path.exists():
        logger.warning(f"Config file not found: {config_path}. Using defaults.")
        config = {}
    else:
        config = load_config(config_path)

    # Setup output directory
    output_dir = Path(args.output) / video_path.stem
    output_dirs = setup_output_directory(output_dir)

    logger.info(f"Processing: {video_path.name}")
    logger.info(f"Output: {output_dir}")
    logger.info("")

    try:
        # Step 1: Tracking
        if args.use_mock:
            from tracking.mock_tracker import MockTracker
            from tracking.video_metadata import VideoMetadata
            mock = MockTracker()
            tracks = mock.generate_tracks(duration=10.0, fps=30)
            metadata = VideoMetadata(fps=30, duration=10.0, width=640, height=480)
        elif args.skip_tracking:
            import pandas as pd
            tracks_file = output_dirs['tracks'] / f"{video_path.stem}_tracks.csv"
            if not tracks_file.exists():
                logger.error(f"Tracks file not found: {tracks_file}")
                sys.exit(1)
            tracks = pd.read_csv(tracks_file)
            from tracking.video_metadata import VideoMetadata
            metadata = VideoMetadata.from_video(video_path)
        else:
            tracks, metadata = run_tracking(video_path, config, output_dirs)

        # Step 2: Analysis
        analysis_results = run_analysis(tracks, metadata, config, output_dirs)

        # Step 3: Visualization
        output_video = None
        if not args.skip_visualization:
            output_video = run_visualization(
                video_path, tracks, analysis_results, config, output_dirs
            )

        # Step 4: Export
        export_results(
            analysis_results,
            output_dirs,
            video_path.stem,
            formats=config.get('export', {}).get('formats', ['csv', 'json'])
        )

        # Summary
        print_summary(analysis_results, output_video)

    except Exception as e:
        logger.error(f"Error during processing: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)

    logger.info("Done! 🎉")
    return 0


if __name__ == '__main__':
    sys.exit(main())
