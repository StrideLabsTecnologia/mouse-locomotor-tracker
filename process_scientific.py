#!/usr/bin/env python3
"""
Scientific Mouse Locomotor Tracker
====================================

Complete scientific analysis pipeline that generates:
1. Tracked video with professional HUD
2. Scientific metrics (CatWalk/DigiGait standards)
3. Publication-ready figures
4. Scientific reports (LaTeX, Markdown, JSON)
5. Data exports (CSV, JSON, HDF5, NWB)

For academic research, thesis work, and publications.

Author: Stride Labs
Version: 2.0.0
"""

import sys
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Tuple, Dict
from collections import deque

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from analysis.scientific_metrics import ScientificGaitAnalyzer, ScientificGaitMetrics
from visualization.publication_figures import PublicationFigureGenerator
from export.scientific_report import ScientificReportGenerator, ReportMetadata, generate_scientific_reports


# =============================================================================
# Motion Tracker (from process_professional.py)
# =============================================================================
class MotionTracker:
    """Motion-based mouse tracking."""

    def __init__(self, frame_width: int, frame_height: int):
        self.roi_y1 = int(frame_height * 0.40)
        self.roi_y2 = int(frame_height * 0.78)
        self.frame_buffer: List[np.ndarray] = []
        self.buffer_size = 3
        self.positions: deque = deque(maxlen=100)
        self.prev_center = None

    def track(self, frame: np.ndarray) -> Optional[dict]:
        """Track mouse using motion detection."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        roi = gray[self.roi_y1:self.roi_y2, :]

        self.frame_buffer.append(roi)
        if len(self.frame_buffer) > self.buffer_size:
            self.frame_buffer.pop(0)

        if len(self.frame_buffer) < 2:
            return None

        motion = np.zeros_like(roi, dtype=np.float32)
        for i in range(1, len(self.frame_buffer)):
            diff = cv2.absdiff(self.frame_buffer[i], self.frame_buffer[i-1])
            motion += diff.astype(np.float32)

        motion = (motion / (len(self.frame_buffer) - 1)).astype(np.uint8)
        _, mask = cv2.threshold(motion, 10, 255, cv2.THRESH_BINARY)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.dilate(mask, kernel, iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None

        best = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(best)

        if area < 500:
            return None

        M = cv2.moments(best)
        if M['m00'] == 0:
            return None

        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00']) + self.roi_y1

        if self.prev_center:
            alpha = 0.6
            cx = int(self.prev_center[0] * (1-alpha) + cx * alpha)
            cy = int(self.prev_center[1] * (1-alpha) + cy * alpha)

        self.prev_center = (cx, cy)
        self.positions.append((cx, cy))

        cnt_full = best.copy()
        cnt_full[:, :, 1] += self.roi_y1

        return {'center': (cx, cy), 'contour': cnt_full, 'area': area}


# =============================================================================
# Scientific Processor
# =============================================================================
class ScientificProcessor:
    """
    Complete scientific analysis pipeline.

    Generates:
    - Tracked video with HUD
    - Scientific metrics
    - Publication figures
    - Reports (LaTeX, Markdown)
    - Data exports
    """

    def __init__(
        self,
        experiment_id: str = "EXP001",
        subject_id: str = "MOUSE001",
        experimenter: str = "Unknown",
        institution: str = "Stride Labs",
        journal: str = "nature",
    ):
        self.experiment_id = experiment_id
        self.subject_id = subject_id
        self.experimenter = experimenter
        self.institution = institution
        self.journal = journal

        # Analysis components
        self.tracker = None
        self.gait_analyzer = None
        self.figure_generator = None
        self.report_generator = ScientificReportGenerator()

        # Data storage
        self.positions: List[Tuple[int, int]] = []
        self.timestamps: List[float] = []

    def process(
        self,
        input_path: str,
        output_dir: str = None,
        generate_video: bool = True,
        generate_figures: bool = True,
        generate_reports: bool = True,
        preview: bool = False,
        max_frames: int = None,
    ) -> Dict:
        """
        Run complete scientific analysis pipeline.

        Args:
            input_path: Input video path
            output_dir: Output directory for all files
            generate_video: Generate tracked video
            generate_figures: Generate publication figures
            generate_reports: Generate scientific reports
            preview: Show preview window
            max_frames: Limit frames for testing
        """
        print("=" * 70)
        print("  SCIENTIFIC MOUSE LOCOMOTOR TRACKER")
        print("  Stride Labs - Publication-Grade Analysis")
        print("=" * 70)

        # Setup paths
        input_path = Path(input_path)
        if output_dir is None:
            output_dir = input_path.parent / f"{input_path.stem}_analysis"
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Open video
        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open: {input_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print(f"\n  Input: {input_path.name}")
        print(f"  Resolution: {width}x{height} @ {fps:.1f} FPS")
        print(f"  Duration: {total_frames/fps:.1f}s ({total_frames} frames)")
        print(f"  Output: {output_dir}")
        print("-" * 70)

        # Initialize components
        self.tracker = MotionTracker(width, height)
        cm_per_pixel = 40.0 / width  # Calibration

        self.gait_analyzer = ScientificGaitAnalyzer(
            fps=fps,
            cm_per_pixel=cm_per_pixel,
        )

        self.figure_generator = PublicationFigureGenerator(
            journal=self.journal,
            output_dir=output_dir,
        )

        # Video writer
        video_out = None
        if generate_video:
            video_path = output_dir / f"{self.experiment_id}_tracked.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_out = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))

        # Process frames
        self.positions = []
        self.timestamps = []
        frame_idx = 0
        tracked_count = 0

        print("\n  Phase 1: Tracking...")

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if max_frames and frame_idx >= max_frames:
                break

            result = self.tracker.track(frame)

            if result:
                tracked_count += 1
                center = result['center']
                self.positions.append(center)
                self.timestamps.append(frame_idx / fps)

                if generate_video:
                    # Draw tracking visualization
                    vis = self._draw_tracking(frame, result, frame_idx, fps)
                    video_out.write(vis)

                    if preview:
                        cv2.imshow('Scientific Tracker', cv2.resize(vis, (1280, 720)))
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
            else:
                if generate_video:
                    video_out.write(frame)

            frame_idx += 1
            if frame_idx % 30 == 0:
                pct = frame_idx / min(total_frames, max_frames or total_frames) * 100
                print(f"\r  Progress: {pct:.1f}%", end="")

        cap.release()
        if video_out:
            video_out.release()
        cv2.destroyAllWindows()

        tracking_rate = tracked_count / frame_idx * 100 if frame_idx > 0 else 0
        print(f"\n  Tracking complete: {tracking_rate:.1f}% ({tracked_count}/{frame_idx} frames)")

        # Phase 2: Scientific Analysis
        print("\n  Phase 2: Scientific Analysis...")

        positions_array = np.array(self.positions)
        timestamps_array = np.array(self.timestamps)

        metrics = self.gait_analyzer.analyze(positions_array, timestamps_array)
        metrics_dict = self.gait_analyzer.to_dict(metrics)

        # Print scientific report to console
        print("\n" + self.gait_analyzer.generate_report(metrics))

        # Phase 3: Generate Figures
        if generate_figures and len(self.positions) > 10:
            print("\n  Phase 3: Generating Publication Figures...")

            # Velocity timeseries
            self.figure_generator.plot_velocity_timeseries(
                metrics.velocity_timeseries,
                timestamps_array[:len(metrics.velocity_timeseries)],
                output=output_dir / f"{self.experiment_id}_velocity.pdf",
                title="Velocity Profile",
            )

            # Trajectory heatmap
            self.figure_generator.plot_trajectory_heatmap(
                positions_array,
                output=output_dir / f"{self.experiment_id}_trajectory.pdf",
                velocity=metrics.velocity_timeseries[:len(positions_array)],
            )

            # Gait summary
            self.figure_generator.plot_gait_summary(
                metrics_dict,
                output=output_dir / f"{self.experiment_id}_summary.pdf",
                title=f"Gait Analysis: {self.subject_id}",
            )

            # Stride analysis
            if metrics.strides:
                self.figure_generator.plot_stride_analysis(
                    metrics.strides,
                    output=output_dir / f"{self.experiment_id}_strides.pdf",
                )

            print("  Figures saved.")

        # Phase 4: Generate Reports
        if generate_reports:
            print("\n  Phase 4: Generating Scientific Reports...")

            metadata = ReportMetadata(
                experiment_id=self.experiment_id,
                subject_id=self.subject_id,
                experimenter=self.experimenter,
                institution=self.institution,
            )

            # LaTeX report
            self.report_generator.generate_latex_report(
                metrics_dict, metadata,
                output_dir / f"{self.experiment_id}_report.tex"
            )

            # Markdown report
            self.report_generator.generate_markdown_report(
                metrics_dict, metadata,
                output_dir / f"{self.experiment_id}_report.md"
            )

            # JSON report
            self.report_generator.generate_json_report(
                metrics_dict, metadata,
                output_dir / f"{self.experiment_id}_report.json"
            )

        # Phase 5: Export Data
        print("\n  Phase 5: Exporting Data...")

        # CSV export
        self._export_csv(output_dir / f"{self.experiment_id}_data.csv", metrics)

        print("\n" + "=" * 70)
        print("  ANALYSIS COMPLETE")
        print("=" * 70)
        print(f"\n  Output directory: {output_dir}")
        print("\n  Generated files:")
        for f in sorted(output_dir.glob("*")):
            print(f"    - {f.name}")
        print("=" * 70)

        return {
            'metrics': metrics,
            'metrics_dict': metrics_dict,
            'output_dir': output_dir,
            'tracking_rate': tracking_rate,
        }

    def _draw_tracking(
        self,
        frame: np.ndarray,
        result: dict,
        frame_idx: int,
        fps: float,
    ) -> np.ndarray:
        """Draw minimal tracking visualization."""
        vis = frame.copy()

        center = result['center']
        contour = result['contour']

        # Contour
        cv2.drawContours(vis, [contour], -1, (0, 255, 0), 2, cv2.LINE_AA)

        # Center
        cv2.circle(vis, center, 8, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.circle(vis, center, 5, (0, 255, 255), -1, cv2.LINE_AA)

        # Trail
        positions = list(self.tracker.positions)
        if len(positions) > 1:
            for i in range(1, len(positions)):
                alpha = i / len(positions)
                color = (0, int(255 * alpha), int(200 * alpha))
                cv2.line(vis, positions[i-1], positions[i], color, 1, cv2.LINE_AA)

        # Info panel
        h, w = vis.shape[:2]
        overlay = vis.copy()
        cv2.rectangle(overlay, (10, 10), (200, 60), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, vis, 0.3, 0, vis)

        time_str = f"Time: {frame_idx/fps:.1f}s"
        pos_str = f"Pos: ({center[0]}, {center[1]})"

        cv2.putText(vis, time_str, (20, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(vis, pos_str, (20, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 255), 1, cv2.LINE_AA)

        return vis

    def _export_csv(self, path: Path, metrics: ScientificGaitMetrics):
        """Export tracking data to CSV."""
        import csv

        with open(path, 'w', newline='') as f:
            writer = csv.writer(f)

            # Header
            writer.writerow(['frame', 'time_s', 'x_px', 'y_px', 'velocity_cm_s'])

            # Data
            for i, (pos, t) in enumerate(zip(self.positions, self.timestamps)):
                vel = metrics.velocity_timeseries[i] if i < len(metrics.velocity_timeseries) else 0
                writer.writerow([i, f"{t:.4f}", pos[0], pos[1], f"{vel:.2f}"])

        print(f"  CSV exported: {path.name}")


# =============================================================================
# Main
# =============================================================================
def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Scientific Mouse Locomotor Tracker",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python process_scientific.py video.mp4
  python process_scientific.py video.mp4 -o results/ --experiment EXP001
  python process_scientific.py video.mp4 --no-video --figures-only
        """
    )

    parser.add_argument("input", help="Input video path")
    parser.add_argument("-o", "--output", help="Output directory")
    parser.add_argument("--experiment", default="EXP001", help="Experiment ID")
    parser.add_argument("--subject", default="MOUSE001", help="Subject ID")
    parser.add_argument("--experimenter", default="Unknown", help="Experimenter name")
    parser.add_argument("--institution", default="Stride Labs", help="Institution")
    parser.add_argument("--journal", default="nature",
                        choices=['nature', 'cell', 'default'],
                        help="Target journal style for figures")
    parser.add_argument("--no-video", action="store_true", help="Skip video generation")
    parser.add_argument("--no-figures", action="store_true", help="Skip figure generation")
    parser.add_argument("--no-reports", action="store_true", help="Skip report generation")
    parser.add_argument("--preview", action="store_true", help="Show preview window")
    parser.add_argument("--max-frames", type=int, help="Limit frames for testing")

    args = parser.parse_args()

    processor = ScientificProcessor(
        experiment_id=args.experiment,
        subject_id=args.subject,
        experimenter=args.experimenter,
        institution=args.institution,
        journal=args.journal,
    )

    results = processor.process(
        input_path=args.input,
        output_dir=args.output,
        generate_video=not args.no_video,
        generate_figures=not args.no_figures,
        generate_reports=not args.no_reports,
        preview=args.preview,
        max_frames=args.max_frames,
    )


if __name__ == "__main__":
    main()
