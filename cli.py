#!/usr/bin/env python3
"""
Mouse Locomotor Tracker - Professional CLI
==========================================

World-class command-line interface with rich output.

Author: Stride Labs
License: MIT
"""

import sys
from pathlib import Path
from typing import Optional

try:
    import typer
    from rich.console import Console
    from rich.panel import Panel
    from rich.progress import (
        Progress,
        SpinnerColumn,
        TextColumn,
        BarColumn,
        TaskProgressColumn,
        TimeRemainingColumn,
    )
    from rich.table import Table
    from rich import print as rprint
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    print("Install rich and typer for enhanced CLI: pip install typer rich")

import cv2
import numpy as np

# Initialize app
app = typer.Typer(
    name="mlt",
    help="Mouse Locomotor Tracker - Professional biomechanical analysis",
    add_completion=False,
)

console = Console() if RICH_AVAILABLE else None


def create_banner():
    """Create professional ASCII banner."""
    banner = """
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║   ███╗   ███╗██╗  ████████╗                                  ║
║   ████╗ ████║██║  ╚══██╔══╝  Mouse Locomotor Tracker        ║
║   ██╔████╔██║██║     ██║     Professional Edition v1.0      ║
║   ██║╚██╔╝██║██║     ██║                                     ║
║   ██║ ╚═╝ ██║███████╗██║     Stride Labs                    ║
║   ╚═╝     ╚═╝╚══════╝╚═╝                                     ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
"""
    return banner


class MotionTracker:
    """High-performance motion-based mouse tracker."""

    def __init__(self, frame_width: int, frame_height: int):
        self.roi_y1 = int(frame_height * 0.40)
        self.roi_y2 = int(frame_height * 0.78)
        self.frame_buffer = []
        self.buffer_size = 3
        self.positions = []
        self.prev_center = None

    def track(self, frame: np.ndarray) -> Optional[dict]:
        """Track mouse using frame differencing."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        roi = gray[self.roi_y1:self.roi_y2, :]

        self.frame_buffer.append(roi)
        if len(self.frame_buffer) > self.buffer_size:
            self.frame_buffer.pop(0)

        if len(self.frame_buffer) < 2:
            return None

        motion_accum = np.zeros_like(roi, dtype=np.float32)
        for i in range(1, len(self.frame_buffer)):
            diff = cv2.absdiff(self.frame_buffer[i], self.frame_buffer[i-1])
            motion_accum += diff.astype(np.float32)

        motion_accum = (motion_accum / (len(self.frame_buffer) - 1)).astype(np.uint8)
        _, motion_mask = cv2.threshold(motion_accum, 10, 255, cv2.THRESH_BINARY)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_CLOSE, kernel, iterations=3)
        motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_OPEN, kernel)
        motion_mask = cv2.dilate(motion_mask, kernel, iterations=2)

        contours, _ = cv2.findContours(motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None

        best_cnt = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(best_cnt)

        if area < 500:
            return None

        M = cv2.moments(best_cnt)
        if M['m00'] == 0:
            return None

        cx_roi = int(M['m10'] / M['m00'])
        cy_roi = int(M['m01'] / M['m00'])
        cx = cx_roi
        cy = cy_roi + self.roi_y1

        cnt_full = best_cnt.copy()
        cnt_full[:, :, 1] += self.roi_y1

        if self.prev_center:
            alpha = 0.6
            cx = int(self.prev_center[0] * (1-alpha) + cx * alpha)
            cy = int(self.prev_center[1] * (1-alpha) + cy * alpha)

        self.prev_center = (cx, cy)
        self.positions.append((cx, cy))
        if len(self.positions) > 50:
            self.positions.pop(0)

        return {
            'contour': cnt_full,
            'center': (cx, cy),
            'area': area
        }


@app.command()
def process(
    input_path: Path = typer.Argument(..., help="Input video file path"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output video path"),
    preview: bool = typer.Option(False, "--preview", "-p", help="Show preview window"),
    max_frames: Optional[int] = typer.Option(None, "--max-frames", "-m", help="Limit frames to process"),
    export_csv: bool = typer.Option(False, "--csv", help="Export tracking data to CSV"),
    export_json: bool = typer.Option(False, "--json", help="Export results to JSON"),
):
    """
    Process video and track mouse locomotion.

    Example:
        mlt process video.mp4 -o tracked.mp4 --csv
    """
    if RICH_AVAILABLE:
        console.print(create_banner(), style="cyan")
    else:
        print(create_banner())

    if not input_path.exists():
        if RICH_AVAILABLE:
            console.print(f"[red]Error:[/red] File not found: {input_path}")
        else:
            print(f"Error: File not found: {input_path}")
        raise typer.Exit(1)

    cap = cv2.VideoCapture(str(input_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if max_frames:
        total = min(total, max_frames)

    # Video info table
    if RICH_AVAILABLE:
        table = Table(title="Video Information", show_header=False)
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="green")
        table.add_row("Input", str(input_path))
        table.add_row("Resolution", f"{w}x{h}")
        table.add_row("FPS", f"{fps:.1f}")
        table.add_row("Frames", str(total))
        table.add_row("Duration", f"{total/fps:.1f}s")
        console.print(table)
    else:
        print(f"Input: {input_path}")
        print(f"Resolution: {w}x{h} @ {fps:.1f} FPS")
        print(f"Frames: {total}")

    tracker = MotionTracker(w, h)
    cm_per_px = 40.0 / w

    if output is None:
        output = input_path.parent / f"{input_path.stem}_tracked.mp4"

    out = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    positions = []
    velocities = []
    distance = 0.0
    avg_speed = 0.0
    peak_speed = 0.0
    dt = 1.0 / fps
    tracked = 0

    # Processing with progress bar
    if RICH_AVAILABLE:
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=40),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            console=console,
        )

        with progress:
            task = progress.add_task("[cyan]Processing frames...", total=total)

            for idx in range(total):
                ret, frame = cap.read()
                if not ret:
                    break

                result = tracker.track(frame)

                if result:
                    tracked += 1
                    center = result['center']
                    positions.append(center)

                    vel = 0.0
                    if len(positions) >= 2:
                        dx = (positions[-1][0] - positions[-2][0]) * cm_per_px
                        dy = (positions[-1][1] - positions[-2][1]) * cm_per_px
                        d = np.sqrt(dx**2 + dy**2)
                        distance += d
                        vel = min(d / dt, 35)

                    velocities.append(vel)
                    valid = [v for v in velocities if 0.2 < v < 35]
                    if valid:
                        avg_speed = np.mean(valid)
                        peak_speed = max(peak_speed, max(valid))

                    vis = _draw_overlay(frame, result, vel, avg_speed, distance, tracker.positions)
                else:
                    vis = frame.copy()

                out.write(vis)

                if preview:
                    cv2.imshow('MLT - Mouse Locomotor Tracker', cv2.resize(vis, (1280, 720)))
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                progress.update(task, advance=1)
    else:
        for idx in range(total):
            ret, frame = cap.read()
            if not ret:
                break

            result = tracker.track(frame)
            if result:
                tracked += 1
                # ... same processing logic ...

            if idx % 30 == 0:
                print(f"\rProgress: {idx}/{total} ({idx*100//total}%)", end="")

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    # Results summary
    tracking_rate = tracked / total * 100 if total > 0 else 0

    if RICH_AVAILABLE:
        results_table = Table(title="Results Summary", show_header=False)
        results_table.add_column("Metric", style="cyan")
        results_table.add_column("Value", style="green")
        results_table.add_row("Tracking Rate", f"{tracking_rate:.1f}%")
        results_table.add_row("Avg Speed", f"{avg_speed:.2f} cm/s")
        results_table.add_row("Peak Speed", f"{peak_speed:.2f} cm/s")
        results_table.add_row("Total Distance", f"{distance:.1f} cm")
        results_table.add_row("Output", str(output))
        console.print(results_table)

        console.print(Panel.fit(
            f"[green]Processing complete![/green]\n"
            f"Video saved to: [cyan]{output}[/cyan]",
            title="Success",
            border_style="green"
        ))
    else:
        print(f"\n\nResults:")
        print(f"  Tracking Rate: {tracking_rate:.1f}%")
        print(f"  Avg Speed: {avg_speed:.2f} cm/s")
        print(f"  Peak Speed: {peak_speed:.2f} cm/s")
        print(f"  Distance: {distance:.1f} cm")
        print(f"  Output: {output}")

    # Export options
    if export_csv:
        csv_path = output.with_suffix('.csv')
        _export_csv(positions, velocities, fps, csv_path)
        if RICH_AVAILABLE:
            console.print(f"[green]CSV exported:[/green] {csv_path}")

    if export_json:
        json_path = output.with_suffix('.json')
        _export_json(positions, velocities, avg_speed, peak_speed, distance, tracking_rate, json_path)
        if RICH_AVAILABLE:
            console.print(f"[green]JSON exported:[/green] {json_path}")


def _draw_overlay(frame, result, vel, avg_speed, distance, positions):
    """Draw professional overlay on frame."""
    vis = frame.copy()
    center = result['center']
    contour = result['contour']

    cv2.drawContours(vis, [contour], -1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.circle(vis, center, 6, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.circle(vis, center, 3, (0, 255, 255), -1, cv2.LINE_AA)

    if len(positions) > 1:
        pts = positions
        for i in range(1, len(pts)):
            a = i / len(pts)
            c = (0, int(150*a), int(180*a))
            cv2.line(vis, pts[i-1], pts[i], c, 1, cv2.LINE_AA)

    # HUD
    panel = vis.copy()
    cv2.rectangle(panel, (8, 8), (180, 75), (0, 0, 0), -1)
    cv2.addWeighted(panel, 0.6, vis, 0.4, 0, vis)
    cv2.rectangle(vis, (8, 8), (180, 75), (0, 200, 255), 1)

    cv2.putText(vis, "MLT v1.0", (14, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(vis, f"Speed: {vel:.1f} cm/s", (14, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255,255,255), 1, cv2.LINE_AA)
    cv2.putText(vis, f"Avg: {avg_speed:.1f} cm/s", (14, 54),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255,255,255), 1, cv2.LINE_AA)
    cv2.putText(vis, f"Dist: {distance:.0f} cm", (14, 68),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255,255,255), 1, cv2.LINE_AA)

    return vis


def _export_csv(positions, velocities, fps, path):
    """Export tracking data to CSV."""
    import csv
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['frame', 'time_s', 'x_px', 'y_px', 'velocity_cm_s'])
        for i, (pos, vel) in enumerate(zip(positions, velocities)):
            writer.writerow([i, i/fps, pos[0], pos[1], f"{vel:.3f}"])


def _export_json(positions, velocities, avg_speed, peak_speed, distance, tracking_rate, path):
    """Export results to JSON."""
    import json
    data = {
        "metadata": {
            "version": "1.0",
            "tool": "Mouse Locomotor Tracker",
            "author": "Stride Labs"
        },
        "summary": {
            "tracking_rate_percent": round(tracking_rate, 2),
            "avg_speed_cm_s": round(avg_speed, 3),
            "peak_speed_cm_s": round(peak_speed, 3),
            "total_distance_cm": round(distance, 2),
            "n_frames_tracked": len(positions)
        },
        "trajectory": {
            "x": [p[0] for p in positions],
            "y": [p[1] for p in positions],
            "velocity": [round(v, 3) for v in velocities]
        }
    }
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


@app.command()
def info(video_path: Path = typer.Argument(..., help="Video file to analyze")):
    """
    Display video information without processing.
    """
    if not video_path.exists():
        console.print(f"[red]Error:[/red] File not found: {video_path}")
        raise typer.Exit(1)

    cap = cv2.VideoCapture(str(video_path))

    info_table = Table(title=f"Video Info: {video_path.name}")
    info_table.add_column("Property", style="cyan")
    info_table.add_column("Value", style="green")

    info_table.add_row("Resolution", f"{int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
    info_table.add_row("FPS", f"{cap.get(cv2.CAP_PROP_FPS):.2f}")
    info_table.add_row("Frame Count", str(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))))
    info_table.add_row("Duration", f"{cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS):.2f}s")
    info_table.add_row("Codec", str(int(cap.get(cv2.CAP_PROP_FOURCC))))

    cap.release()
    console.print(info_table)


@app.command()
def version():
    """Show version information."""
    if RICH_AVAILABLE:
        console.print(Panel.fit(
            "[cyan]Mouse Locomotor Tracker[/cyan]\n"
            "Version: [green]1.0.0[/green]\n"
            "Author: [yellow]Stride Labs[/yellow]\n"
            "License: MIT",
            title="MLT",
            border_style="cyan"
        ))
    else:
        print("Mouse Locomotor Tracker v1.0.0")
        print("Author: Stride Labs")


if __name__ == "__main__":
    app()
