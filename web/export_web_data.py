#!/usr/bin/env python3
"""
Export tracking data for web visualization.
Generates JSON file with all metrics and timeseries data.
"""

import json
import csv
from pathlib import Path


def export_web_data(
    report_json: str,
    data_csv: str,
    output_path: str,
    fps: float = 30.0
):
    """Export all data needed for web visualization."""

    # Load report
    with open(report_json, 'r') as f:
        report = json.load(f)

    # Load timeseries data
    frames = []
    times = []
    x_positions = []
    y_positions = []
    velocities = []

    with open(data_csv, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            frames.append(int(row['frame']))
            times.append(float(row['time_s']))
            x_positions.append(int(row['x_px']))
            y_positions.append(int(row['y_px']))
            velocities.append(float(row['velocity_cm_s']))

    # Calculate activity for each frame
    activities = []
    for v in velocities:
        if v < 2:
            activities.append('REST')
        elif v < 10:
            activities.append('WALK')
        elif v < 25:
            activities.append('TROT')
        else:
            activities.append('GALLOP')

    # Build web data structure
    web_data = {
        "metadata": report["metadata"],
        "fps": fps,
        "summary": {
            "velocity": report["metrics"]["spatiotemporal"]["velocity"],
            "stride_length": report["metrics"]["spatiotemporal"]["stride_length"],
            "cadence": report["metrics"]["spatiotemporal"]["cadence"],
            "step_cycle": report["metrics"]["spatiotemporal"]["step_cycle"],
            "swing_time": report["metrics"]["spatiotemporal"]["swing_time"],
            "stance_time": report["metrics"]["spatiotemporal"]["stance_time"],
            "duty_factor": report["metrics"]["spatiotemporal"]["duty_factor"],
            "swing_speed": report["metrics"]["spatiotemporal"]["swing_speed"],
            "acceleration": report["metrics"]["kinematic"]["acceleration"],
            "jerk": report["metrics"]["kinematic"]["jerk"],
            "regularity_index": report["metrics"]["variability"]["regularity_index"],
            "stride_variability": report["metrics"]["variability"]["stride_variability"],
            "total_distance_cm": report["metrics"]["distance_duration"]["total_distance_cm"],
            "total_duration_s": report["metrics"]["distance_duration"]["total_duration_s"],
            "tracking_rate": report["metrics"]["quality"]["tracking_rate"],
            "n_strides_detected": report["metrics"]["quality"]["n_strides_detected"],
            "data_quality_score": report["metrics"]["quality"]["data_quality_score"],
        },
        "activity": report["metrics"]["activity"],
        "timeseries": {
            "frames": frames,
            "times": times,
            "x_positions": x_positions,
            "y_positions": y_positions,
            "velocities": velocities,
            "activities": activities,
        }
    }

    # Write output
    with open(output_path, 'w') as f:
        json.dump(web_data, f, indent=2)

    print(f"Web data exported to: {output_path}")
    print(f"  - {len(frames)} frames")
    print(f"  - {report['metrics']['quality']['n_strides_detected']} strides")
    print(f"  - {report['metrics']['distance_duration']['total_distance_cm']:.1f} cm total distance")


if __name__ == "__main__":
    base_dir = Path(__file__).parent.parent

    export_web_data(
        report_json=str(base_dir / "output/scientific_analysis/DEMO001_report.json"),
        data_csv=str(base_dir / "output/scientific_analysis/DEMO001_data.csv"),
        output_path=str(base_dir / "web/assets/tracking_data.json"),
        fps=30.0
    )
