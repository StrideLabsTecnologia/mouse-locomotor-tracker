<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/OpenCV-4.5%2B-green?logo=opencv&logoColor=white" alt="OpenCV">
  <img src="https://img.shields.io/badge/License-MIT-yellow" alt="License">
  <img src="https://img.shields.io/badge/code%20style-black-000000.svg" alt="Code Style">
  <img src="https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit" alt="Pre-commit">
</p>

<h1 align="center">
  <br>
  Mouse Locomotor Tracker
  <br>
</h1>

<h4 align="center">Professional biomechanical analysis for rodent locomotion research</h4>

<p align="center">
  <a href="#key-features">Key Features</a> •
  <a href="#installation">Installation</a> •
  <a href="#quick-start">Quick Start</a> •
  <a href="#cli-usage">CLI Usage</a> •
  <a href="#api-reference">API</a> •
  <a href="#citation">Citation</a>
</p>

---

## Overview

**Mouse Locomotor Tracker (MLT)** is a professional-grade Python toolkit for automated tracking and biomechanical analysis of rodent locomotion. Designed for neuroscience research, it provides:

- **Motion-based tracking** optimized for treadmill experiments
- **Biomechanical metrics**: velocity, acceleration, gait cycles, coordination
- **Publication-ready visualizations**: trajectory overlays, polar plots, speed profiles
- **Scientific export formats**: CSV, JSON, HDF5, NWB

```
╔══════════════════════════════════════════════════════════════╗
║   ███╗   ███╗██╗  ████████╗                                  ║
║   ████╗ ████║██║  ╚══██╔══╝  Mouse Locomotor Tracker        ║
║   ██╔████╔██║██║     ██║     Professional Edition v1.0      ║
║   ██║╚██╔╝██║██║     ██║                                     ║
║   ██║ ╚═╝ ██║███████╗██║     Stride Labs                    ║
║   ╚═╝     ╚═╝╚══════╝╚═╝                                     ║
╚══════════════════════════════════════════════════════════════╝
```

---

## Key Features

### Tracking
- **Motion-based detection**: Frame differencing optimized for treadmill ROI
- **100% tracking rate** on standard treadmill videos
- **Temporal smoothing**: Exponential moving average for stable tracking
- **Optional DeepLabCut integration**: 27 anatomical keypoints

### Analysis
| Module | Metrics |
|--------|---------|
| **Velocity** | Instantaneous, average, peak speed; acceleration |
| **Coordination** | Circular statistics, phase coupling, Rayleigh test |
| **Gait Cycles** | Cadence, stride length, duty factor, symmetry |
| **Kinematics** | Joint angles, range of motion, limb lengths |

### Visualization
- Trajectory overlays with gradient trails
- Circular coordination plots (polar)
- Speed profiles with acceleration overlay
- Real-time dashboard with gauges
- Publication-ready figure export

### Export
- **CSV**: Frame-by-frame tracking data
- **JSON**: Complete results with metadata
- **HDF5**: Efficient binary format for large datasets
- **NWB**: Neurodata Without Borders (neuroscience standard)

---

## Installation

### Basic Installation

```bash
pip install mouse-locomotor-tracker
```

### With CLI Support (Recommended)

```bash
pip install mouse-locomotor-tracker[cli]
```

### Full Installation (All Features)

```bash
pip install mouse-locomotor-tracker[all]
```

### Development Installation

```bash
git clone https://github.com/stridelabs/mouse-locomotor-tracker.git
cd mouse-locomotor-tracker
pip install -e ".[dev]"
pre-commit install
```

---

## Quick Start

### Command Line

```bash
# Process video with default settings
mlt process video.mp4 -o tracked.mp4

# With CSV export and preview
mlt process video.mp4 -o tracked.mp4 --csv --preview

# Get video info
mlt info video.mp4
```

### Python API

```python
from analysis import VelocityAnalyzer, GaitCycleDetector
from visualization import TrajectoryVisualizer

# Analyze velocity
analyzer = VelocityAnalyzer(frame_rate=30, pixel_to_mm=0.1)
metrics = analyzer.analyze(x_coords, y_coords)

print(f"Average Speed: {metrics.mean_speed:.2f} cm/s")
print(f"Peak Speed: {metrics.max_speed:.2f} cm/s")
print(f"Distance: {metrics.total_distance:.1f} cm")

# Detect gait cycles
detector = GaitCycleDetector(fps=30)
gait = detector.detect_cycles(stride_signal, x_positions)

print(f"Cadence: {gait.cadence:.2f} Hz")
print(f"Stride Length: {gait.mean_stride_length:.2f} cm")
```

---

## CLI Usage

### Process Command

```bash
mlt process INPUT [OPTIONS]

Arguments:
  INPUT                 Input video file path

Options:
  -o, --output PATH     Output video path
  -p, --preview         Show preview window
  -m, --max-frames INT  Limit frames to process
  --csv                 Export tracking data to CSV
  --json                Export results to JSON
  --help                Show this message and exit
```

### Examples

```bash
# Basic processing
mlt process experiment_001.mp4

# Full analysis with exports
mlt process experiment_001.mp4 \
    --output results/tracked.mp4 \
    --csv \
    --json \
    --max-frames 1000

# Quick preview
mlt process video.mp4 --preview --max-frames 100
```

---

## API Reference

### Analysis Module

```python
from analysis import (
    VelocityAnalyzer,
    CircularCoordinationAnalyzer,
    GaitCycleDetector,
    JointAngleAnalyzer,
)

# Velocity Analysis
analyzer = VelocityAnalyzer(frame_rate=30, pixel_to_mm=0.1)
metrics = analyzer.analyze(x, y)
# Returns: VelocityMetrics(mean_speed, max_speed, acceleration, ...)

# Coordination Analysis
coord = CircularCoordinationAnalyzer()
stats = coord.analyze_pair(phases_a, phases_b)
# Returns: CircularStatistics(mean_angle, R, rayleigh_z, p_value)

# Gait Cycle Detection
detector = GaitCycleDetector(fps=30, min_cycle_duration=0.1)
gait = detector.detect_cycles(stride_signal, x_positions)
# Returns: GaitMetrics(cadence, stride_length, duty_factor, ...)
```

### Visualization Module

```python
from visualization import (
    TrajectoryVisualizer,
    CoordinationPlotter,
    SpeedProfilePlotter,
    VideoGenerator,
)

# Trajectory overlay
viz = TrajectoryVisualizer(trail_length=50, color_scheme='velocity')
frame_with_overlay = viz.draw(frame, positions, velocities)

# Polar coordination plot
plotter = CoordinationPlotter()
fig = plotter.plot_pair(phases, title="LH-RH Coordination")
fig.savefig("coordination.png", dpi=300)

# Generate annotated video
generator = VideoGenerator(fps=30, codec='h264')
generator.process_video(input_path, output_path, tracking_data)
```

---

## Project Structure

```
mouse-locomotor-tracker/
├── analysis/              # Biomechanical analysis
│   ├── velocity.py        # Speed & acceleration
│   ├── coordination.py    # Circular statistics
│   ├── gait_cycles.py     # Cycle detection
│   ├── kinematics.py      # Joint angles
│   └── metrics.py         # Data structures
├── visualization/         # Plotting & video
│   ├── trajectory_overlay.py
│   ├── circular_plots.py
│   ├── speed_plots.py
│   ├── dashboard.py
│   └── video_generator.py
├── tracking/              # Pose estimation
│   ├── dlc_adapter.py     # DeepLabCut wrapper
│   ├── marker_config.py   # Keypoint definitions
│   └── track_processor.py # Post-processing
├── export/                # Data export
│   ├── csv_exporter.py
│   ├── json_exporter.py
│   └── report_generator.py
├── tests/                 # Test suite
├── docs/                  # Documentation
├── cli.py                 # Command-line interface
├── main.py                # Entry point
└── pyproject.toml         # Project configuration
```

---

## Metrics Reference

### Velocity Metrics

| Metric | Unit | Description |
|--------|------|-------------|
| `mean_speed` | cm/s | Average instantaneous speed |
| `max_speed` | cm/s | Peak speed recorded |
| `min_speed` | cm/s | Minimum speed (excluding stops) |
| `total_distance` | cm | Cumulative distance traveled |
| `acceleration` | cm/s² | Rate of speed change |

### Coordination Metrics

| Metric | Range | Description |
|--------|-------|-------------|
| `mean_angle` | -180° to 180° | Mean phase relationship |
| `R` | 0 to 1 | Resultant vector length (coordination strength) |
| `rayleigh_z` | ≥0 | Rayleigh test statistic |
| `p_value` | 0 to 1 | Statistical significance |

### Gait Metrics

| Metric | Unit | Description |
|--------|------|-------------|
| `cadence` | Hz | Steps per second |
| `stride_length` | cm | Distance per cycle |
| `duty_factor` | ratio | Stance/cycle duration |
| `symmetry_index` | % | Left-right symmetry |

---

## Citation

If you use MLT in your research, please cite:

```bibtex
@software{mlt2024,
  author = {Stride Labs},
  title = {Mouse Locomotor Tracker: Professional Biomechanical Analysis},
  year = {2024},
  url = {https://github.com/stridelabs/mouse-locomotor-tracker},
  version = {1.0.0}
}
```

---

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'feat: add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## License

MIT License - see [LICENSE](LICENSE) for details.

---

<p align="center">
  Made with ❤️ by <a href="https://stridelabs.cl">Stride Labs</a>
</p>
