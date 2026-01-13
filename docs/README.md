# Mouse Locomotor Tracker

A comprehensive Python library for analyzing mouse locomotion from video recordings using DeepLabCut pose estimation.

## Overview

Mouse Locomotor Tracker provides automated analysis of rodent locomotion, including:

- **Velocity Analysis**: Speed, acceleration, and drag detection
- **Limb Coordination**: Phase relationships between limbs using circular statistics
- **Gait Cycle Detection**: Cadence, stride length, and gait regularity metrics
- **Export**: JSON, CSV, and HDF5 output formats

```
+------------------+     +------------------+     +------------------+
|                  |     |                  |     |                  |
|  Video Input     +---->+  DeepLabCut      +---->+  Analysis        |
|  (.avi, .mp4)    |     |  Pose Tracking   |     |  Pipeline        |
|                  |     |                  |     |                  |
+------------------+     +------------------+     +--------+---------+
                                                          |
                         +--------------------------------+
                         |
          +--------------+---------------+----------------+
          |              |               |                |
          v              v               v                v
    +-----------+  +-----------+  +------------+  +------------+
    | Velocity  |  | Coordin.  |  | Gait       |  | Export     |
    | Analysis  |  | Analysis  |  | Cycles     |  | Module     |
    +-----------+  +-----------+  +------------+  +------------+
```

## Features

### Velocity Analysis
- Instantaneous and average speed computation
- Acceleration and deceleration detection
- Drag event identification and quantification
- Multiple smoothing filter options

### Limb Coordination
- Circular statistics for phase analysis
- All standard limb pair combinations
- Gait pattern classification (trot, pace, bound)
- Resultant vector length (R) for coordination strength

### Gait Cycle Detection
- Automatic cycle detection using peak finding
- Cadence (step frequency) calculation
- Stride length estimation
- Gait regularity metrics (CV of cycle duration)

### Export Options
- JSON for full results with metadata
- CSV for summary statistics
- HDF5 for large datasets
- Matplotlib visualizations

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/stridelabs/mouse-locomotor-tracker.git
cd mouse-locomotor-tracker

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

### Basic Usage

```python
from mouse_locomotor_tracker import LocomotorPipeline
from mouse_locomotor_tracker.tracking import VideoMetadata, MarkerSet

# Configure markers
markers = MarkerSet(
    name="mouse_ventral",
    markers=["snout", "foreL", "foreR", "hindL", "hindR", "torso", "tail"],
    limb_pairs={
        "LH_RH": ("hindL", "hindR"),
        "diagonal": ("foreL", "hindR"),
    },
    speed_markers=["snout", "torso", "tail"]
)

# Load tracking data (DeepLabCut output)
import pandas as pd
tracks = pd.read_hdf("tracking_results.h5")

# Create metadata
metadata = VideoMetadata(
    duration=30.0,      # seconds
    fps=100,            # frames per second
    n_frames=3000,
    width=640,
    height=480,
    pixel_width_mm=0.3125
)

# Run analysis
pipeline = LocomotorPipeline()
results = pipeline.process_tracks(
    tracks=tracks,
    metadata=metadata,
    model_name="DLC_model",
    markers=markers.markers,
    limb_pairs=markers.limb_pairs,
    speed_markers=markers.speed_markers
)

# Export results
pipeline.export_results("results.json", format="json")

# Access specific metrics
print(f"Mean Speed: {results['summary']['mean_speed_cm_s']:.2f} cm/s")
print(f"Mean Cadence: {results['summary']['mean_cadence_hz']:.2f} Hz")
print(f"Coordination (R): {results['summary']['mean_coordination_R']:.3f}")
```

## Architecture

```
mouse-locomotor-tracker/
|
+-- tracking/                  # Tracking module
|   +-- __init__.py
|   +-- dlc_adapter.py        # DeepLabCut interface
|   +-- marker_config.py      # Marker configuration
|   +-- video_metadata.py     # Video file metadata
|   +-- mock_tracker.py       # Synthetic data for testing
|   +-- track_processor.py    # Track post-processing
|
+-- analysis/                  # Analysis modules
|   +-- __init__.py
|   +-- velocity.py           # VelocityAnalyzer
|   +-- coordination.py       # CircularCoordinationAnalyzer
|   +-- gait_cycles.py        # GaitCycleDetector
|   +-- pipeline.py           # LocomotorPipeline
|
+-- visualization/             # Visualization tools
|   +-- __init__.py
|   +-- plotter.py            # Static plots
|   +-- video_overlay.py      # Video annotation
|
+-- export/                    # Export functionality
|   +-- __init__.py
|   +-- json_export.py
|   +-- csv_export.py
|   +-- hdf5_export.py
|
+-- config/                    # Configuration files
|   +-- default_config.yaml
|   +-- markers_ventral.yaml
|   +-- markers_lateral.yaml
|
+-- tests/                     # Test suite
|   +-- conftest.py
|   +-- test_velocity.py
|   +-- test_coordination.py
|   +-- test_gait_cycles.py
|   +-- test_integration.py
|
+-- docs/                      # Documentation
    +-- README.md
    +-- INSTALLATION.md
    +-- USER_GUIDE.md
    +-- API.md
    +-- METRICS.md
```

## Workflow Diagram

```
                    +-------------------+
                    |   Raw Video       |
                    |   (.avi, .mp4)    |
                    +---------+---------+
                              |
                              v
                    +-------------------+
                    |   DeepLabCut      |
                    |   Pose Estimation |
                    +---------+---------+
                              |
                              v
                    +-------------------+
                    |   Tracking Data   |
                    |   (HDF5/CSV)      |
                    +---------+---------+
                              |
              +---------------+---------------+
              |               |               |
              v               v               v
       +------+------+ +------+------+ +------+------+
       |  Velocity   | | Coordination| | Gait Cycle  |
       |  Analyzer   | |  Analyzer   | |  Detector   |
       +------+------+ +------+------+ +------+------+
              |               |               |
              +---------------+---------------+
                              |
                              v
                    +-------------------+
                    |   Results         |
                    |   Aggregation     |
                    +---------+---------+
                              |
              +---------------+---------------+
              |               |               |
              v               v               v
       +------+------+ +------+------+ +------+------+
       |    JSON     | |    CSV      | |   Plots     |
       |   Export    | |   Export    | |   & Video   |
       +-------------+ +-------------+ +-------------+
```

## API Reference

### Core Classes

| Class | Description |
|-------|-------------|
| `LocomotorPipeline` | Main pipeline orchestrating all analysis modules |
| `VelocityAnalyzer` | Speed and acceleration computation |
| `CircularCoordinationAnalyzer` | Limb phase relationship analysis |
| `GaitCycleDetector` | Gait cycle detection and stride metrics |
| `VideoMetadata` | Video file metadata container |
| `MarkerSet` | Marker configuration for tracking |

### Quick Reference

```python
# Velocity Analysis
from analysis.velocity import VelocityAnalyzer
analyzer = VelocityAnalyzer(smoothing_factor=10)
speed = analyzer.compute_speed(positions, fps, pixel_to_mm)
accel = analyzer.compute_acceleration(speed, fps)
drag, recovery, stats = analyzer.detect_drag_events(accel, fps)

# Coordination Analysis
from analysis.coordination import CircularCoordinationAnalyzer
coord = CircularCoordinationAnalyzer()
mean_phi, R = coord.circular_mean(phase_angles)
results = coord.analyze_all_limb_pairs(tracks, limb_pairs, duration)

# Gait Cycle Detection
from analysis.gait_cycles import GaitCycleDetector
detector = GaitCycleDetector()
n_cycles, peaks, troughs = detector.detect_cycles(stride, fps)
cadence = detector.compute_cadence(stride, duration)
stride_length = detector.compute_stride_length(cadence, avg_speed)
```

## Documentation

- [Installation Guide](INSTALLATION.md) - Detailed installation instructions
- [User Guide](USER_GUIDE.md) - Step-by-step usage tutorial
- [API Reference](API.md) - Complete API documentation
- [Metrics Guide](METRICS.md) - Explanation of computed metrics

## Requirements

- Python 3.8+
- NumPy >= 1.20
- Pandas >= 1.3
- SciPy >= 1.7
- Matplotlib >= 3.4
- DeepLabCut >= 2.2 (optional, for pose estimation)

## Testing

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=. --cov-report=html

# Run specific test module
pytest tests/test_velocity.py -v

# Run only fast tests (skip slow performance tests)
pytest tests/ -m "not slow"
```

## Coverage Targets

| Module | Target | Critical Paths |
|--------|--------|----------------|
| Domain/Business Logic | 90%+ | 100% |
| Analysis Modules | 90%+ | 100% |
| Integration | 70%+ | 80% |

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

MIT License - see [LICENSE](../LICENSE) for details.

## Citation

If you use this software in your research, please cite:

```bibtex
@software{mouse_locomotor_tracker,
  title = {Mouse Locomotor Tracker},
  author = {Stride Labs},
  year = {2024},
  url = {https://github.com/stridelabs/mouse-locomotor-tracker}
}
```

## Acknowledgments

This project builds upon methodologies from:

- Allodi et al. (2021) - Locomotor analysis methods
- DeepLabCut - Pose estimation framework
- Circular statistics implementations

## Contact

- **Author**: Stride Labs
- **Email**: contact@stridelabs.cl
- **Website**: https://stridelabs.cl
