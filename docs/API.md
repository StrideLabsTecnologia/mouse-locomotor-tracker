# API Reference

Complete API documentation for Mouse Locomotor Tracker.

## Table of Contents

1. [Tracking Module](#tracking-module)
2. [Analysis Module](#analysis-module)
3. [Visualization Module](#visualization-module)
4. [Export Module](#export-module)

---

## Tracking Module

### `tracking.VideoMetadata`

Container for video file metadata.

```python
from mouse_locomotor_tracker.tracking import VideoMetadata
```

#### Constructor

```python
VideoMetadata(
    duration: float,       # Total duration in seconds
    fps: int,              # Frames per second
    n_frames: int,         # Total number of frames
    width: int,            # Image width in pixels
    height: int,           # Image height in pixels
    pixel_width_mm: float  # Physical width per pixel (mm)
)
```

#### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `duration` | `float` | Video duration in seconds |
| `fps` | `int` | Frames per second |
| `n_frames` | `int` | Total frame count |
| `width` | `int` | Frame width in pixels |
| `height` | `int` | Frame height in pixels |
| `pixel_width_mm` | `float` | Physical size per pixel (mm) |

#### Methods

##### `to_dict() -> dict`

Convert metadata to dictionary format.

```python
metadata = VideoMetadata(duration=30.0, fps=100, n_frames=3000,
                        width=640, height=480, pixel_width_mm=0.3125)
d = metadata.to_dict()
# {'dur': 30.0, 'fps': 100, 'nFrame': 3000, 'imW': 640, 'imH': 480, 'xPixW': 0.3125}
```

##### `from_dict(d: dict) -> VideoMetadata` (classmethod)

Create VideoMetadata from dictionary.

```python
d = {'dur': 30.0, 'fps': 100, 'nFrame': 3000, 'imW': 640, 'imH': 480, 'xPixW': 0.3125}
metadata = VideoMetadata.from_dict(d)
```

---

### `tracking.MarkerSet`

Configuration for tracking markers.

```python
from mouse_locomotor_tracker.tracking import MarkerSet
```

#### Constructor

```python
MarkerSet(
    name: str,                              # Configuration name
    markers: List[str],                     # List of marker names
    limb_pairs: Dict[str, Tuple[str, str]], # Limb pair definitions
    speed_markers: List[str]                # Markers for speed calculation
)
```

#### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `name` | `str` | Configuration identifier |
| `markers` | `List[str]` | All marker names |
| `limb_pairs` | `Dict` | Mapping of pair names to limb tuples |
| `speed_markers` | `List[str]` | Markers used for body speed |

#### Methods

##### `get_all_markers() -> List[str]`

Return list of all marker names.

```python
markers = marker_set.get_all_markers()
# ['snout', 'foreL', 'foreR', 'hindL', 'hindR', 'torso', 'tail']
```

##### `get_limb_pair(pair_name: str) -> Tuple[str, str]`

Get marker tuple for a limb pair.

```python
pair = marker_set.get_limb_pair("LH_RH")
# ('hindL', 'hindR')
```

#### Predefined Marker Sets

```python
from mouse_locomotor_tracker.tracking import MOUSE_VENTRAL, MOUSE_LATERAL

# Ventral view (bottom camera)
MOUSE_VENTRAL  # 11 markers, 6 limb pairs

# Lateral view (side camera)
MOUSE_LATERAL  # 6 markers for joint angles
```

---

### `tracking.MockTracker`

Generate synthetic tracking data for testing.

```python
from mouse_locomotor_tracker.tracking import MockTracker
```

#### Constructor

```python
MockTracker(
    markers: List[str],            # Marker names
    model_name: str = "DLC_mock"   # Model identifier
)
```

#### Methods

##### `generate_tracks(metadata, gait_frequency, speed_cm_s, noise_level) -> pd.DataFrame`

Generate synthetic tracking data.

```python
tracker = MockTracker(markers=["snout", "hindL", "hindR", "tail"])
tracks = tracker.generate_tracks(
    metadata=metadata,
    gait_frequency=4.0,    # Hz
    speed_cm_s=15.0,       # cm/s
    noise_level=1.0        # pixels
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `metadata` | `VideoMetadata` | required | Video metadata |
| `gait_frequency` | `float` | `4.0` | Gait cycle frequency (Hz) |
| `speed_cm_s` | `float` | `15.0` | Locomotion speed (cm/s) |
| `noise_level` | `float` | `1.0` | Tracking noise (pixels) |

**Returns:** `pd.DataFrame` with MultiIndex columns matching DeepLabCut format.

---

## Analysis Module

### `analysis.VelocityAnalyzer`

Analyze speed and acceleration from tracking data.

```python
from mouse_locomotor_tracker.analysis import VelocityAnalyzer
```

#### Constructor

```python
VelocityAnalyzer(
    smoothing_factor: int = 10,        # Speed smoothing window
    accel_smoothing_factor: int = 12,  # Acceleration smoothing window
    speed_threshold: float = 5.0,      # Movement threshold (cm/s)
    drag_threshold: float = 0.25       # Drag event threshold (seconds)
)
```

#### Methods

##### `compute_speed(positions, fps, pixel_to_mm) -> np.ndarray`

Compute instantaneous speed from position array.

```python
analyzer = VelocityAnalyzer()
positions = np.array([[x1, y1], [x2, y2], ...])  # Shape: (n_frames, 2)
speed = analyzer.compute_speed(
    positions=positions,
    fps=100,
    pixel_to_mm=0.3125
)
# Returns: 1D array of speed values in cm/s
```

##### `compute_speed_from_markers(tracks, model_name, speed_markers, fps, pixel_to_mm) -> np.ndarray`

Compute body speed from multiple markers.

```python
speed = analyzer.compute_speed_from_markers(
    tracks=tracks,
    model_name="DLC_model",
    speed_markers=["snout", "torso", "tail"],
    fps=100,
    pixel_to_mm=0.3125
)
```

##### `compute_acceleration(speed, fps) -> np.ndarray`

Compute acceleration from speed profile.

```python
accel = analyzer.compute_acceleration(speed=speed, fps=100)
# Returns: 1D array of acceleration values in cm/s^2
```

##### `detect_drag_events(acceleration, fps) -> Tuple[np.ndarray, np.ndarray, dict]`

Detect drag and recovery events.

```python
drag_idx, recovery_idx, stats = analyzer.detect_drag_events(
    acceleration=accel,
    fps=100
)

# stats contains:
# - drag_count: Number of drag events
# - recovery_count: Number of recovery events
# - drag_duration: Total drag duration (seconds)
# - recovery_duration: Total recovery duration (seconds)
# - peak_acceleration: Maximum acceleration
# - min_acceleration: Minimum acceleration
```

##### `apply_smoothing(data, method, window_size) -> np.ndarray`

Apply smoothing filter to data.

```python
smoothed = analyzer.apply_smoothing(
    data=noisy_data,
    method='moving_average',  # or 'savgol'
    window_size=10
)
```

---

### `analysis.CircularCoordinationAnalyzer`

Analyze limb coordination using circular statistics.

```python
from mouse_locomotor_tracker.analysis import CircularCoordinationAnalyzer
```

#### Constructor

```python
CircularCoordinationAnalyzer(
    interpolation_factor: int = 4,  # Data interpolation factor
    smoothing_factor: int = 10      # Smoothing window
)
```

#### Methods

##### `circular_mean(phi) -> Tuple[float, float]`

Compute circular mean and resultant vector length.

```python
analyzer = CircularCoordinationAnalyzer()
mean_angle, R = analyzer.circular_mean(phi=phase_angles)
# mean_angle: Mean direction in radians
# R: Resultant length [0, 1]
```

**Mathematical Definition:**

```
X = mean(cos(phi))
Y = mean(sin(phi))
R = sqrt(X^2 + Y^2)
mean_phi = atan2(Y, X)
```

##### `compute_limb_coordination(stride_0, stride_1, mov_duration) -> Tuple`

Compute coordination between two limbs.

```python
phi, R, mean_phase_deg, n_steps = analyzer.compute_limb_coordination(
    stride_0=left_hind_stride,
    stride_1=right_hind_stride,
    mov_duration=30.0
)
# phi: Array of phase angles per cycle
# R: Resultant length (coordination strength)
# mean_phase_deg: Mean phase in degrees
# n_steps: Number of steps detected
```

##### `analyze_all_limb_pairs(tracks_dict, limb_pairs, mov_duration) -> dict`

Analyze all defined limb pairs.

```python
tracks_dict = {
    'hindL': hind_left_stride,
    'hindR': hind_right_stride,
    'foreL': fore_left_stride,
    'foreR': fore_right_stride
}

limb_pairs = {
    'LH_RH': ('hindL', 'hindR'),
    'LF_RF': ('foreL', 'foreR'),
}

results = analyzer.analyze_all_limb_pairs(
    tracks_dict=tracks_dict,
    limb_pairs=limb_pairs,
    mov_duration=30.0
)

# results['LH_RH'] = {'R': 0.92, 'mean_phase_deg': 175.3, 'n_steps': 45}
```

##### `interpret_coordination(R, mean_phase) -> str`

Interpret coordination pattern.

```python
pattern = analyzer.interpret_coordination(R=0.9, mean_phase=180.0)
# Returns: 'alternating'

# Possible returns:
# - 'synchronized': In-phase (0 deg)
# - 'alternating': Anti-phase (180 deg)
# - 'leading': ~90 deg lead
# - 'lagging': ~90 deg lag
# - 'weak_coordination': 0.3 < R < 0.7
# - 'no_coordination': R < 0.3
```

##### `measure_cycles(stride) -> Tuple[int, np.ndarray]`

Detect gait cycles in stride data.

```python
n_cycles, peak_indices = analyzer.measure_cycles(stride=stride_array)
```

##### `iqr_mean(data) -> float`

Compute mean after removing outliers via IQR method.

```python
robust_mean = analyzer.iqr_mean(data=noisy_array)
```

---

### `analysis.GaitCycleDetector`

Detect and analyze gait cycles.

```python
from mouse_locomotor_tracker.analysis import GaitCycleDetector
```

#### Constructor

```python
GaitCycleDetector(
    min_peak_distance: int = None,     # Min frames between peaks (auto if None)
    interpolation_factor: int = 4,     # Interpolation factor
    smoothing_factor: int = 10         # Smoothing window
)
```

#### Methods

##### `detect_cycles(stride, fps) -> Tuple[int, np.ndarray, np.ndarray]`

Detect gait cycles using peak detection.

```python
detector = GaitCycleDetector()
n_cycles, peaks, troughs = detector.detect_cycles(
    stride=stride_array,
    fps=100
)
# n_cycles: Number of cycles detected
# peaks: Indices of stride maxima
# troughs: Indices of stride minima
```

##### `compute_cadence(stride, duration) -> float`

Compute stepping frequency.

```python
cadence = detector.compute_cadence(
    stride=stride_array,
    duration=30.0  # seconds
)
# Returns: Steps per second (Hz)
```

##### `compute_stride_length(cadence, avg_speed) -> float`

Compute average stride length.

```python
stride_length = detector.compute_stride_length(
    cadence=4.0,      # Hz
    avg_speed=16.0    # cm/s
)
# Returns: 4.0 cm (stride_length = speed / cadence)
```

##### `analyze_gait_regularity(stride, fps) -> dict`

Analyze gait cycle regularity.

```python
regularity = detector.analyze_gait_regularity(
    stride=stride_array,
    fps=100
)

# Returns:
# {
#     'mean_cycle_duration': 0.25,      # seconds
#     'std_cycle_duration': 0.02,       # seconds
#     'cv_cycle_duration': 0.08,        # coefficient of variation
#     'mean_stride_amplitude': 15.3,    # pixels
#     'std_stride_amplitude': 2.1       # pixels
# }
```

##### `interpolate_stride(stride, duration) -> np.ndarray`

Interpolate stride data for improved detection.

```python
interpolated = detector.interpolate_stride(
    stride=stride_array,
    duration=30.0
)
# Returns: Array with length = original * interpolation_factor
```

##### `compute_all_limb_metrics(limb_strides, duration, avg_speed, fps) -> dict`

Compute metrics for all limbs.

```python
limb_strides = {
    'hindL': hl_stride,
    'hindR': hr_stride,
    'foreL': fl_stride,
    'foreR': fr_stride
}

metrics = detector.compute_all_limb_metrics(
    limb_strides=limb_strides,
    duration=30.0,
    avg_speed=15.0,
    fps=100
)

# metrics['hindL'] = {
#     'cadence': 4.2,
#     'stride_length': 3.6,
#     'mean_cycle_duration': 0.24,
#     'cv_cycle_duration': 0.08,
#     ...
# }
```

##### `detect_gait_events(stride, fps) -> dict`

Detect specific gait events (touchdown, liftoff).

```python
events = detector.detect_gait_events(stride=stride_array, fps=100)

# Returns:
# {
#     'peaks': np.array([...]),        # Maximum extension indices
#     'troughs': np.array([...]),      # Maximum flexion indices
#     'stance_start': np.array([...]), # Paw touchdown indices
#     'swing_start': np.array([...])   # Paw liftoff indices
# }
```

---

### `analysis.LocomotorPipeline`

Main analysis pipeline integrating all modules.

```python
from mouse_locomotor_tracker import LocomotorPipeline
```

#### Constructor

```python
LocomotorPipeline(config: dict = None)
```

**Config Options:**

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `smoothing_factor` | `int` | `10` | Speed smoothing window |
| `interpolation_factor` | `int` | `4` | Stride interpolation |
| `speed_threshold` | `float` | `5.0` | Movement threshold (cm/s) |
| `drag_threshold` | `float` | `0.25` | Drag event threshold (s) |

#### Methods

##### `process_tracks(tracks, metadata, model_name, markers, limb_pairs, speed_markers) -> dict`

Run complete analysis pipeline.

```python
pipeline = LocomotorPipeline(config={'smoothing_factor': 10})

results = pipeline.process_tracks(
    tracks=tracks_dataframe,
    metadata=video_metadata,
    model_name="DLC_model",
    markers=marker_list,
    limb_pairs=limb_pair_dict,
    speed_markers=speed_marker_list
)
```

**Returns:** Dictionary with structure:

```python
{
    'metadata': {...},      # Video metadata
    'velocity': {...},      # Speed and acceleration results
    'coordination': {...},  # Limb coordination results
    'gait_cycles': {...},   # Gait cycle metrics
    'summary': {...}        # Aggregate statistics
}
```

##### `export_results(output_path, format) -> None`

Export results to file.

```python
pipeline.export_results("results.json", format="json")
pipeline.export_results("summary.csv", format="csv")
```

**Supported Formats:**

| Format | Content | File Extension |
|--------|---------|----------------|
| `json` | Full results | `.json` |
| `csv` | Summary only | `.csv` |

---

## Visualization Module

### `visualization.Plotter`

Static visualization utilities.

```python
from mouse_locomotor_tracker.visualization import Plotter
```

#### Methods

##### `plot_speed_profile(speed, fps, ax=None) -> matplotlib.axes.Axes`

Plot speed over time.

```python
plotter = Plotter()
ax = plotter.plot_speed_profile(speed=speed_array, fps=100)
```

##### `plot_coordination_polar(coordination_results, ax=None) -> matplotlib.axes.Axes`

Create polar plot of coordination.

```python
ax = plotter.plot_coordination_polar(
    coordination_results=results['coordination']
)
```

##### `plot_gait_metrics(gait_results, ax=None) -> matplotlib.axes.Axes`

Plot gait metrics comparison.

```python
ax = plotter.plot_gait_metrics(
    gait_results=results['gait_cycles']
)
```

##### `create_summary_figure(results, output_path=None)`

Create comprehensive summary figure.

```python
plotter.create_summary_figure(
    results=results,
    output_path="analysis_summary.png"
)
```

---

## Export Module

### `export.JSONExporter`

Export results to JSON format.

```python
from mouse_locomotor_tracker.export import JSONExporter
```

#### Methods

##### `export(results, output_path) -> None`

Export results to JSON file.

```python
exporter = JSONExporter()
exporter.export(results=results, output_path="output.json")
```

---

### `export.CSVExporter`

Export results to CSV format.

```python
from mouse_locomotor_tracker.export import CSVExporter
```

#### Methods

##### `export(results, output_path) -> None`

Export summary statistics to CSV.

```python
exporter = CSVExporter()
exporter.export(results=results, output_path="summary.csv")
```

##### `export_detailed(results, output_dir) -> None`

Export detailed results to multiple CSV files.

```python
exporter.export_detailed(
    results=results,
    output_dir="results/"
)
# Creates:
# - results/velocity.csv
# - results/coordination.csv
# - results/gait_cycles.csv
# - results/summary.csv
```

---

## Type Definitions

### Common Types

```python
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd

# Position data
Positions = np.ndarray  # Shape: (n_frames, 2)

# Stride data
StrideArray = np.ndarray  # Shape: (n_frames,)

# Phase angles
PhaseArray = np.ndarray  # Shape: (n_cycles,)

# Tracking DataFrame
TrackingData = pd.DataFrame  # MultiIndex columns

# Results dictionary
AnalysisResults = Dict[str, Any]
```

### Result Type Schemas

```python
# Velocity Results
VelocityResults = {
    'mean_speed': float,        # cm/s
    'max_speed': float,         # cm/s
    'min_speed': float,         # cm/s
    'std_speed': float,         # cm/s
    'speed_profile': List[float]
}

# Coordination Results
CoordinationResults = {
    'pair_name': {
        'R': float,              # [0, 1]
        'mean_phase_deg': float, # [-180, 180]
        'n_steps': int
    }
}

# Gait Results
GaitResults = {
    'limb_name': {
        'cadence': float,           # Hz
        'stride_length': float,     # cm
        'n_cycles': int,
        'mean_cycle_duration': float,  # seconds
        'cv_cycle_duration': float     # [0, 1]
    }
}

# Summary Results
SummaryResults = {
    'duration': float,              # seconds
    'mean_speed_cm_s': float,
    'mean_coordination_R': float,
    'mean_cadence_hz': float,
    'mean_stride_length_cm': float
}
```

---

## Error Handling

### Common Exceptions

```python
# ValueError for invalid input
try:
    analyzer.compute_speed(empty_array, fps=100, pixel_to_mm=0.3)
except ValueError as e:
    print(f"Invalid input: {e}")

# KeyError for missing markers
try:
    speed = analyzer.compute_speed_from_markers(
        tracks, model_name, ['nonexistent_marker'], fps, pixel_to_mm
    )
except KeyError as e:
    print(f"Missing marker: {e}")
```

### Validation Functions

```python
from mouse_locomotor_tracker.utils import validate_tracks, validate_metadata

# Validate tracking data structure
is_valid, errors = validate_tracks(tracks)
if not is_valid:
    print(f"Invalid tracking data: {errors}")

# Validate metadata
is_valid, errors = validate_metadata(metadata)
```

---

## Version Information

```python
import mouse_locomotor_tracker

print(mouse_locomotor_tracker.__version__)  # '0.1.0'
print(mouse_locomotor_tracker.__author__)   # 'Stride Labs'
```
