# User Guide

A comprehensive guide to using Mouse Locomotor Tracker for analyzing rodent locomotion.

## Table of Contents

1. [Getting Started](#getting-started)
2. [Processing a Video](#processing-a-video)
3. [Configuring Markers](#configuring-markers)
4. [Understanding Results](#understanding-results)
5. [Interpreting Metrics](#interpreting-metrics)
6. [Advanced Usage](#advanced-usage)
7. [Examples](#examples)

## Getting Started

### Basic Workflow

```
+-------------+     +-------------+     +-------------+     +-------------+
|   Record    |     |   Track     |     |   Analyze   |     |   Export    |
|   Video     +---->+   Poses     +---->+   Motion    +---->+   Results   |
+-------------+     +-------------+     +-------------+     +-------------+
     ^                    |                   |                   |
     |                    v                   v                   v
  Camera            DeepLabCut         This Library          JSON/CSV
  Setup              or Manual                               Reports
```

### Prerequisites

1. **Video recordings** of mouse locomotion
2. **Pose tracking data** from DeepLabCut (or equivalent)
3. **Python environment** with Mouse Locomotor Tracker installed

## Processing a Video

### Step 1: Prepare Your Data

Your tracking data should be in HDF5 or CSV format from DeepLabCut:

```
tracking_data/
+-- video_01DLC_resnet50_mouseJan1shuffle1_500000.h5
+-- video_01DLC_resnet50_mouseJan1shuffle1_500000_labeled.mp4
+-- video_01.avi
```

### Step 2: Load Tracking Data

```python
import pandas as pd
from mouse_locomotor_tracker import LocomotorPipeline
from mouse_locomotor_tracker.tracking import VideoMetadata, MarkerSet

# Load DeepLabCut tracking data
tracks = pd.read_hdf("tracking_data/video_01DLC_resnet50_mouseJan1shuffle1_500000.h5")

# Check the structure
print("Columns:", tracks.columns.levels)
print("Shape:", tracks.shape)
```

### Step 3: Configure Video Metadata

```python
# Option 1: Manual configuration
metadata = VideoMetadata(
    duration=30.0,        # Total duration in seconds
    fps=100,              # Frames per second
    n_frames=3000,        # Total number of frames
    width=640,            # Image width in pixels
    height=480,           # Image height in pixels
    pixel_width_mm=0.3125 # Physical size of one pixel (mm)
)

# Option 2: Extract from video file
from mouse_locomotor_tracker.tracking import extract_video_metadata
metadata = extract_video_metadata("tracking_data/video_01.avi")
```

### Step 4: Configure Markers

```python
# Standard ventral view markers
markers = MarkerSet(
    name="mouse_ventral",
    markers=[
        "snout", "snoutL", "snoutR",
        "foreL", "foreR",
        "hindL", "hindR",
        "torso", "torsoL", "torsoR",
        "tail"
    ],
    limb_pairs={
        "LH_RH": ("hindL", "hindR"),      # Left-Right hind
        "LH_LF": ("hindL", "foreL"),       # Ipsilateral left
        "RH_RF": ("hindR", "foreR"),       # Ipsilateral right
        "LF_RH": ("foreL", "hindR"),       # Diagonal
        "RF_LH": ("foreR", "hindL"),       # Diagonal
        "LF_RF": ("foreL", "foreR"),       # Left-Right fore
    },
    speed_markers=["snout", "torso", "torsoL", "torsoR", "tail"]
)
```

### Step 5: Run Analysis

```python
# Create pipeline
pipeline = LocomotorPipeline(config={
    'smoothing_factor': 10,
    'speed_threshold': 5.0,
    'interpolation_factor': 4
})

# Get model name from tracking data
model_name = tracks.columns.get_level_values(0)[0]

# Process tracks
results = pipeline.process_tracks(
    tracks=tracks,
    metadata=metadata,
    model_name=model_name,
    markers=markers.markers,
    limb_pairs=markers.limb_pairs,
    speed_markers=markers.speed_markers
)

# Print summary
print(f"\n=== Analysis Summary ===")
print(f"Duration: {results['summary']['duration']:.1f} s")
print(f"Mean Speed: {results['summary']['mean_speed_cm_s']:.2f} cm/s")
print(f"Mean Cadence: {results['summary']['mean_cadence_hz']:.2f} Hz")
print(f"Coordination (R): {results['summary']['mean_coordination_R']:.3f}")
```

### Step 6: Export Results

```python
# Export to JSON (full results)
pipeline.export_results("results/video_01_analysis.json", format="json")

# Export to CSV (summary only)
pipeline.export_results("results/video_01_summary.csv", format="csv")
```

## Configuring Markers

### Standard Marker Sets

#### Ventral View (Bottom Camera)

```
         snoutL  snout  snoutR
              \   |   /
               \  |  /
                \ | /
        foreL ----+---- foreR
                  |
       torsoL --torso-- torsoR
                  |
        hindL ----+---- hindR
                  |
                 tail
```

```python
MOUSE_VENTRAL = MarkerSet(
    name="mouse_ventral",
    markers=[
        "snout", "snoutL", "snoutR",
        "foreL", "foreR",
        "hindL", "hindR",
        "torso", "torsoL", "torsoR",
        "tail"
    ],
    limb_pairs={
        "LH_RH": ("hindL", "hindR"),
        "LH_LF": ("hindL", "foreL"),
        "RH_RF": ("hindR", "foreR"),
        "LF_RH": ("foreL", "hindR"),
        "RF_LH": ("foreR", "hindL"),
        "LF_RF": ("foreL", "foreR"),
    },
    speed_markers=["snout", "torso", "torsoL", "torsoR", "tail"]
)
```

#### Lateral View (Side Camera)

```
     crest
       |
      hip
       |
     knee
       |
     ankle
       |
      foot
       |
      toe
```

```python
MOUSE_LATERAL = MarkerSet(
    name="mouse_lateral",
    markers=["toe", "foot", "ankle", "knee", "hip", "crest"],
    limb_pairs={},  # Joint angles instead of limb pairs
    speed_markers=["hip", "crest"]
)
```

### Custom Marker Configuration

```python
# Create custom marker set
custom_markers = MarkerSet(
    name="custom_experiment",
    markers=["head", "body", "leftPaw", "rightPaw", "tailBase"],
    limb_pairs={
        "left_right": ("leftPaw", "rightPaw"),
    },
    speed_markers=["head", "body", "tailBase"]
)
```

### Loading from YAML

```yaml
# config/markers_custom.yaml
name: custom_markers
markers:
  - head
  - body
  - leftPaw
  - rightPaw
  - tailBase
limb_pairs:
  left_right:
    - leftPaw
    - rightPaw
speed_markers:
  - head
  - body
```

```python
from mouse_locomotor_tracker.tracking import load_marker_config
markers = load_marker_config("config/markers_custom.yaml")
```

## Understanding Results

### Result Structure

```python
results = {
    'metadata': {
        'dur': 30.0,           # Duration (seconds)
        'fps': 100,            # Frames per second
        'nFrame': 3000,        # Total frames
        'imW': 640,            # Image width
        'imH': 480,            # Image height
        'xPixW': 0.3125        # Pixel width (mm)
    },
    'velocity': {
        'mean_speed': 15.2,    # Mean speed (cm/s)
        'max_speed': 45.8,     # Maximum speed (cm/s)
        'min_speed': 0.0,      # Minimum speed (cm/s)
        'std_speed': 8.3,      # Speed standard deviation
        'speed_profile': [...]  # Speed time series
    },
    'coordination': {
        'LH_RH': {
            'R': 0.92,              # Resultant length (0-1)
            'mean_phase_deg': 175.3, # Mean phase (degrees)
            'n_steps': 45           # Number of steps
        },
        # ... other limb pairs
    },
    'gait_cycles': {
        'hindL': {
            'cadence': 4.2,         # Steps per second (Hz)
            'stride_length': 3.6,   # Stride length (cm)
            'n_cycles': 126         # Total cycles detected
        },
        # ... other limbs
    },
    'summary': {
        'duration': 30.0,
        'mean_speed_cm_s': 15.2,
        'mean_coordination_R': 0.85,
        'mean_cadence_hz': 4.1,
        'mean_stride_length_cm': 3.7
    }
}
```

### Accessing Specific Results

```python
# Velocity
mean_speed = results['velocity']['mean_speed']
speed_profile = results['velocity']['speed_profile']

# Coordination for specific limb pair
lh_rh_R = results['coordination']['LH_RH']['R']
lh_rh_phase = results['coordination']['LH_RH']['mean_phase_deg']

# Gait metrics for specific limb
hindL_cadence = results['gait_cycles']['hindL']['cadence']
hindL_stride = results['gait_cycles']['hindL']['stride_length']

# Summary statistics
summary = results['summary']
```

## Interpreting Metrics

### Velocity Metrics

| Metric | Description | Typical Range | Notes |
|--------|-------------|---------------|-------|
| Mean Speed | Average locomotion speed | 5-30 cm/s | Depends on paradigm |
| Max Speed | Peak instantaneous speed | 30-80 cm/s | During acceleration |
| Std Speed | Speed variability | 3-15 cm/s | Higher = more variable |

### Coordination Metrics

#### Resultant Length (R)

| R Value | Interpretation |
|---------|----------------|
| 0.9-1.0 | Strong coordination |
| 0.7-0.9 | Good coordination |
| 0.5-0.7 | Moderate coordination |
| 0.3-0.5 | Weak coordination |
| 0.0-0.3 | No coordination |

#### Phase Angle

| Phase | Pattern | Description |
|-------|---------|-------------|
| 0 deg | In-phase | Limbs move together |
| 180 deg | Anti-phase | Limbs alternate |
| 90 deg | Quarter phase | One limb leads by 1/4 cycle |

#### Gait Patterns

```
TROT (alternating diagonal):
    LH_RH: ~180 deg (alternating)
    LF_RH: ~0 deg (synchronized)
    RF_LH: ~0 deg (synchronized)

PACE (ipsilateral):
    LH_LF: ~0 deg (synchronized)
    RH_RF: ~0 deg (synchronized)
    LH_RH: ~180 deg (alternating)

BOUND/GALLOP:
    LH_RH: ~0 deg (synchronized)
    LF_RF: ~0 deg (synchronized)
    Front-hind: ~180 deg
```

### Gait Cycle Metrics

| Metric | Description | Typical Range |
|--------|-------------|---------------|
| Cadence | Steps per second | 2-8 Hz |
| Stride Length | Distance per step | 2-6 cm |
| Cycle Duration | Time per cycle | 0.1-0.5 s |

## Advanced Usage

### Batch Processing

```python
import os
from pathlib import Path

def process_batch(data_dir: str, output_dir: str):
    """Process all tracking files in a directory."""

    data_path = Path(data_dir)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Find all HDF5 tracking files
    tracking_files = list(data_path.glob("*DLC*.h5"))

    results_list = []

    for track_file in tracking_files:
        print(f"Processing: {track_file.name}")

        # Load tracking data
        tracks = pd.read_hdf(track_file)
        model_name = tracks.columns.get_level_values(0)[0]

        # Find corresponding video
        video_name = track_file.stem.split("DLC")[0] + ".avi"
        video_path = data_path / video_name

        # Get metadata
        if video_path.exists():
            metadata = extract_video_metadata(str(video_path))
        else:
            # Use defaults
            metadata = VideoMetadata(
                duration=len(tracks) / 100,
                fps=100,
                n_frames=len(tracks),
                width=640,
                height=480,
                pixel_width_mm=0.3125
            )

        # Process
        pipeline = LocomotorPipeline()
        results = pipeline.process_tracks(
            tracks=tracks,
            metadata=metadata,
            model_name=model_name,
            markers=MOUSE_VENTRAL.markers,
            limb_pairs=MOUSE_VENTRAL.limb_pairs,
            speed_markers=MOUSE_VENTRAL.speed_markers
        )

        # Add filename to results
        results['filename'] = track_file.name
        results_list.append(results)

        # Export individual results
        output_file = output_path / f"{track_file.stem}_analysis.json"
        pipeline.export_results(str(output_file), format="json")

    # Create aggregate summary
    summaries = [r['summary'] | {'filename': r['filename']} for r in results_list]
    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(output_path / "batch_summary.csv", index=False)

    return results_list

# Run batch processing
results = process_batch("data/tracking/", "results/")
```

### Custom Analysis Pipeline

```python
from mouse_locomotor_tracker.analysis import (
    VelocityAnalyzer,
    CircularCoordinationAnalyzer,
    GaitCycleDetector
)

class CustomAnalysisPipeline:
    """Custom pipeline with specialized analysis."""

    def __init__(self, config: dict = None):
        self.config = config or {}

        # Initialize analyzers
        self.velocity_analyzer = VelocityAnalyzer(
            smoothing_factor=self.config.get('smoothing_factor', 10),
            speed_threshold=self.config.get('speed_threshold', 5.0)
        )
        self.coord_analyzer = CircularCoordinationAnalyzer(
            interpolation_factor=self.config.get('interpolation_factor', 4)
        )
        self.gait_detector = GaitCycleDetector()

    def analyze_episode(self, tracks, metadata, start_frame, end_frame):
        """Analyze a specific episode within the recording."""

        # Extract episode
        episode_tracks = tracks.iloc[start_frame:end_frame]
        episode_duration = (end_frame - start_frame) / metadata.fps

        # Run analysis on episode
        # ... custom analysis code

        return episode_results

    def detect_movement_bouts(self, speed_profile, threshold=5.0):
        """Detect movement bouts based on speed threshold."""

        moving = speed_profile > threshold

        bouts = []
        in_bout = False
        bout_start = 0

        for i, is_moving in enumerate(moving):
            if is_moving and not in_bout:
                bout_start = i
                in_bout = True
            elif not is_moving and in_bout:
                bouts.append((bout_start, i))
                in_bout = False

        if in_bout:
            bouts.append((bout_start, len(moving)))

        return bouts
```

### Visualization

```python
import matplotlib.pyplot as plt
import numpy as np

def plot_analysis_results(results: dict, output_path: str = None):
    """Create comprehensive visualization of analysis results."""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Speed profile
    ax1 = axes[0, 0]
    speed = results['velocity'].get('speed_profile', [])
    if speed:
        ax1.plot(speed[:500], 'b-', alpha=0.7)
        ax1.axhline(results['velocity']['mean_speed'], color='r', linestyle='--',
                    label=f"Mean: {results['velocity']['mean_speed']:.1f} cm/s")
        ax1.set_xlabel('Frame')
        ax1.set_ylabel('Speed (cm/s)')
        ax1.set_title('Speed Profile')
        ax1.legend()

    # 2. Coordination polar plot
    ax2 = axes[0, 1]
    ax2 = plt.subplot(222, projection='polar')

    for pair_name, data in results['coordination'].items():
        angle = np.deg2rad(data['mean_phase_deg'])
        r = data['R']
        ax2.annotate('', xy=(angle, r), xytext=(0, 0),
                    arrowprops=dict(arrowstyle='->', color='blue', lw=2))
        ax2.text(angle, r + 0.1, pair_name, fontsize=8, ha='center')

    ax2.set_title('Limb Coordination')
    ax2.set_ylim(0, 1.2)

    # 3. Cadence comparison
    ax3 = axes[1, 0]
    limbs = list(results['gait_cycles'].keys())
    cadences = [results['gait_cycles'][l]['cadence'] for l in limbs]
    ax3.bar(limbs, cadences, color='steelblue')
    ax3.set_ylabel('Cadence (Hz)')
    ax3.set_title('Limb Cadence')
    ax3.tick_params(axis='x', rotation=45)

    # 4. Stride length comparison
    ax4 = axes[1, 1]
    stride_lengths = [results['gait_cycles'][l]['stride_length'] for l in limbs]
    ax4.bar(limbs, stride_lengths, color='coral')
    ax4.set_ylabel('Stride Length (cm)')
    ax4.set_title('Stride Length')
    ax4.tick_params(axis='x', rotation=45)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

# Usage
plot_analysis_results(results, "results/analysis_plot.png")
```

## Examples

### Example 1: Basic Analysis

```python
"""Basic locomotion analysis example."""

import pandas as pd
from mouse_locomotor_tracker import LocomotorPipeline
from mouse_locomotor_tracker.tracking import VideoMetadata, MOUSE_VENTRAL

# Load data
tracks = pd.read_hdf("data/example_tracking.h5")
model_name = tracks.columns.get_level_values(0)[0]

# Configure
metadata = VideoMetadata(
    duration=30.0, fps=100, n_frames=3000,
    width=640, height=480, pixel_width_mm=0.3125
)

# Analyze
pipeline = LocomotorPipeline()
results = pipeline.process_tracks(
    tracks, metadata, model_name,
    MOUSE_VENTRAL.markers,
    MOUSE_VENTRAL.limb_pairs,
    MOUSE_VENTRAL.speed_markers
)

# Report
print(f"Speed: {results['summary']['mean_speed_cm_s']:.1f} cm/s")
print(f"Cadence: {results['summary']['mean_cadence_hz']:.1f} Hz")
print(f"Coordination: {results['summary']['mean_coordination_R']:.2f}")
```

### Example 2: Comparing Groups

```python
"""Compare locomotion between experimental groups."""

import pandas as pd
from scipy import stats

def compare_groups(control_files: list, treatment_files: list):
    """Compare locomotion metrics between control and treatment groups."""

    control_results = [analyze_file(f) for f in control_files]
    treatment_results = [analyze_file(f) for f in treatment_files]

    # Extract metrics
    control_speeds = [r['summary']['mean_speed_cm_s'] for r in control_results]
    treatment_speeds = [r['summary']['mean_speed_cm_s'] for r in treatment_results]

    control_cadences = [r['summary']['mean_cadence_hz'] for r in control_results]
    treatment_cadences = [r['summary']['mean_cadence_hz'] for r in treatment_results]

    # Statistical comparison
    speed_stat, speed_p = stats.ttest_ind(control_speeds, treatment_speeds)
    cadence_stat, cadence_p = stats.ttest_ind(control_cadences, treatment_cadences)

    print("=== Group Comparison ===")
    print(f"\nSpeed (cm/s):")
    print(f"  Control: {np.mean(control_speeds):.2f} +/- {np.std(control_speeds):.2f}")
    print(f"  Treatment: {np.mean(treatment_speeds):.2f} +/- {np.std(treatment_speeds):.2f}")
    print(f"  p-value: {speed_p:.4f}")

    print(f"\nCadence (Hz):")
    print(f"  Control: {np.mean(control_cadences):.2f} +/- {np.std(control_cadences):.2f}")
    print(f"  Treatment: {np.mean(treatment_cadences):.2f} +/- {np.std(treatment_cadences):.2f}")
    print(f"  p-value: {cadence_p:.4f}")
```

### Example 3: Time-Series Analysis

```python
"""Analyze locomotion changes over time within a session."""

def analyze_time_windows(results: dict, window_size: int = 500):
    """Analyze metrics in time windows."""

    speed_profile = results['velocity']['speed_profile']

    windows = []
    for i in range(0, len(speed_profile), window_size):
        window = speed_profile[i:i+window_size]
        if len(window) > 100:  # Minimum window size
            windows.append({
                'start_frame': i,
                'end_frame': i + len(window),
                'mean_speed': np.mean(window),
                'std_speed': np.std(window),
                'max_speed': np.max(window)
            })

    return pd.DataFrame(windows)

# Usage
time_analysis = analyze_time_windows(results)
print(time_analysis)
```

## Next Steps

- Read the [API Reference](API.md) for detailed class documentation
- See [Metrics Guide](METRICS.md) for mathematical details
- Check [Examples](../examples/) for more use cases
