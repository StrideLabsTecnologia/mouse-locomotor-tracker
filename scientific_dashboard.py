"""
Scientific Locomotor Analysis Dashboard v3.0 - TOP 0.1%
=========================================================

Professional dashboard with REAL-TIME animated mouse tracking visualization
and synchronized scientific analysis plots.

Inspired by:
- Kiehn Lab Locomotor-Allodi2021
- EstimAI_ project features

Features:
- Real-time animated dorsal view of mouse tracking points
- Speed Profile with variability band (mean +/- std)
- Acceleration Profile with drag/recovery events
- Cadence Profile showing stride patterns
- Left-Right Alternation Profile (polar plot)
- DeepLabCut integration

Run: streamlit run scientific_dashboard.py

Author: Stride Labs
License: MIT
"""

import json
import time
import base64
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass, field
import io

try:
    import cv2
except ImportError:
    cv2 = None
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import streamlit.components.v1 as components
import base64
from scipy.signal import find_peaks
from scipy.ndimage import uniform_filter1d

# =============================================================================
# Page Configuration
# =============================================================================

st.set_page_config(
    page_title="Scientific Locomotor Analysis - Stride Labs",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# Professional CSS - Matching S3 Reference Design
# =============================================================================

st.markdown("""
<!-- Google Fonts -->
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">

<style>
    /* === ROOT VARIABLES === */
    :root {
        --bg-primary: #0a0e1a;
        --bg-secondary: #0f1629;
        --bg-card: #141b2d;
        --bg-card-hover: #1a2340;
        --border-color: rgba(59, 130, 246, 0.15);
        --border-hover: rgba(59, 130, 246, 0.3);
        --text-primary: #ffffff;
        --text-secondary: #94a3b8;
        --text-muted: #64748b;
        --accent-blue: #60a5fa;
        --accent-cyan: #22d3ee;
        --accent-green: #10b981;
        --accent-orange: #f59e0b;
        --accent-red: #ef4444;
        --accent-purple: #a78bfa;
    }

    /* === BASE RESET === */
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
    }

    /* === MAIN APP BACKGROUND === */
    .stApp {
        background: linear-gradient(180deg, var(--bg-primary) 0%, var(--bg-secondary) 100%) !important;
    }

    .stApp > header {
        background: transparent !important;
    }

    /* === SIDEBAR === */
    [data-testid="stSidebar"] {
        background: var(--bg-primary) !important;
        border-right: 1px solid var(--border-color) !important;
    }

    [data-testid="stSidebar"] > div:first-child {
        background: transparent !important;
        padding-top: 1rem !important;
    }

    /* Sidebar text */
    [data-testid="stSidebar"] .stMarkdown p,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] .stTextInput label,
    [data-testid="stSidebar"] span {
        color: var(--text-secondary) !important;
    }

    /* Sidebar headers */
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: var(--accent-blue) !important;
        font-weight: 600 !important;
        font-size: 0.85rem !important;
        text-transform: uppercase !important;
        letter-spacing: 0.05em !important;
        margin-bottom: 0.75rem !important;
    }

    /* === HIDE STREAMLIT BRANDING === */
    #MainMenu, footer, header[data-testid="stHeader"] {
        visibility: hidden !important;
        height: 0 !important;
    }

    /* === MAIN CONTENT AREA === */
    .main .block-container {
        padding: 1rem 2rem 2rem 2rem !important;
        max-width: 100% !important;
    }

    /* === TYPOGRAPHY === */
    .main-header {
        font-size: 1.75rem !important;
        font-weight: 700 !important;
        background: linear-gradient(90deg, var(--accent-blue), var(--accent-purple)) !important;
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
        background-clip: text !important;
        text-align: center !important;
        margin: 0.5rem 0 1.5rem 0 !important;
        letter-spacing: -0.02em !important;
    }

    .section-header {
        font-size: 0.9rem !important;
        font-weight: 600 !important;
        color: var(--accent-blue) !important;
        margin: 0.75rem 0 0.5rem 0 !important;
        display: flex !important;
        align-items: center !important;
        gap: 0.5rem !important;
    }

    /* === RADIO BUTTONS (Data Source) === */
    [data-testid="stRadio"] > label {
        color: var(--accent-blue) !important;
        font-weight: 600 !important;
        font-size: 0.8rem !important;
        text-transform: uppercase !important;
        letter-spacing: 0.03em !important;
    }

    [data-testid="stRadio"] > div {
        gap: 0.25rem !important;
    }

    [data-testid="stRadio"] > div > label {
        background: transparent !important;
        border: none !important;
        padding: 0.4rem 0 !important;
    }

    [data-testid="stRadio"] > div > label > div:first-child {
        background: var(--bg-card) !important;
        border: 1px solid var(--border-color) !important;
    }

    [data-testid="stRadio"] > div > label[data-checked="true"] > div:first-child {
        background: var(--accent-blue) !important;
        border-color: var(--accent-blue) !important;
    }

    [data-testid="stRadio"] > div > label > div:last-child p {
        color: var(--text-secondary) !important;
        font-size: 0.85rem !important;
    }

    /* === SLIDERS === */
    [data-testid="stSlider"] > label {
        color: var(--text-muted) !important;
        font-size: 0.75rem !important;
        text-transform: uppercase !important;
    }

    [data-testid="stSlider"] [data-baseweb="slider"] {
        margin-top: 0.5rem !important;
    }

    [data-testid="stSlider"] [data-testid="stThumbValue"] {
        color: var(--accent-blue) !important;
        font-weight: 600 !important;
    }

    .stSlider > div > div > div > div {
        background: linear-gradient(90deg, var(--accent-blue), var(--accent-cyan)) !important;
    }

    /* === BUTTONS === */
    .stButton > button {
        background: linear-gradient(135deg, #1e40af 0%, var(--accent-blue) 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.6rem 1rem !important;
        font-weight: 600 !important;
        font-size: 0.85rem !important;
        transition: all 0.2s ease !important;
        box-shadow: 0 2px 8px rgba(59, 130, 246, 0.3) !important;
        width: 100% !important;
    }

    .stButton > button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 16px rgba(59, 130, 246, 0.4) !important;
        background: linear-gradient(135deg, #2563eb 0%, #3b82f6 100%) !important;
    }

    /* === METRICS === */
    [data-testid="stMetric"] {
        background: var(--bg-card) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 12px !important;
        padding: 1rem 1.25rem !important;
    }

    [data-testid="stMetricLabel"] {
        color: var(--text-muted) !important;
        font-size: 0.7rem !important;
        text-transform: uppercase !important;
        letter-spacing: 0.05em !important;
        font-weight: 500 !important;
    }

    [data-testid="stMetricValue"] {
        color: var(--text-primary) !important;
        font-size: 1.5rem !important;
        font-weight: 700 !important;
        letter-spacing: -0.02em !important;
    }

    /* === TEXT INPUTS === */
    .stTextInput > div > div > input {
        background: var(--bg-card) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 8px !important;
        color: var(--text-primary) !important;
        padding: 0.6rem 0.75rem !important;
    }

    .stTextInput > div > div > input:focus {
        border-color: var(--accent-blue) !important;
        box-shadow: 0 0 0 2px rgba(96, 165, 250, 0.2) !important;
    }

    /* === NUMBER INPUTS === */
    .stNumberInput > div > div > input {
        background: var(--bg-card) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 8px !important;
        color: var(--text-primary) !important;
    }

    .stNumberInput button {
        background: var(--bg-card-hover) !important;
        border-color: var(--border-color) !important;
        color: var(--text-secondary) !important;
    }

    /* === SELECT BOX === */
    .stSelectbox > div > div {
        background: var(--bg-card) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 8px !important;
    }

    /* === PROGRESS BAR === */
    .stProgress > div > div {
        background: var(--bg-card) !important;
        border-radius: 4px !important;
    }

    .stProgress > div > div > div {
        background: linear-gradient(90deg, var(--accent-blue), var(--accent-cyan)) !important;
        border-radius: 4px !important;
    }

    /* === EXPANDERS === */
    .streamlit-expanderHeader {
        background: var(--bg-card) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 8px !important;
        color: var(--text-primary) !important;
    }

    /* === METRIC CARDS ROW === */
    .metric-card {
        background: var(--bg-card) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 12px !important;
        padding: 1rem 1.5rem !important;
        transition: all 0.2s ease !important;
    }

    .metric-card:hover {
        border-color: var(--border-hover) !important;
        background: var(--bg-card-hover) !important;
    }

    .metric-label {
        color: var(--text-muted) !important;
        font-size: 0.7rem !important;
        text-transform: uppercase !important;
        letter-spacing: 0.05em !important;
        font-weight: 500 !important;
        margin-bottom: 0.25rem !important;
    }

    .metric-value {
        color: var(--text-primary) !important;
        font-size: 1.5rem !important;
        font-weight: 700 !important;
        letter-spacing: -0.02em !important;
    }

    /* === CHART CONTAINERS === */
    .chart-container {
        background: var(--bg-card) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 12px !important;
        padding: 1rem !important;
        margin-bottom: 1rem !important;
    }

    /* === VIDEO PLAYER CONTAINER === */
    .video-container-wrapper {
        background: var(--bg-card) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 12px !important;
        overflow: hidden !important;
    }

    /* === TABS === */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem !important;
        background: transparent !important;
    }

    .stTabs [data-baseweb="tab"] {
        background: var(--bg-card) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 8px !important;
        color: var(--text-secondary) !important;
        padding: 0.5rem 1rem !important;
    }

    .stTabs [aria-selected="true"] {
        background: var(--accent-blue) !important;
        border-color: var(--accent-blue) !important;
        color: white !important;
    }

    /* === DIVIDERS === */
    hr {
        border-color: var(--border-color) !important;
        margin: 1rem 0 !important;
    }

    /* === PLOTLY CHARTS DARK THEME === */
    .js-plotly-plot .plotly .main-svg {
        background: transparent !important;
    }

    .js-plotly-plot .plotly .bg {
        fill: var(--bg-card) !important;
    }

    /* === SUCCESS/INFO BOXES === */
    .success-box {
        background: rgba(16, 185, 129, 0.1) !important;
        border: 1px solid rgba(16, 185, 129, 0.3) !important;
        border-radius: 8px !important;
        padding: 0.75rem 1rem !important;
        color: var(--accent-green) !important;
        font-weight: 500 !important;
    }

    .info-box {
        background: rgba(96, 165, 250, 0.1) !important;
        border: 1px solid rgba(96, 165, 250, 0.3) !important;
        border-radius: 8px !important;
        padding: 0.75rem 1rem !important;
        color: var(--accent-blue) !important;
    }

    /* === SIDEBAR METRICS === */
    .sidebar-metric {
        background: var(--bg-card) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 8px !important;
        padding: 0.75rem !important;
        text-align: center !important;
    }

    .sidebar-metric-label {
        color: var(--text-muted) !important;
        font-size: 0.65rem !important;
        text-transform: uppercase !important;
        letter-spacing: 0.03em !important;
    }

    .sidebar-metric-value {
        color: var(--text-primary) !important;
        font-size: 1.1rem !important;
        font-weight: 700 !important;
    }

    /* === LOGO AREA === */
    .logo-container {
        display: flex !important;
        align-items: center !important;
        gap: 0.5rem !important;
        padding: 0.5rem 0 1rem 0 !important;
        border-bottom: 1px solid var(--border-color) !important;
        margin-bottom: 1rem !important;
    }

    .logo-icon {
        width: 24px !important;
        height: 24px !important;
        background: linear-gradient(135deg, var(--accent-blue), var(--accent-purple)) !important;
        border-radius: 6px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
    }

    .logo-text {
        color: var(--text-secondary) !important;
        font-size: 0.9rem !important;
        font-weight: 600 !important;
    }

    /* === FOOTER === */
    .app-footer {
        text-align: center !important;
        color: var(--text-muted) !important;
        font-size: 0.75rem !important;
        padding: 1rem 0 !important;
        border-top: 1px solid var(--border-color) !important;
        margin-top: 2rem !important;
    }

    .app-footer a {
        color: var(--text-muted) !important;
        text-decoration: none !important;
    }

    .app-footer a:hover {
        color: var(--accent-blue) !important;
    }

    /* === REMOVE DEFAULT STREAMLIT PADDING === */
    .element-container {
        margin-bottom: 0.5rem !important;
    }

    /* === FILE UPLOADER === */
    [data-testid="stFileUploader"] {
        background: var(--bg-card) !important;
        border: 1px dashed var(--border-color) !important;
        border-radius: 8px !important;
        padding: 1rem !important;
    }

    [data-testid="stFileUploader"]:hover {
        border-color: var(--accent-blue) !important;
    }

    /* === SPINNER === */
    .stSpinner > div {
        border-top-color: var(--accent-blue) !important;
    }

    /* === CAPTIONS === */
    .stCaption {
        color: var(--text-muted) !important;
        font-size: 0.75rem !important;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class MouseMarkerColors:
    """Color scheme for mouse body markers (ventral view)."""
    snout: str = "#00ff00"      # Green - nose
    snoutL: str = "#00dd00"     # Green - left snout
    snoutR: str = "#00dd00"     # Green - right snout
    foreL: str = "#ff6600"      # Orange - left forepaw
    foreR: str = "#0066ff"      # Blue - right forepaw
    hindL: str = "#ff0000"      # Red - left hindpaw
    hindR: str = "#00ffff"      # Cyan - right hindpaw
    torso: str = "#ffff00"      # Yellow - center torso
    torsoL: str = "#ffaa00"     # Light orange - left torso
    torsoR: str = "#00aaff"     # Light blue - right torso
    tail: str = "#ff00ff"       # Magenta - tail base

    def get_color(self, marker: str) -> str:
        return getattr(self, marker, "#ffffff")

    @property
    def all_markers(self) -> List[str]:
        return ['snout', 'snoutL', 'snoutR', 'foreL', 'foreR',
                'hindL', 'hindR', 'torso', 'torsoL', 'torsoR', 'tail']

    def to_list(self) -> List[str]:
        return [self.get_color(m) for m in self.all_markers]


@dataclass
class AnalysisMetrics:
    """Container for computed analysis metrics."""
    avg_speed: float = 0.0
    max_speed: float = 0.0
    avg_acceleration: float = 0.0
    peak_acceleration: float = 0.0
    cadence_hz: float = 0.0
    n_cycles: int = 0
    r_value: float = 0.0
    mean_phase_deg: float = 0.0
    duration_s: float = 0.0


# =============================================================================
# Cached Video Loading (prevents re-encoding on every rerun)
# =============================================================================

@st.cache_data(show_spinner=False)
def load_video_base64(video_path: str) -> str:
    """Load and cache video as base64 string."""
    with open(video_path, 'rb') as f:
        video_bytes = f.read()
    return base64.b64encode(video_bytes).decode()


# =============================================================================
# Locomotion Analysis Functions (Based on Locomotor-Allodi2021)
# =============================================================================

def calculate_speed_profile(
    positions: np.ndarray,
    fps: float,
    belt_speed_cms: float,
    smooth_factor: int = 10,
    pixel_to_cm: float = 0.05
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate speed profile from position data.

    Based on Locomotor-Allodi2021/bottom/profiler.py methodology:
    - Compute frame-to-frame displacement
    - Convert pixels to cm/s
    - Add belt speed contribution
    - Apply smoothing filter
    """
    if len(positions) < 2:
        return np.array([belt_speed_cms]), np.array([0.0])

    # Calculate displacement
    dx = np.diff(positions[:, 0])
    dy = np.diff(positions[:, 1])
    displacement = np.sqrt(dx**2 + dy**2) * pixel_to_cm

    # Convert to velocity (cm/s)
    velocity = displacement * fps

    # Add belt speed (mouse speed = body velocity + belt contribution)
    # Negative because mouse moves opposite to belt direction
    velocity = belt_speed_cms - velocity
    velocity = np.maximum(velocity, 0)  # Clip negative values

    # Smooth the velocity
    if len(velocity) > smooth_factor:
        velocity_smooth = uniform_filter1d(velocity, size=smooth_factor)
    else:
        velocity_smooth = velocity

    # Calculate standard deviation for variability band
    velocity_std = pd.Series(velocity).rolling(
        window=max(10, smooth_factor),
        min_periods=1
    ).std().fillna(0).values

    return velocity_smooth, velocity_std


def calculate_acceleration_profile(
    velocity: np.ndarray,
    fps: float,
    smooth_factor: int = 12
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate acceleration profile from velocity data.

    Based on Locomotor-Allodi2021 - separates drag and recovery phases.
    """
    if len(velocity) < 2:
        return np.array([0.0]), np.array([0.0]), np.array([0.0])

    # Derivative of velocity
    acceleration = np.diff(velocity) * fps

    # Smooth
    if len(acceleration) > smooth_factor:
        acceleration_smooth = uniform_filter1d(acceleration, size=smooth_factor)
    else:
        acceleration_smooth = acceleration

    # Separate positive (recovery) and negative (drag) phases
    recovery = np.maximum(acceleration_smooth, 0)
    drag = np.minimum(acceleration_smooth, 0)

    return acceleration_smooth, recovery, drag


def calculate_stride_metrics(
    hind_left_x: np.ndarray,
    hind_right_x: np.ndarray,
    body_pos_x: np.ndarray,
    fps: float
) -> Tuple[np.ndarray, np.ndarray, List[int], float]:
    """
    Calculate stride and cadence metrics.

    Based on Locomotor-Allodi2021 stride analysis:
    - Stride = difference between left and right hindpaw positions
    - Cadence = number of complete cycles per second
    """
    # Relative stride to body position
    stride_left = hind_left_x - body_pos_x
    stride_right = hind_right_x - body_pos_x

    # Stride difference (alternation pattern)
    stride_diff = stride_left - stride_right

    # Smooth
    stride_smooth = uniform_filter1d(stride_diff, size=5)

    # Detect cycle peaks
    mean_val = np.mean(stride_smooth)
    peaks, _ = find_peaks(stride_smooth - mean_val, distance=int(fps * 0.1))

    # Calculate cadence (Hz)
    if len(peaks) >= 2:
        avg_period_frames = np.mean(np.diff(peaks))
        cadence_hz = fps / avg_period_frames
    else:
        cadence_hz = 0.0

    return stride_smooth, stride_diff, list(peaks), cadence_hz


def calculate_circular_coordination(
    stride_diff: np.ndarray,
    cycle_indices: List[int]
) -> Tuple[np.ndarray, float, float]:
    """
    Calculate circular coordination metrics (polar plot data).

    Based on Locomotor-Allodi2021 heurCircular function:
    - Phase angle for each complete cycle
    - R-value (vector strength) indicating coordination consistency
    - Mean phase angle
    """
    if len(cycle_indices) < 2:
        return np.array([np.pi]), 0.5, 180.0

    phases = []

    for i in range(len(cycle_indices) - 1):
        start_idx = cycle_indices[i]
        end_idx = cycle_indices[i + 1]

        if end_idx > len(stride_diff):
            break

        # Get cycle segment
        cycle_data = stride_diff[start_idx:end_idx]

        if len(cycle_data) < 2:
            continue

        # Normalize cycle to [-1, 1]
        cycle_min, cycle_max = cycle_data.min(), cycle_data.max()
        if cycle_max - cycle_min > 0:
            cycle_norm = 2 * (cycle_data - cycle_min) / (cycle_max - cycle_min) - 1
        else:
            cycle_norm = np.zeros_like(cycle_data)

        # Create x-axis for integration (0 to 2*pi)
        x_axis = np.linspace(0, 2 * np.pi, len(cycle_norm))

        # Integrate to get phase (based on Allodi2021 formula)
        integral = np.trapz(cycle_norm, x_axis)
        phase = (4 - integral) * np.pi / 4

        # Normalize to [-pi, pi]
        phase = np.arctan2(np.sin(phase), np.cos(phase))
        phases.append(phase)

    if len(phases) == 0:
        return np.array([np.pi]), 0.5, 180.0

    phases = np.array(phases)

    # Calculate circular mean and R-value
    cos_sum = np.sum(np.cos(phases))
    sin_sum = np.sum(np.sin(phases))
    n = len(phases)

    mean_angle = np.arctan2(sin_sum, cos_sum)
    r_value = np.sqrt(cos_sum**2 + sin_sum**2) / n

    mean_phase_deg = np.degrees(mean_angle)
    if mean_phase_deg < 0:
        mean_phase_deg += 360

    return phases, r_value, mean_phase_deg


# =============================================================================
# Synthetic Data Generation
# =============================================================================

def generate_synthetic_tracking_data(
    n_frames: int = 500,
    fps: int = 30,
    belt_speed_cms: float = 20.0
) -> pd.DataFrame:
    """
    Generate realistic synthetic mouse tracking data for demonstration.
    Simulates ventral view locomotion on a treadmill.
    """
    np.random.seed(42)
    t = np.linspace(0, n_frames / fps, n_frames)

    # Base position (center of treadmill)
    center_x, center_y = 400, 300

    # Slight drift and oscillation
    drift_x = np.cumsum(np.random.randn(n_frames) * 0.3)
    drift_x = drift_x - np.mean(drift_x)  # Center it
    drift_y = np.cumsum(np.random.randn(n_frames) * 0.2)
    drift_y = drift_y - np.mean(drift_y)

    base_x = center_x + drift_x
    base_y = center_y + drift_y

    # Body orientation oscillation
    body_angle = 0.03 * np.sin(2 * np.pi * t * 0.4)

    # Gait parameters
    gait_freq = 3.5  # Hz
    gait_phase = 2 * np.pi * gait_freq * t

    data = {
        'frame': np.arange(n_frames),
        'time': t,
    }

    # Torso (center reference)
    data['torso_x'] = base_x
    data['torso_y'] = base_y

    # Snout (head, forward)
    snout_offset = 75
    data['snout_x'] = base_x + snout_offset * np.cos(body_angle)
    data['snout_y'] = base_y - snout_offset * np.sin(body_angle)

    data['snoutL_x'] = data['snout_x'] - 12 * np.sin(body_angle)
    data['snoutL_y'] = data['snout_y'] - 12 * np.cos(body_angle)

    data['snoutR_x'] = data['snout_x'] + 12 * np.sin(body_angle)
    data['snoutR_y'] = data['snout_y'] + 12 * np.cos(body_angle)

    # Torso sides
    torso_width = 22
    data['torsoL_x'] = base_x - torso_width * np.sin(body_angle)
    data['torsoL_y'] = base_y - torso_width * np.cos(body_angle)

    data['torsoR_x'] = base_x + torso_width * np.sin(body_angle)
    data['torsoR_y'] = base_y + torso_width * np.cos(body_angle)

    # Tail (behind)
    tail_offset = 65
    data['tail_x'] = base_x - tail_offset * np.cos(body_angle)
    data['tail_y'] = base_y + tail_offset * np.sin(body_angle)

    # Forepaws - alternating gait pattern
    fore_amp_x = 22
    fore_amp_y = 12
    fore_base_offset = 35

    data['foreL_x'] = data['torsoL_x'] + fore_base_offset + fore_amp_x * np.sin(gait_phase)
    data['foreL_y'] = data['torsoL_y'] + fore_amp_y * np.cos(gait_phase)

    data['foreR_x'] = data['torsoR_x'] + fore_base_offset + fore_amp_x * np.sin(gait_phase + np.pi)
    data['foreR_y'] = data['torsoR_y'] + fore_amp_y * np.cos(gait_phase + np.pi)

    # Hindpaws - alternating (phase offset from forepaws)
    hind_amp_x = 28
    hind_amp_y = 18
    hind_lateral_offset = 25

    data['hindL_x'] = data['tail_x'] - hind_lateral_offset * np.sin(body_angle) + hind_amp_x * np.sin(gait_phase + np.pi)
    data['hindL_y'] = data['tail_y'] - hind_lateral_offset * np.cos(body_angle) + hind_amp_y * np.cos(gait_phase + np.pi)

    data['hindR_x'] = data['tail_x'] + hind_lateral_offset * np.sin(body_angle) + hind_amp_x * np.sin(gait_phase)
    data['hindR_y'] = data['tail_y'] + hind_lateral_offset * np.cos(body_angle) + hind_amp_y * np.cos(gait_phase)

    # Add realistic noise
    noise_level = 1.2
    markers = ['snout', 'snoutL', 'snoutR', 'foreL', 'foreR',
               'hindL', 'hindR', 'torso', 'torsoL', 'torsoR', 'tail']
    for marker in markers:
        data[f'{marker}_x'] += np.random.randn(n_frames) * noise_level
        data[f'{marker}_y'] += np.random.randn(n_frames) * noise_level

    df = pd.DataFrame(data)

    # Pre-compute analysis metrics
    positions = np.column_stack([df['torso_x'].values, df['torso_y'].values])

    # Speed
    velocity, velocity_std = calculate_speed_profile(positions, fps, belt_speed_cms)
    df['velocity'] = np.concatenate([[velocity[0]], velocity])[:n_frames]
    df['velocity_std'] = np.concatenate([[velocity_std[0]], velocity_std])[:n_frames]

    # Acceleration
    accel, recovery, drag = calculate_acceleration_profile(df['velocity'].values, fps)
    df['acceleration'] = np.concatenate([[0], accel])[:n_frames]
    df['recovery'] = np.concatenate([[0], recovery])[:n_frames]
    df['drag'] = np.concatenate([[0], drag])[:n_frames]

    # Stride metrics
    stride_smooth, stride_diff, peaks, cadence = calculate_stride_metrics(
        df['hindL_x'].values, df['hindR_x'].values, df['torso_x'].values, fps
    )
    df['stride_diff'] = stride_diff[:n_frames] if len(stride_diff) >= n_frames else np.concatenate([stride_diff, [stride_diff[-1]] * (n_frames - len(stride_diff))])
    df['stride_smooth'] = stride_smooth[:n_frames] if len(stride_smooth) >= n_frames else np.concatenate([stride_smooth, [stride_smooth[-1]] * (n_frames - len(stride_smooth))])

    # Store cycle peaks for polar plot
    df.attrs['cycle_peaks'] = peaks
    df.attrs['cadence_hz'] = cadence
    df.attrs['fps'] = fps
    df.attrs['belt_speed'] = belt_speed_cms

    return df


# =============================================================================
# Visualization Functions
# =============================================================================

def create_synced_video_player(
    video_path: str,
    current_time: float,
    total_duration: float,
    fps: int,
    playing: bool,
    width: int = 450,
    height: int = 300
) -> str:
    """Create synchronized video player with HTML5 and JavaScript.

    This component syncs video playback with the current frame time.
    """
    # Read video file and encode to base64
    try:
        with open(video_path, 'rb') as f:
            video_data = base64.b64encode(f.read()).decode('utf-8')
    except FileNotFoundError:
        return "<p style='color: #ff4444;'>Video not found</p>"

    # JavaScript to sync video time with Streamlit frame
    html_content = f"""
    <div id="video-container" style="
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
        border-radius: 12px;
        padding: 10px;
        border: 1px solid #3b82f6;
    ">
        <video
            id="synced-video"
            width="{width}"
            height="{height - 50}"
            style="
                border-radius: 8px;
                display: block;
                margin: 0 auto;
            "
            preload="auto"
        >
            <source src="data:video/mp4;base64,{video_data}" type="video/mp4">
            Your browser does not support the video tag.
        </video>
        <div id="video-controls" style="
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 15px;
            margin-top: 10px;
            padding: 8px;
            background: rgba(30, 58, 95, 0.8);
            border-radius: 8px;
        ">
            <span id="time-display" style="
                color: #60a5fa;
                font-family: monospace;
                font-size: 14px;
                min-width: 100px;
            ">
                {current_time:.2f}s / {total_duration:.2f}s
            </span>
            <div style="
                flex: 1;
                height: 6px;
                background: #1e3a5f;
                border-radius: 3px;
                position: relative;
                cursor: pointer;
            " id="progress-bar">
                <div id="progress-fill" style="
                    width: {(current_time / max(total_duration, 0.01)) * 100}%;
                    height: 100%;
                    background: linear-gradient(90deg, #3b82f6, #60a5fa);
                    border-radius: 3px;
                    transition: width 0.1s;
                "></div>
                <div id="progress-indicator" style="
                    position: absolute;
                    left: {(current_time / max(total_duration, 0.01)) * 100}%;
                    top: 50%;
                    transform: translate(-50%, -50%);
                    width: 14px;
                    height: 14px;
                    background: #ff4444;
                    border: 2px solid white;
                    border-radius: 50%;
                    box-shadow: 0 0 6px rgba(255, 68, 68, 0.6);
                "></div>
            </div>
        </div>
    </div>
    <script>
        (function() {{
            const video = document.getElementById('synced-video');
            const timeDisplay = document.getElementById('time-display');
            const progressFill = document.getElementById('progress-fill');
            const progressIndicator = document.getElementById('progress-indicator');
            const targetTime = {current_time};
            const totalDuration = {total_duration};
            const isPlaying = {"true" if playing else "false"};

            // Set video to current time
            video.addEventListener('loadedmetadata', function() {{
                video.currentTime = targetTime;
                if (isPlaying) {{
                    video.play().catch(e => console.log('Autoplay prevented:', e));
                }}
            }});

            // Update time display during playback
            video.addEventListener('timeupdate', function() {{
                const t = video.currentTime;
                timeDisplay.textContent = t.toFixed(2) + 's / ' + totalDuration.toFixed(2) + 's';
                const pct = (t / totalDuration) * 100;
                progressFill.style.width = pct + '%';
                progressIndicator.style.left = pct + '%';
            }});

            // If not playing, ensure video is at correct time
            if (!isPlaying) {{
                video.pause();
                video.currentTime = targetTime;
            }}
        }})();
    </script>
    """
    return html_content


def create_animated_mouse_figure(
    df: pd.DataFrame,
    colors: MouseMarkerColors,
    width: int = 900,
    height: int = 280
) -> go.Figure:
    """Create fully animated Plotly figure with all frames (client-side animation)."""

    markers = colors.all_markers
    marker_colors = colors.to_list()
    marker_sizes = [20, 14, 14, 22, 22, 24, 24, 18, 16, 16, 17]

    # Create frames for animation
    frames = []
    for idx in range(len(df)):
        row = df.iloc[idx]
        x_coords = [row[f'{m}_x'] for m in markers]
        y_coords = [row[f'{m}_y'] for m in markers]

        frame_data = []
        for i, marker in enumerate(markers):
            frame_data.append(go.Scatter(
                x=[x_coords[i]],
                y=[y_coords[i]],
                mode='markers',
                marker=dict(
                    size=marker_sizes[i],
                    color=marker_colors[i],
                    line=dict(color='white', width=1.5)
                ),
                name=marker,
                showlegend=False
            ))

        frames.append(go.Frame(data=frame_data, name=str(idx)))

    # Initial frame
    row = df.iloc[0]
    x_coords = [row[f'{m}_x'] for m in markers]
    y_coords = [row[f'{m}_y'] for m in markers]

    # Calculate data ranges for proper axis limits
    all_x = []
    all_y = []
    for m in markers:
        all_x.extend(df[f'{m}_x'].values)
        all_y.extend(df[f'{m}_y'].values)

    x_min, x_max = min(all_x), max(all_x)
    y_min, y_max = min(all_y), max(all_y)
    x_padding = (x_max - x_min) * 0.15
    y_padding = (y_max - y_min) * 0.15

    fig = go.Figure()

    # Treadmill background - Blue theme (layer='below' so markers appear on top)
    fig.add_shape(
        type="rect",
        x0=x_min - x_padding, y0=y_min - y_padding,
        x1=x_max + x_padding, y1=y_max + y_padding,
        fillcolor="#1e3a5f",
        line=dict(color="#3b82f6", width=2),
        layer='below'
    )

    # Add markers
    for i, marker in enumerate(markers):
        fig.add_trace(go.Scatter(
            x=[x_coords[i]],
            y=[y_coords[i]],
            mode='markers',
            marker=dict(
                size=marker_sizes[i],
                color=marker_colors[i],
                line=dict(color='white', width=1.5)
            ),
            name=marker,
            showlegend=False
        ))

    # Add frames
    fig.frames = frames

    # Animation controls - Blue theme with dynamic ranges
    fig.update_layout(
        width=width,
        height=height,
        plot_bgcolor='#0f172a',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=10, r=10, t=5, b=60),
        xaxis=dict(range=[x_min - x_padding, x_max + x_padding], showgrid=False, showticklabels=False, zeroline=False),
        yaxis=dict(range=[y_min - y_padding, y_max + y_padding], showgrid=False, showticklabels=False, zeroline=False, scaleanchor='x'),
        updatemenus=[dict(
            type="buttons",
            showactive=False,
            y=-0.15,
            x=0.5,
            xanchor="center",
            buttons=[
                dict(label="▶ Play",
                     method="animate",
                     args=[None, {"frame": {"duration": 50, "redraw": True},
                                 "fromcurrent": True,
                                 "transition": {"duration": 0}}]),
                dict(label="⏸ Pause",
                     method="animate",
                     args=[[None], {"frame": {"duration": 0, "redraw": False},
                                   "mode": "immediate",
                                   "transition": {"duration": 0}}])
            ]
        )],
        sliders=[dict(
            active=0,
            yanchor="top",
            xanchor="left",
            currentvalue=dict(prefix="Frame: ", visible=True, xanchor="center"),
            transition=dict(duration=0),
            pad=dict(b=10, t=30),
            len=0.9,
            x=0.05,
            y=-0.02,
            steps=[dict(args=[[str(i)], {"frame": {"duration": 0, "redraw": True},
                                         "mode": "immediate",
                                         "transition": {"duration": 0}}],
                       method="animate",
                       label=str(i)) for i in range(0, len(df), max(1, len(df)//50))]
        )]
    )

    return fig


def create_mouse_animation_frame(
    df: pd.DataFrame,
    frame_idx: int,
    colors: MouseMarkerColors,
    width: int = 800,
    height: int = 400
) -> go.Figure:
    """Create single frame synchronized with playback controls.

    This version uses dynamic ranges based on actual data.
    """
    markers = colors.all_markers
    marker_colors = colors.to_list()
    marker_sizes = [20, 14, 14, 22, 22, 24, 24, 18, 16, 16, 17]

    # Calculate dynamic ranges from all data
    all_x = []
    all_y = []
    for m in markers:
        all_x.extend(df[f'{m}_x'].values)
        all_y.extend(df[f'{m}_y'].values)

    x_min, x_max = min(all_x), max(all_x)
    y_min, y_max = min(all_y), max(all_y)
    x_padding = (x_max - x_min) * 0.15
    y_padding = (y_max - y_min) * 0.15

    row = df.iloc[frame_idx]
    x_coords = [row[f'{m}_x'] for m in markers]
    y_coords = [row[f'{m}_y'] for m in markers]
    current_time = row['time']

    fig = go.Figure()

    # Treadmill belt background - Blue theme
    fig.add_shape(
        type="rect",
        x0=x_min - x_padding, y0=y_min - y_padding,
        x1=x_max + x_padding, y1=y_max + y_padding,
        fillcolor="#1e3a5f",
        line=dict(color="#3b82f6", width=2),
        layer="below"
    )

    # Frame info annotation
    fig.add_annotation(
        x=0.5, y=1.05,
        xref='paper', yref='paper',
        text=f'Frame: {frame_idx} | Time: {current_time:.2f}s',
        showarrow=False,
        font=dict(size=12, color='#60a5fa'),
        bgcolor='rgba(30, 58, 95, 0.8)',
        borderpad=4,
    )

    # Skeleton connections
    connections = [
        ('snoutL', 'snout'), ('snout', 'snoutR'),
        ('snout', 'torso'), ('torso', 'tail'),
        ('torsoL', 'torso'), ('torso', 'torsoR'),
        ('foreL', 'torsoL'), ('foreR', 'torsoR'),
        ('hindL', 'tail'), ('hindR', 'tail'),
        ('snoutL', 'torsoL'), ('snoutR', 'torsoR'),
    ]

    marker_to_idx = {m: i for i, m in enumerate(markers)}

    for c1, c2 in connections:
        i1, i2 = marker_to_idx[c1], marker_to_idx[c2]
        fig.add_trace(go.Scatter(
            x=[x_coords[i1], x_coords[i2]],
            y=[y_coords[i1], y_coords[i2]],
            mode='lines',
            line=dict(color='rgba(150,150,150,0.6)', width=2),
            showlegend=False,
            hoverinfo='skip'
        ))

    # Marker points
    for i, marker in enumerate(markers):
        fig.add_trace(go.Scatter(
            x=[x_coords[i]],
            y=[y_coords[i]],
            mode='markers',
            marker=dict(
                size=marker_sizes[i],
                color=marker_colors[i],
                line=dict(color='white', width=1.5),
                symbol='circle'
            ),
            name=marker,
            showlegend=False,
            hovertemplate=f"<b>{marker}</b><br>x: %{{x:.1f}}<br>y: %{{y:.1f}}<extra></extra>"
        ))

    fig.update_layout(
        width=width,
        height=height,
        plot_bgcolor='#0f172a',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=10, r=10, t=30, b=5),
        xaxis=dict(
            range=[x_min - x_padding, x_max + x_padding],
            showgrid=False,
            showticklabels=False,
            zeroline=False,
        ),
        yaxis=dict(
            range=[y_min - y_padding, y_max + y_padding],
            showgrid=False,
            showticklabels=False,
            zeroline=False,
            scaleanchor='x',
            scaleratio=1
        ),
        # Smooth transition to reduce flickering
        transition=dict(duration=50, easing='linear'),
    )

    return fig


def create_speed_profile_plot(
    df: pd.DataFrame,
    current_frame: int,
    belt_speed: float,
    width: int = 800,
    height: int = 320
) -> go.Figure:
    """Create Speed Profile with variability band."""

    fig = go.Figure()

    time = df['time'].values
    velocity = df['velocity'].values
    velocity_std = df['velocity_std'].values
    max_time = time[-1]

    # Full time range for reference lines
    time_range = [0, max_time]

    # Variability band (up to current frame)
    if current_frame > 1:
        t_slice = time[:current_frame]
        v_slice = velocity[:current_frame]
        v_std_slice = velocity_std[:current_frame]

        fig.add_trace(go.Scatter(
            x=np.concatenate([t_slice, t_slice[::-1]]),
            y=np.concatenate([v_slice - v_std_slice, (v_slice + v_std_slice)[::-1]]),
            fill='toself',
            fillcolor='rgba(100, 150, 200, 0.25)',
            line=dict(color='rgba(0,0,0,0)'),
            showlegend=True,
            name='±1 SD',
            hoverinfo='skip'
        ))

        # Main velocity line
        fig.add_trace(go.Scatter(
            x=t_slice,
            y=v_slice,
            mode='lines',
            line=dict(color='#4488dd', width=2.5),
            name='Avg. Speed',
            hovertemplate='%{y:.1f} cm/s<extra></extra>'
        ))

    # Belt speed reference (dashed red)
    fig.add_trace(go.Scatter(
        x=time_range,
        y=[belt_speed, belt_speed],
        mode='lines',
        line=dict(color='#dd4444', width=2, dash='dash'),
        name=f'Belt = {belt_speed:.0f} cm/s'
    ))

    # Current frame indicator - vertical line and marker
    if current_frame > 0:
        current_time = time[min(current_frame, len(time)-1)]
        current_vel = velocity[min(current_frame, len(velocity)-1)]

        # Vertical time indicator line
        fig.add_vline(
            x=current_time,
            line=dict(color='#ff4444', width=2, dash='solid'),
            annotation_text=f'{current_time:.2f}s',
            annotation_position='top',
            annotation_font=dict(size=10, color='#ff4444')
        )

        # Current point marker
        fig.add_trace(go.Scatter(
            x=[current_time],
            y=[current_vel],
            mode='markers',
            marker=dict(size=12, color='#ff0000', symbol='circle',
                       line=dict(color='white', width=2)),
            name='Current',
            showlegend=False
        ))

    fig.update_layout(
        title=dict(text='Speed Profile', font=dict(size=13, color='#94a3b8'), x=0.5),
        width=width,
        height=height,
        plot_bgcolor='#141b2d',
        paper_bgcolor='#141b2d',
        margin=dict(l=55, r=15, t=35, b=40),
        xaxis=dict(
            title=dict(text='Time (s)', font=dict(color='#64748b', size=10)),
            range=[0, max_time],
            gridcolor='rgba(59, 130, 246, 0.1)',
            showgrid=True,
            tickfont=dict(size=10, color='#64748b'),
            linecolor='rgba(59, 130, 246, 0.2)',
            zerolinecolor='rgba(59, 130, 246, 0.2)'
        ),
        yaxis=dict(
            title=dict(text='Speed (cm/s)', font=dict(color='#64748b', size=10)),
            range=[0, max(velocity.max() * 1.2, belt_speed * 1.5)],
            gridcolor='rgba(59, 130, 246, 0.1)',
            showgrid=True,
            tickfont=dict(size=10, color='#64748b'),
            linecolor='rgba(59, 130, 246, 0.2)',
            zerolinecolor='rgba(59, 130, 246, 0.2)'
        ),
        legend=dict(
            x=0.02, y=0.98,
            bgcolor='rgba(20, 27, 45, 0.9)',
            font=dict(size=9, color='#94a3b8'),
            bordercolor='rgba(59, 130, 246, 0.2)',
            borderwidth=1
        ),
        showlegend=True,
        transition=dict(duration=50, easing='linear'),
    )

    return fig


def create_acceleration_profile_plot(
    df: pd.DataFrame,
    current_frame: int,
    width: int = 800,
    height: int = 320
) -> go.Figure:
    """Create Acceleration Profile with drag/recovery phases."""

    fig = go.Figure()

    time = df['time'].values
    accel = df['acceleration'].values
    max_time = time[-1]

    # Zero line
    fig.add_trace(go.Scatter(
        x=[0, max_time],
        y=[0, 0],
        mode='lines',
        line=dict(color='#666666', width=1, dash='dash'),
        name='Zero',
        showlegend=False
    ))

    if current_frame > 1:
        t_slice = time[:current_frame]
        a_slice = accel[:current_frame]

        # Recovery (positive) - blue
        recovery = np.maximum(a_slice, 0)
        fig.add_trace(go.Scatter(
            x=t_slice,
            y=recovery,
            mode='lines',
            fill='tozeroy',
            fillcolor='rgba(70, 130, 200, 0.4)',
            line=dict(color='#4682c8', width=1.5),
            name='Recovery',
            hovertemplate='%{y:.2f} cm/s²<extra></extra>'
        ))

        # Drag (negative) - orange
        drag = np.minimum(a_slice, 0)
        fig.add_trace(go.Scatter(
            x=t_slice,
            y=drag,
            mode='lines',
            fill='tozeroy',
            fillcolor='rgba(230, 140, 60, 0.4)',
            line=dict(color='#e68c3c', width=1.5),
            name='Drag',
            hovertemplate='%{y:.2f} cm/s²<extra></extra>'
        ))

    # Peak marker
    if current_frame > 5:
        a_slice = accel[:current_frame]
        peak_idx = np.argmax(np.abs(a_slice))
        fig.add_trace(go.Scatter(
            x=[time[peak_idx]],
            y=[a_slice[peak_idx]],
            mode='markers',
            marker=dict(size=8, color='#cc0000', symbol='diamond'),
            name='Peak',
            showlegend=True
        ))

    # Current time vertical indicator
    if current_frame > 0:
        current_time = time[min(current_frame, len(time)-1)]
        fig.add_vline(
            x=current_time,
            line=dict(color='#ff4444', width=2, dash='solid'),
            annotation_text=f'{current_time:.2f}s',
            annotation_position='top',
            annotation_font=dict(size=10, color='#ff4444')
        )

    fig.update_layout(
        title=dict(text='Acceleration Profile', font=dict(size=13, color='#94a3b8'), x=0.5),
        width=width,
        height=height,
        plot_bgcolor='#141b2d',
        paper_bgcolor='#141b2d',
        margin=dict(l=55, r=15, t=35, b=40),
        xaxis=dict(
            title=dict(text='Time (s)', font=dict(color='#64748b', size=10)),
            range=[0, max_time],
            gridcolor='rgba(59, 130, 246, 0.1)',
            showgrid=True,
            tickfont=dict(size=10, color='#64748b'),
            linecolor='rgba(59, 130, 246, 0.2)',
            zerolinecolor='rgba(59, 130, 246, 0.2)',
        ),
        yaxis=dict(
            title=dict(text='Accel (cm/s²)', font=dict(color='#64748b', size=10)),
            gridcolor='rgba(59, 130, 246, 0.1)',
            showgrid=True,
            tickfont=dict(size=10, color='#64748b'),
            linecolor='rgba(59, 130, 246, 0.2)',
            zerolinecolor='rgba(59, 130, 246, 0.2)',
        ),
        legend=dict(
            x=0.65, y=0.98,
            bgcolor='rgba(20, 27, 45, 0.9)',
            font=dict(size=9, color='#94a3b8'),
            orientation='h',
            bordercolor='rgba(59, 130, 246, 0.2)',
            borderwidth=1,
        ),
        showlegend=True,
        transition=dict(duration=50, easing='linear'),
    )

    return fig


def create_cadence_profile_plot(
    df: pd.DataFrame,
    current_frame: int,
    width: int = 800,
    height: int = 320
) -> go.Figure:
    """Create Cadence Profile showing stride patterns."""

    fig = go.Figure()

    time = df['time'].values
    stride = df['stride_smooth'].values if 'stride_smooth' in df else df['stride_diff'].values
    max_time = time[-1]

    # Mean crossing line
    mean_val = np.mean(stride)
    fig.add_trace(go.Scatter(
        x=[0, max_time],
        y=[mean_val, mean_val],
        mode='lines',
        line=dict(color='#888888', width=1.5, dash='dash'),
        name='Mean'
    ))

    if current_frame > 1:
        t_slice = time[:current_frame]
        s_slice = stride[:current_frame]

        # Stride pattern
        fig.add_trace(go.Scatter(
            x=t_slice,
            y=s_slice,
            mode='lines',
            line=dict(color='#2288aa', width=2),
            name='Stride diff (L-R)',
            hovertemplate='%{y:.1f} px<extra></extra>'
        ))

        # Detect and mark cycles
        peaks, _ = find_peaks(s_slice - mean_val, distance=10)
        if len(peaks) > 0:
            fig.add_trace(go.Scatter(
                x=time[peaks],
                y=stride[peaks],
                mode='markers',
                marker=dict(size=8, color='#dd3333', symbol='circle'),
                name='Cycles'
            ))

    # Current time vertical indicator
    if current_frame > 0:
        current_time = time[min(current_frame, len(time)-1)]
        fig.add_vline(
            x=current_time,
            line=dict(color='#ff4444', width=2, dash='solid'),
            annotation_text=f'{current_time:.2f}s',
            annotation_position='top',
            annotation_font=dict(size=10, color='#ff4444')
        )

    fig.update_layout(
        title=dict(text='Cadence Profile', font=dict(size=13, color='#94a3b8'), x=0.5),
        width=width,
        height=height,
        plot_bgcolor='#141b2d',
        paper_bgcolor='#141b2d',
        margin=dict(l=55, r=15, t=35, b=40),
        xaxis=dict(
            title=dict(text='Time (s)', font=dict(color='#64748b', size=10)),
            range=[0, max_time],
            gridcolor='rgba(59, 130, 246, 0.1)',
            showgrid=True,
            tickfont=dict(size=10, color='#64748b'),
            linecolor='rgba(59, 130, 246, 0.2)',
            zerolinecolor='rgba(59, 130, 246, 0.2)',
        ),
        yaxis=dict(
            title=dict(text='Stride (px)', font=dict(color='#64748b', size=10)),
            gridcolor='rgba(59, 130, 246, 0.1)',
            showgrid=True,
            tickfont=dict(size=10, color='#64748b'),
            linecolor='rgba(59, 130, 246, 0.2)',
            zerolinecolor='rgba(59, 130, 246, 0.2)',
        ),
        legend=dict(
            x=0.02, y=0.98,
            bgcolor='rgba(20, 27, 45, 0.9)',
            font=dict(size=9, color='#94a3b8'),
            bordercolor='rgba(59, 130, 246, 0.2)',
            borderwidth=1,
        ),
        showlegend=True,
        transition=dict(duration=50, easing='linear'),
    )

    return fig


def create_polar_coordination_plot(
    df: pd.DataFrame,
    current_frame: int,
    width: int = 200,
    height: int = 200
) -> go.Figure:
    """Create Left-Right Alternation polar plot."""

    fig = go.Figure()

    stride_diff = df['stride_diff'].values[:current_frame] if current_frame > 0 else df['stride_diff'].values[:1]

    # Get cycle peaks
    cycle_peaks = df.attrs.get('cycle_peaks', [])
    valid_peaks = [p for p in cycle_peaks if p < current_frame]

    if len(valid_peaks) >= 2:
        # Calculate phases for each cycle
        phases, r_value, mean_phase = calculate_circular_coordination(stride_diff, valid_peaks)

        # Convert to degrees for plotting
        theta_deg = np.degrees(phases) % 360
        r_vals = np.ones(len(phases))

        # Previous points (blue circles)
        if len(phases) > 1:
            fig.add_trace(go.Scatterpolar(
                r=r_vals[:-1],
                theta=theta_deg[:-1],
                mode='markers',
                marker=dict(size=14, color='#4466bb', line=dict(color='white', width=1)),
                name='Cycles',
                showlegend=False
            ))

        # Current point (red)
        if len(phases) > 0:
            fig.add_trace(go.Scatterpolar(
                r=[r_vals[-1]],
                theta=[theta_deg[-1]],
                mode='markers',
                marker=dict(size=16, color='#dd2222', line=dict(color='white', width=2)),
                name='Current',
                showlegend=False
            ))

        # Mean vector (red arrow)
        fig.add_trace(go.Scatterpolar(
            r=[0, r_value],
            theta=[mean_phase, mean_phase],
            mode='lines',
            line=dict(color='#cc3333', width=4),
            showlegend=False
        ))

        # Arrow head
        fig.add_trace(go.Scatterpolar(
            r=[r_value],
            theta=[mean_phase],
            mode='markers',
            marker=dict(size=12, color='#cc3333', symbol='triangle-up'),
            showlegend=False
        ))
    else:
        # Show waiting state
        fig.add_annotation(
            x=0.5, y=0.5,
            text="Collecting cycles...",
            showarrow=False,
            font=dict(size=12, color='#666'),
            xref='paper', yref='paper'
        )

    fig.update_layout(
        title=dict(text='Left-Right Alternation', font=dict(size=13, color='#94a3b8'), x=0.5),
        width=width,
        height=height,
        paper_bgcolor='#141b2d',
        polar=dict(
            bgcolor='#0f1629',
            radialaxis=dict(
                visible=True,
                range=[0, 1.15],
                showline=False,
                tickmode='array',
                tickvals=[0.5, 1.0],
                gridcolor='rgba(59, 130, 246, 0.2)',
                tickfont=dict(size=9, color='#64748b')
            ),
            angularaxis=dict(
                tickmode='array',
                tickvals=[0, 45, 90, 135, 180, 225, 270, 315],
                ticktext=['0°', '', '90°', '', '180°', '', '270°', ''],
                direction='clockwise',
                rotation=90,
                gridcolor='rgba(59, 130, 246, 0.2)',
                tickfont=dict(size=9, color='#64748b'),
                linecolor='rgba(59, 130, 246, 0.3)',
            )
        ),
        margin=dict(l=30, r=30, t=40, b=30),
        showlegend=False,
        transition=dict(duration=50, easing='linear'),
    )

    return fig


# =============================================================================
# Main Dashboard
# =============================================================================

def main():
    """Main dashboard with real-time animation."""

    # Initialize session state
    if 'data' not in st.session_state:
        st.session_state.data = None
    if 'frame' not in st.session_state:
        st.session_state.frame = 0
    if 'playing' not in st.session_state:
        st.session_state.playing = False
    if 'play_start_time' not in st.session_state:
        st.session_state.play_start_time = None
    if 'play_start_frame' not in st.session_state:
        st.session_state.play_start_frame = 0
    if 'subject' not in st.session_state:
        st.session_state.subject = "SOD1 presymptomatic P49"
    if 'speed' not in st.session_state:
        st.session_state.speed = 1.0
    if 'fps_playback' not in st.session_state:
        st.session_state.fps_playback = 30

    # Sidebar
    with st.sidebar:
        # Logo
        st.markdown('''
        <div class="logo-container">
            <div class="logo-icon">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="white">
                    <path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5"/>
                </svg>
            </div>
            <span class="logo-text">Stride Labs</span>
        </div>
        ''', unsafe_allow_html=True)

        # Data source
        st.markdown('<p style="color: #60a5fa; font-size: 0.75rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.5rem;">📁 Data Source</p>', unsafe_allow_html=True)
        data_source = st.radio(
            "Select:",
            ["Demo Data", "Upload CSV", "Upload H5 (DLC)"],
            index=0,
            label_visibility="collapsed"
        )

        st.markdown("---")

        # Analysis settings
        st.markdown('<p style="color: #60a5fa; font-size: 0.75rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.5rem;">⚙️ Settings</p>', unsafe_allow_html=True)

        st.markdown('<p style="color: #64748b; font-size: 0.7rem; text-transform: uppercase; margin-bottom: 0.25rem;">Belt Speed (cm/s)</p>', unsafe_allow_html=True)
        belt_speed = st.slider("Belt Speed", 10.0, 40.0, 20.0, 1.0, label_visibility="collapsed")

        st.markdown('<p style="color: #64748b; font-size: 0.7rem; text-transform: uppercase; margin-bottom: 0.25rem; margin-top: 0.75rem;">FPS</p>', unsafe_allow_html=True)
        fps = st.number_input("FPS", 10, 120, 30, step=5, label_visibility="collapsed")

        st.markdown("---")

        # Playback controls - SYNCHRONIZED
        st.markdown('<p style="color: #60a5fa; font-size: 0.75rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.5rem;">▶️ PLAYBACK CONTROLS</p>', unsafe_allow_html=True)
        st.markdown('<p style="color: #64748b; font-size: 0.7rem; margin-bottom: 0.5rem;">Video & Graphs Synchronized</p>', unsafe_allow_html=True)

        if st.session_state.data is not None:
            n_frames = len(st.session_state.data)
            fps = st.session_state.fps_playback

            # Calculate current frame if playing
            if st.session_state.playing and st.session_state.play_start_time is not None:
                elapsed = time.time() - st.session_state.play_start_time
                elapsed_frames = int(elapsed * fps)
                new_frame = st.session_state.play_start_frame + elapsed_frames
                if new_frame >= n_frames:
                    new_frame = n_frames - 1
                    st.session_state.playing = False
                st.session_state.frame = new_frame

            # Big Play/Pause button
            col1, col2 = st.columns(2)
            with col1:
                if st.session_state.playing:
                    if st.button("⏸ PAUSE", use_container_width=True, type="secondary"):
                        st.session_state.playing = False
                        st.rerun()
                else:
                    if st.button("▶️ PLAY", use_container_width=True, type="primary"):
                        st.session_state.playing = True
                        st.session_state.play_start_time = time.time()
                        st.session_state.play_start_frame = st.session_state.frame
                        st.rerun()
            with col2:
                if st.button("⏹ STOP", use_container_width=True, type="secondary"):
                    st.session_state.playing = False
                    st.session_state.frame = 0
                    st.session_state.play_start_time = None
                    st.rerun()

            # Progress bar
            current_time = st.session_state.data.iloc[st.session_state.frame]['time']
            total_time = st.session_state.data['time'].max()
            st.progress(st.session_state.frame / max(1, n_frames - 1))

            # Time display
            st.markdown(f'''
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0.5rem; margin-top: 0.75rem;">
                <div class="sidebar-metric">
                    <div class="sidebar-metric-label">TIME</div>
                    <div class="sidebar-metric-value">{current_time:.2f}s</div>
                </div>
                <div class="sidebar-metric">
                    <div class="sidebar-metric-label">FRAME</div>
                    <div class="sidebar-metric-value">{st.session_state.frame}/{n_frames-1}</div>
                </div>
            </div>
            ''', unsafe_allow_html=True)

            # Status indicator
            if st.session_state.playing:
                st.markdown('''
                <div style="background: rgba(34, 197, 94, 0.2); border: 1px solid rgba(34, 197, 94, 0.5);
                            border-radius: 8px; padding: 8px; text-align: center; margin-top: 10px;">
                    <span style="color: #22c55e; font-weight: 600;">🔴 PLAYING...</span>
                </div>
                ''', unsafe_allow_html=True)

            # Polar plot in sidebar (fixed position)
            st.markdown('<p style="color: #64748b; font-size: 0.65rem; margin-top: 15px; margin-bottom: 5px; text-align: center;">L-R Alternation</p>', unsafe_allow_html=True)
            polar_fig = create_polar_coordination_plot(st.session_state.data, st.session_state.frame)
            st.plotly_chart(
                polar_fig,
                use_container_width=True,
                config={'displayModeBar': False, 'staticPlot': True},
                key="polar_sidebar"
            )

            # Manual seek slider (only when paused)
            if not st.session_state.playing:
                st.markdown('<p style="color: #64748b; font-size: 0.65rem; margin-top: 10px; margin-bottom: 5px;">Manual Seek:</p>', unsafe_allow_html=True)
                new_frame = st.slider(
                    "Seek",
                    0, n_frames - 1,
                    st.session_state.frame,
                    key="frame_slider",
                    label_visibility="collapsed"
                )
                if new_frame != st.session_state.frame:
                    st.session_state.frame = new_frame

        st.markdown("---")

        # Subject info
        st.markdown('<p style="color: #60a5fa; font-size: 0.75rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.5rem;">📊 Subject</p>', unsafe_allow_html=True)
        st.session_state.subject = st.text_input("Name", st.session_state.subject, label_visibility="collapsed")

        st.markdown("---")
        st.markdown('<p style="color: #64748b; font-size: 0.7rem; text-align: center;">v3.0 Scientific | Stride Labs</p>', unsafe_allow_html=True)

    # Main content
    st.markdown(f'<h1 class="main-header">Analysis for {st.session_state.subject}</h1>', unsafe_allow_html=True)

    # Load data
    if data_source == "Demo Data":
        if st.session_state.data is None:
            with st.spinner("Generating locomotion data..."):
                st.session_state.data = generate_synthetic_tracking_data(
                    n_frames=400, fps=fps, belt_speed_cms=belt_speed
                )
            st.markdown('<div class="success-box">✓ Demo data loaded successfully!</div>', unsafe_allow_html=True)
            time.sleep(0.5)
            st.rerun()

    elif data_source == "Upload CSV":
        uploaded = st.file_uploader("Upload tracking CSV", type=["csv"])
        if uploaded:
            st.session_state.data = pd.read_csv(uploaded)
            st.success("CSV loaded!")

    elif data_source == "Upload H5 (DLC)":
        st.info("Upload DeepLabCut H5 output file")
        uploaded = st.file_uploader("Upload H5 file", type=["h5"])
        if uploaded:
            try:
                df = pd.read_hdf(io.BytesIO(uploaded.read()))
                st.session_state.data = df
                st.success("H5 loaded!")
            except Exception as e:
                st.error(f"Error loading H5: {e}")

    # Display dashboard
    if st.session_state.data is not None:
        df = st.session_state.data
        colors = MouseMarkerColors()
        frame = st.session_state.frame  # Current frame for synchronization

        # Current time info
        current_time = df.iloc[frame]['time']
        total_duration = df['time'].max()
        is_playing = st.session_state.playing
        n_frames = len(df)

        # Big progress bar at top
        sync_status = "🔴 REPRODUCIENDO" if is_playing else "⏸ PAUSADO"
        sync_color = "#22c55e" if is_playing else "#f59e0b"
        progress_pct = (frame / max(1, n_frames - 1)) * 100

        st.markdown(f'''
        <div style="background: linear-gradient(135deg, rgba(59, 130, 246, 0.15), rgba(139, 92, 246, 0.1));
                    border: 1px solid rgba(59, 130, 246, 0.3); border-radius: 8px; padding: 8px 12px; margin-bottom: 10px;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                <span style="color: {sync_color}; font-weight: 700; font-size: 0.9rem;">{sync_status}</span>
                <span style="color: #e2e8f0; font-size: 0.85rem;">
                    <b style="color: #60a5fa;">{current_time:.2f}s</b> / {total_duration:.2f}s
                    <span style="color: #64748b; margin-left: 10px;">F:{frame}/{n_frames-1}</span>
                </span>
            </div>
            <div style="background: #1e293b; border-radius: 6px; height: 8px; overflow: hidden;">
                <div style="background: linear-gradient(90deg, #3b82f6, #8b5cf6); height: 100%; width: {progress_pct:.1f}%;
                            border-radius: 6px; transition: width 0.05s;"></div>
            </div>
        </div>
        ''', unsafe_allow_html=True)

        # Video + Mouse Animation SIDE BY SIDE
        video_frames_dir = Path("/tmp/claude/video_frames")
        video_col, mouse_col = st.columns([1, 1])

        with video_col:
            if video_frames_dir.exists():
                st.markdown('<p class="section-header">🎬 Video</p>', unsafe_allow_html=True)

                # Map data frame to video frame (video has 274 frames at 30fps = 9.13s)
                video_total_frames = 274
                data_duration = df['time'].max()

                # Calculate which video frame corresponds to current time
                video_frame_idx = int((current_time / data_duration) * (video_total_frames - 1))
                video_frame_idx = max(1, min(video_frame_idx + 1, video_total_frames))

                frame_path = video_frames_dir / f"frame_{video_frame_idx:04d}.jpg"
                if frame_path.exists():
                    st.image(str(frame_path), use_container_width=True)

        with mouse_col:
            st.markdown('<p class="section-header">🐭 Tracking</p>', unsafe_allow_html=True)
            mouse_fig = create_mouse_animation_frame(df, frame, colors, width=400, height=280)
            st.plotly_chart(
                mouse_fig,
                use_container_width=True,
                config={'displayModeBar': False},
                key="mouse_animation"
            )

        # Dynamic metrics - update with current frame
        current_speed = df.iloc[frame]['velocity']
        current_accel = df.iloc[frame]['acceleration']
        cadence = df.attrs.get('cadence_hz', 0)

        st.markdown(f'''
        <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 0.5rem; margin: 0.5rem 0;">
            <div class="metric-card" style="padding: 8px;">
                <div class="metric-label" style="font-size: 0.6rem;">SPEED</div>
                <div class="metric-value" style="font-size: 1rem; color: #60a5fa;">{current_speed:.1f} cm/s</div>
            </div>
            <div class="metric-card" style="padding: 8px;">
                <div class="metric-label" style="font-size: 0.6rem;">ACCEL</div>
                <div class="metric-value" style="font-size: 1rem; color: {'#22c55e' if current_accel >= 0 else '#f97316'};">{current_accel:+.2f} cm/s²</div>
            </div>
            <div class="metric-card" style="padding: 8px;">
                <div class="metric-label" style="font-size: 0.6rem;">TIME</div>
                <div class="metric-value" style="font-size: 1rem; color: #a78bfa;">{current_time:.2f} s</div>
            </div>
            <div class="metric-card" style="padding: 8px;">
                <div class="metric-label" style="font-size: 0.6rem;">CADENCE</div>
                <div class="metric-value" style="font-size: 1rem;">{cadence:.1f} Hz</div>
            </div>
        </div>
        ''', unsafe_allow_html=True)

        # Analysis plots - 3 graphs with reduced height for visibility
        st.markdown('<p class="section-header">📊 Analysis Profiles</p>', unsafe_allow_html=True)

        # Speed - compact height
        speed_fig = create_speed_profile_plot(df, frame, belt_speed, height=180)
        st.plotly_chart(speed_fig, use_container_width=True, config={'displayModeBar': False}, key="speed_plot")

        # Acceleration - compact height
        accel_fig = create_acceleration_profile_plot(df, frame, height=180)
        st.plotly_chart(accel_fig, use_container_width=True, config={'displayModeBar': False}, key="accel_plot")

        # Cadence - compact height
        cadence_fig = create_cadence_profile_plot(df, frame, height=180)
        st.plotly_chart(cadence_fig, use_container_width=True, config={'displayModeBar': False}, key="cadence_plot")

        # Export
        st.markdown("---")
        st.subheader("📥 Export")

        col1, col2, col3 = st.columns(3)
        with col1:
            csv = df.to_csv(index=False)
            st.download_button("📄 Download CSV", csv, "tracking_data.csv", "text/csv")

        with col2:
            summary = {
                'subject': st.session_state.subject,
                'frames': len(df),
                'duration_s': float(df['time'].max()),
                'belt_speed_cms': belt_speed,
                'fps': fps,
                'avg_speed': float(df['velocity'].mean()),
                'cadence_hz': df.attrs.get('cadence_hz', 0)
            }
            st.download_button("📊 Summary JSON", json.dumps(summary, indent=2), "summary.json")

        with col3:
            st.info("🎬 Use CLI for video export")

        # Auto-advance when playing
        if st.session_state.playing:
            time.sleep(0.1)  # 100ms delay for smooth animation
            st.rerun()

    else:
        st.info("👆 Select a data source to begin analysis")

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #555; font-size: 0.85rem;">
        Scientific Locomotor Analysis v3.0 | Stride Labs | MIT License<br>
        Based on <a href="https://github.com/kiehnlab/Locomotor-Allodi2021" style="color: #888;">Locomotor-Allodi2021</a>
        & <a href="https://github.com/MenaVhs/EstimAI_" style="color: #888;">EstimAI_</a>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
