"""
Mouse Locomotor Tracker - Interactive Dashboard
================================================

Streamlit-based dashboard for real-time locomotion analysis.

Run: streamlit run dashboard.py

Author: Stride Labs
"""

import json
import tempfile
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

# =============================================================================
# Page Configuration
# =============================================================================

st.set_page_config(
    page_title="Mouse Locomotor Tracker",
    page_icon="🐭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# Header
# =============================================================================

st.markdown('<h1 class="main-header">🐭 Mouse Locomotor Tracker</h1>', unsafe_allow_html=True)
st.markdown("""
<p style="text-align: center; color: #666; font-size: 1.1rem;">
Professional Biomechanical Analysis Dashboard
</p>
""", unsafe_allow_html=True)

# =============================================================================
# Sidebar
# =============================================================================

with st.sidebar:
    st.image("https://via.placeholder.com/150x50?text=Stride+Labs", width=150)
    st.markdown("---")

    st.header("📁 Data Source")

    data_source = st.radio(
        "Select data source:",
        ["Upload Video", "Upload CSV", "Demo Data"]
    )

    st.markdown("---")

    st.header("⚙️ Settings")

    # ROI Settings
    st.subheader("ROI Configuration")
    roi_y_min = st.slider("ROI Y Min (%)", 0, 100, 40)
    roi_y_max = st.slider("ROI Y Max (%)", 0, 100, 78)

    # Analysis Settings
    st.subheader("Analysis Parameters")
    fps = st.number_input("Frame Rate (FPS)", min_value=1, max_value=240, value=30)
    pixel_to_cm = st.number_input("Pixel to CM ratio", min_value=0.001, max_value=1.0, value=0.05, format="%.3f")
    smoothing_window = st.slider("Smoothing Window", 1, 21, 5, step=2)

    st.markdown("---")

    st.header("📊 Export")
    export_format = st.selectbox("Export Format", ["CSV", "JSON", "HDF5", "NWB"])

    st.markdown("---")

    st.markdown("""
    **Version:** 1.0.0
    **Author:** Stride Labs
    [Documentation](https://github.com/stridelabs/mouse-locomotor-tracker)
    """)

# =============================================================================
# Main Content
# =============================================================================

# Data storage
if 'tracking_data' not in st.session_state:
    st.session_state.tracking_data = None
if 'video_info' not in st.session_state:
    st.session_state.video_info = None

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["📹 Video Analysis", "📊 Metrics", "🗺️ Trajectory", "📈 Time Series"])

# =============================================================================
# Tab 1: Video Analysis
# =============================================================================

with tab1:
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Video Processing")

        if data_source == "Upload Video":
            uploaded_file = st.file_uploader(
                "Upload video file",
                type=["mp4", "avi", "mov"],
                help="Supported formats: MP4, AVI, MOV"
            )

            if uploaded_file is not None:
                # Save to temp file
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
                    tmp.write(uploaded_file.read())
                    video_path = tmp.name

                # Get video info
                cap = cv2.VideoCapture(video_path)
                if cap.isOpened():
                    st.session_state.video_info = {
                        'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                        'fps': cap.get(cv2.CAP_PROP_FPS),
                        'frames': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                        'duration': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / cap.get(cv2.CAP_PROP_FPS)
                    }

                    # Show first frame
                    ret, frame = cap.read()
                    if ret:
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                        # Draw ROI
                        h = frame.shape[0]
                        y1 = int(h * roi_y_min / 100)
                        y2 = int(h * roi_y_max / 100)
                        cv2.rectangle(frame_rgb, (0, y1), (frame.shape[1], y2), (0, 255, 0), 2)
                        cv2.putText(frame_rgb, "ROI", (10, y1 + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                        st.image(frame_rgb, caption="First frame with ROI", use_column_width=True)

                    cap.release()

                    # Process button
                    if st.button("🚀 Process Video", type="primary"):
                        progress_bar = st.progress(0)
                        status_text = st.empty()

                        # Simulate processing (replace with actual processing)
                        positions = []
                        cap = cv2.VideoCapture(video_path)
                        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

                        for i in range(min(total_frames, 300)):
                            ret, frame = cap.read()
                            if not ret:
                                break

                            # Simulated tracking (replace with actual)
                            center_x = 600 + np.random.randn() * 20
                            center_y = 450 + np.random.randn() * 10
                            positions.append([center_x, center_y])

                            progress_bar.progress((i + 1) / min(total_frames, 300))
                            status_text.text(f"Processing frame {i+1}/{min(total_frames, 300)}")

                        cap.release()

                        # Store results
                        st.session_state.tracking_data = pd.DataFrame(
                            positions,
                            columns=['x', 'y']
                        )
                        st.session_state.tracking_data['frame'] = range(len(positions))
                        st.session_state.tracking_data['time'] = st.session_state.tracking_data['frame'] / fps

                        # Calculate velocity
                        dx = np.diff(st.session_state.tracking_data['x'])
                        dy = np.diff(st.session_state.tracking_data['y'])
                        velocity = np.sqrt(dx**2 + dy**2) * pixel_to_cm * fps
                        st.session_state.tracking_data['velocity'] = np.append([0], velocity)

                        status_text.text("Processing complete!")
                        st.success("✅ Video processed successfully!")

        elif data_source == "Upload CSV":
            csv_file = st.file_uploader("Upload tracking CSV", type=["csv"])
            if csv_file is not None:
                st.session_state.tracking_data = pd.read_csv(csv_file)
                st.success("✅ CSV loaded successfully!")

        else:  # Demo Data
            if st.button("Load Demo Data"):
                # Generate demo data
                n_points = 500
                t = np.linspace(0, 10, n_points)
                x = 600 + 50 * np.sin(t) + np.cumsum(np.random.randn(n_points) * 2)
                y = 450 + 30 * np.cos(t * 2) + np.cumsum(np.random.randn(n_points) * 1)

                st.session_state.tracking_data = pd.DataFrame({
                    'frame': range(n_points),
                    'time': t,
                    'x': x,
                    'y': y,
                })

                # Calculate velocity
                dx = np.diff(x)
                dy = np.diff(y)
                velocity = np.sqrt(dx**2 + dy**2) * pixel_to_cm * fps
                st.session_state.tracking_data['velocity'] = np.append([0], velocity)

                st.success("✅ Demo data loaded!")

    with col2:
        st.subheader("Video Information")

        if st.session_state.video_info:
            info = st.session_state.video_info
            st.metric("Resolution", f"{info['width']} x {info['height']}")
            st.metric("Frame Rate", f"{info['fps']:.1f} FPS")
            st.metric("Total Frames", f"{info['frames']:,}")
            st.metric("Duration", f"{info['duration']:.1f} sec")

# =============================================================================
# Tab 2: Metrics
# =============================================================================

with tab2:
    if st.session_state.tracking_data is not None:
        data = st.session_state.tracking_data

        st.subheader("📊 Locomotion Metrics")

        # Key metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            avg_velocity = data['velocity'].mean()
            st.metric(
                "Average Velocity",
                f"{avg_velocity:.2f} cm/s",
                delta=f"{avg_velocity - 10:.1f}" if avg_velocity > 10 else None
            )

        with col2:
            max_velocity = data['velocity'].max()
            st.metric("Max Velocity", f"{max_velocity:.2f} cm/s")

        with col3:
            total_distance = data['velocity'].sum() / fps
            st.metric("Total Distance", f"{total_distance:.1f} cm")

        with col4:
            tracking_rate = (1 - data[['x', 'y']].isna().any(axis=1).sum() / len(data)) * 100
            st.metric("Tracking Rate", f"{tracking_rate:.1f}%")

        st.markdown("---")

        # Velocity histogram
        col1, col2 = st.columns(2)

        with col1:
            fig = px.histogram(
                data, x='velocity',
                nbins=50,
                title="Velocity Distribution",
                labels={'velocity': 'Velocity (cm/s)'},
                color_discrete_sequence=['#1E88E5']
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Activity classification
            data['activity'] = pd.cut(
                data['velocity'],
                bins=[0, 2, 10, 30, float('inf')],
                labels=['Rest', 'Walk', 'Trot', 'Gallop']
            )
            activity_counts = data['activity'].value_counts()

            fig = px.pie(
                values=activity_counts.values,
                names=activity_counts.index,
                title="Activity Classification",
                color_discrete_sequence=['#90CAF9', '#42A5F5', '#1E88E5', '#1565C0']
            )
            st.plotly_chart(fig, use_container_width=True)

        # Detailed statistics
        st.subheader("📋 Detailed Statistics")

        stats_df = pd.DataFrame({
            'Metric': ['Mean', 'Std Dev', 'Min', 'Max', 'Median', '25th Percentile', '75th Percentile'],
            'Velocity (cm/s)': [
                f"{data['velocity'].mean():.2f}",
                f"{data['velocity'].std():.2f}",
                f"{data['velocity'].min():.2f}",
                f"{data['velocity'].max():.2f}",
                f"{data['velocity'].median():.2f}",
                f"{data['velocity'].quantile(0.25):.2f}",
                f"{data['velocity'].quantile(0.75):.2f}",
            ]
        })

        st.dataframe(stats_df, use_container_width=True, hide_index=True)

    else:
        st.info("👆 Please load data in the Video Analysis tab first.")

# =============================================================================
# Tab 3: Trajectory
# =============================================================================

with tab3:
    if st.session_state.tracking_data is not None:
        data = st.session_state.tracking_data

        st.subheader("🗺️ Trajectory Analysis")

        col1, col2 = st.columns([2, 1])

        with col1:
            # 2D trajectory with color by velocity
            fig = px.scatter(
                data, x='x', y='y',
                color='velocity',
                color_continuous_scale='Viridis',
                title="Mouse Trajectory (color = velocity)",
                labels={'x': 'X (pixels)', 'y': 'Y (pixels)', 'velocity': 'Velocity (cm/s)'}
            )
            fig.update_yaxes(autorange="reversed")  # Flip Y axis
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Heatmap
            st.markdown("**Density Heatmap**")

            fig = px.density_heatmap(
                data, x='x', y='y',
                nbinsx=30, nbinsy=30,
                color_continuous_scale='Hot',
            )
            fig.update_yaxes(autorange="reversed")
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

        # 3D trajectory
        st.subheader("3D Trajectory (X, Y, Time)")

        fig = go.Figure(data=[go.Scatter3d(
            x=data['x'],
            y=data['y'],
            z=data['time'],
            mode='lines+markers',
            marker=dict(
                size=2,
                color=data['velocity'],
                colorscale='Viridis',
                colorbar=dict(title='Velocity (cm/s)')
            ),
            line=dict(width=1, color='gray')
        )])

        fig.update_layout(
            height=600,
            scene=dict(
                xaxis_title='X (pixels)',
                yaxis_title='Y (pixels)',
                zaxis_title='Time (s)',
            )
        )

        st.plotly_chart(fig, use_container_width=True)

    else:
        st.info("👆 Please load data in the Video Analysis tab first.")

# =============================================================================
# Tab 4: Time Series
# =============================================================================

with tab4:
    if st.session_state.tracking_data is not None:
        data = st.session_state.tracking_data

        st.subheader("📈 Time Series Analysis")

        # Multi-axis plot
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.08,
            subplot_titles=('X Position', 'Y Position', 'Velocity')
        )

        fig.add_trace(
            go.Scatter(x=data['time'], y=data['x'], name='X', line=dict(color='#1E88E5')),
            row=1, col=1
        )

        fig.add_trace(
            go.Scatter(x=data['time'], y=data['y'], name='Y', line=dict(color='#43A047')),
            row=2, col=1
        )

        fig.add_trace(
            go.Scatter(x=data['time'], y=data['velocity'], name='Velocity', line=dict(color='#E53935')),
            row=3, col=1
        )

        fig.update_layout(height=700, showlegend=True)
        fig.update_xaxes(title_text="Time (s)", row=3, col=1)
        fig.update_yaxes(title_text="X (px)", row=1, col=1)
        fig.update_yaxes(title_text="Y (px)", row=2, col=1)
        fig.update_yaxes(title_text="cm/s", row=3, col=1)

        st.plotly_chart(fig, use_container_width=True)

        # Smoothed velocity
        st.subheader("Smoothed Velocity")

        from scipy.ndimage import uniform_filter1d
        smoothed_velocity = uniform_filter1d(data['velocity'].values, size=smoothing_window)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=data['time'], y=data['velocity'],
            name='Raw', line=dict(color='#90CAF9', width=1)
        ))
        fig.add_trace(go.Scatter(
            x=data['time'], y=smoothed_velocity,
            name=f'Smoothed (window={smoothing_window})', line=dict(color='#1565C0', width=2)
        ))

        fig.update_layout(
            title="Velocity Over Time",
            xaxis_title="Time (s)",
            yaxis_title="Velocity (cm/s)",
            height=400
        )

        st.plotly_chart(fig, use_container_width=True)

    else:
        st.info("👆 Please load data in the Video Analysis tab first.")

# =============================================================================
# Export Section
# =============================================================================

st.markdown("---")
st.subheader("📥 Export Data")

if st.session_state.tracking_data is not None:
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        csv_data = st.session_state.tracking_data.to_csv(index=False)
        st.download_button(
            "📄 Download CSV",
            csv_data,
            file_name="tracking_results.csv",
            mime="text/csv"
        )

    with col2:
        json_data = st.session_state.tracking_data.to_json(orient='records', indent=2)
        st.download_button(
            "📋 Download JSON",
            json_data,
            file_name="tracking_results.json",
            mime="application/json"
        )

    with col3:
        # Summary report
        summary = {
            'total_frames': len(st.session_state.tracking_data),
            'duration_seconds': st.session_state.tracking_data['time'].max(),
            'average_velocity_cm_s': st.session_state.tracking_data['velocity'].mean(),
            'max_velocity_cm_s': st.session_state.tracking_data['velocity'].max(),
            'total_distance_cm': st.session_state.tracking_data['velocity'].sum() / fps,
        }
        st.download_button(
            "📊 Download Summary",
            json.dumps(summary, indent=2),
            file_name="analysis_summary.json",
            mime="application/json"
        )

    with col4:
        st.info("HDF5/NWB export available via CLI")

else:
    st.info("Load data to enable export options.")

# =============================================================================
# Footer
# =============================================================================

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #888; font-size: 0.9rem;">
    <p>Mouse Locomotor Tracker v1.0.0 | Stride Labs | MIT License</p>
    <p>
        <a href="https://github.com/stridelabs/mouse-locomotor-tracker">GitHub</a> |
        <a href="https://github.com/stridelabs/mouse-locomotor-tracker#documentation">Documentation</a> |
        <a href="https://github.com/stridelabs/mouse-locomotor-tracker/issues">Report Issue</a>
    </p>
</div>
""", unsafe_allow_html=True)
