# =============================================================================
# Mouse Locomotor Tracker - Dockerfile
# =============================================================================
# Multi-stage build for optimized production image
# =============================================================================

# -----------------------------------------------------------------------------
# Stage 1: Builder
# -----------------------------------------------------------------------------
FROM python:3.11-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
COPY pyproject.toml .

# Create virtual environment and install dependencies
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir -e ".[cli,export]"

# -----------------------------------------------------------------------------
# Stage 2: Runtime
# -----------------------------------------------------------------------------
FROM python:3.11-slim as runtime

LABEL maintainer="Stride Labs <dev@stridelabs.cl>"
LABEL description="Mouse Locomotor Tracker - Professional biomechanical analysis"
LABEL version="1.0.0"

# Install runtime dependencies (OpenCV, ffmpeg)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Set working directory
WORKDIR /app

# Copy application code
COPY . .

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash mlt && \
    chown -R mlt:mlt /app
USER mlt

# Create directories for input/output
RUN mkdir -p /app/input /app/output

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV MLT_OUTPUT_DIR=/app/output

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import cv2; import numpy; print('OK')" || exit 1

# Default command
ENTRYPOINT ["python", "cli.py"]
CMD ["--help"]

# =============================================================================
# Usage:
# -----------------------------------------------------------------------------
# Build:
#   docker build -t mlt:latest .
#
# Run analysis:
#   docker run -v $(pwd)/videos:/app/input -v $(pwd)/results:/app/output \
#       mlt:latest process /app/input/video.mp4 --output /app/output
#
# Interactive:
#   docker run -it -v $(pwd)/videos:/app/input mlt:latest /bin/bash
# =============================================================================
