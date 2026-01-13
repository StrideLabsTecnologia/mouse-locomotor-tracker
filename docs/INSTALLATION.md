# Installation Guide

Complete installation instructions for Mouse Locomotor Tracker.

## Table of Contents

1. [System Requirements](#system-requirements)
2. [Installation Methods](#installation-methods)
   - [pip Installation](#pip-installation)
   - [Conda Installation](#conda-installation)
   - [Development Installation](#development-installation)
3. [GPU Configuration](#gpu-configuration-for-deeplabcut)
4. [Troubleshooting](#troubleshooting)

## System Requirements

### Minimum Requirements

| Component | Requirement |
|-----------|-------------|
| OS | Windows 10+, macOS 10.14+, Ubuntu 18.04+ |
| Python | 3.8, 3.9, 3.10, or 3.11 |
| RAM | 8 GB minimum, 16 GB recommended |
| Storage | 2 GB for base installation |
| GPU | Optional (required for DeepLabCut inference) |

### Recommended for GPU Acceleration

| Component | Requirement |
|-----------|-------------|
| GPU | NVIDIA with CUDA support |
| CUDA | 11.2+ |
| cuDNN | 8.1+ |
| VRAM | 6 GB minimum |

## Installation Methods

### pip Installation

The simplest method for most users:

```bash
# Create and activate virtual environment (recommended)
python -m venv locomotor-env
source locomotor-env/bin/activate  # Linux/macOS
# OR
locomotor-env\Scripts\activate  # Windows

# Install from PyPI (when available)
pip install mouse-locomotor-tracker

# Or install from GitHub
pip install git+https://github.com/stridelabs/mouse-locomotor-tracker.git
```

### Conda Installation

Recommended for users who need DeepLabCut integration:

```bash
# Create conda environment
conda create -n locomotor python=3.10 -y
conda activate locomotor

# Install core dependencies
conda install numpy pandas scipy matplotlib h5py -y

# Install DeepLabCut (optional, for pose estimation)
conda install -c conda-forge deeplabcut -y

# Clone and install Mouse Locomotor Tracker
git clone https://github.com/stridelabs/mouse-locomotor-tracker.git
cd mouse-locomotor-tracker
pip install -e .
```

### Development Installation

For contributors and developers:

```bash
# Clone repository
git clone https://github.com/stridelabs/mouse-locomotor-tracker.git
cd mouse-locomotor-tracker

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install in development mode with all dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Verify installation
pytest tests/ -v
```

## Dependencies

### Core Dependencies

```
numpy>=1.20.0
pandas>=1.3.0
scipy>=1.7.0
matplotlib>=3.4.0
h5py>=3.0.0
PyYAML>=6.0
```

### Optional Dependencies

```
# For DeepLabCut integration
deeplabcut>=2.2.0

# For video processing
opencv-python>=4.5.0
ffmpeg-python>=0.2.0

# For development
pytest>=7.0.0
pytest-cov>=3.0.0
black>=22.0.0
flake8>=4.0.0
mypy>=0.950
```

## GPU Configuration for DeepLabCut

DeepLabCut requires GPU support for efficient pose estimation. Follow these steps for GPU configuration:

### NVIDIA GPU Setup (Linux)

```bash
# Check NVIDIA driver
nvidia-smi

# Install CUDA toolkit (Ubuntu example)
# Visit: https://developer.nvidia.com/cuda-downloads

# Install cuDNN
# Visit: https://developer.nvidia.com/cudnn

# Verify CUDA installation
nvcc --version
```

### NVIDIA GPU Setup (Windows)

1. Download and install NVIDIA drivers from [nvidia.com/drivers](https://www.nvidia.com/drivers)
2. Install CUDA Toolkit from [developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads)
3. Install cuDNN from [developer.nvidia.com/cudnn](https://developer.nvidia.com/cudnn)
4. Add CUDA to PATH:
   ```
   C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.x\bin
   C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.x\libnvvp
   ```

### TensorFlow GPU Configuration

```bash
# Install TensorFlow with GPU support
pip install tensorflow-gpu>=2.5.0

# Verify GPU detection
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

### CPU-Only Installation

If you don't have a GPU or only need to run analysis (not pose estimation):

```bash
# Install without GPU dependencies
pip install mouse-locomotor-tracker[cpu]

# Or manually exclude GPU packages
pip install tensorflow-cpu>=2.5.0
```

## Environment Variables

Configure these environment variables for optimal performance:

```bash
# Linux/macOS (.bashrc or .zshrc)
export MLT_DATA_DIR="$HOME/locomotor_data"
export MLT_CONFIG_DIR="$HOME/.config/locomotor"
export TF_CPP_MIN_LOG_LEVEL=2  # Reduce TensorFlow verbosity

# Windows (PowerShell)
$env:MLT_DATA_DIR = "$HOME\locomotor_data"
$env:MLT_CONFIG_DIR = "$HOME\.config\locomotor"
$env:TF_CPP_MIN_LOG_LEVEL = 2
```

## Verification

Verify your installation with these tests:

```bash
# Check version
python -c "import mouse_locomotor_tracker; print(mouse_locomotor_tracker.__version__)"

# Run basic import test
python -c "
from mouse_locomotor_tracker import LocomotorPipeline
from mouse_locomotor_tracker.analysis import VelocityAnalyzer
from mouse_locomotor_tracker.tracking import VideoMetadata, MarkerSet
print('All imports successful!')
"

# Run test suite
pytest tests/ -v

# Check DeepLabCut availability (optional)
python -c "
try:
    import deeplabcut
    print(f'DeepLabCut version: {deeplabcut.__version__}')
except ImportError:
    print('DeepLabCut not installed (optional)')
"
```

## Docker Installation

For isolated environments using Docker:

```dockerfile
# Dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy and install package
COPY . .
RUN pip install --no-cache-dir -e .

# Run tests
CMD ["pytest", "tests/", "-v"]
```

Build and run:

```bash
# Build image
docker build -t locomotor-tracker .

# Run container
docker run -it locomotor-tracker

# Run with mounted data
docker run -v /path/to/data:/data locomotor-tracker python analyze.py
```

## Troubleshooting

### Common Issues

#### Import Error: No module named 'mouse_locomotor_tracker'

```bash
# Ensure package is installed
pip list | grep mouse-locomotor

# Reinstall in development mode
pip install -e .
```

#### NumPy/SciPy Version Conflicts

```bash
# Reinstall compatible versions
pip uninstall numpy scipy -y
pip install numpy>=1.20.0 scipy>=1.7.0
```

#### DeepLabCut Import Errors

```bash
# Install with conda (recommended for DLC)
conda install -c conda-forge deeplabcut

# Check TensorFlow compatibility
python -c "import tensorflow as tf; print(tf.__version__)"
```

#### GPU Not Detected

```bash
# Check CUDA installation
nvcc --version
nvidia-smi

# Check TensorFlow GPU
python -c "
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f'Found {len(gpus)} GPU(s)')
else:
    print('No GPU found')
"
```

#### Memory Errors with Large Videos

```python
# Process videos in chunks
from mouse_locomotor_tracker import LocomotorPipeline

pipeline = LocomotorPipeline(config={
    'chunk_size': 10000,  # Process 10k frames at a time
    'memory_efficient': True
})
```

### Platform-Specific Issues

#### macOS Apple Silicon (M1/M2)

```bash
# Install miniforge for ARM64
brew install miniforge

# Create environment
conda create -n locomotor python=3.10
conda activate locomotor

# Install with ARM64 optimized packages
conda install numpy pandas scipy matplotlib -y
pip install tensorflow-macos tensorflow-metal  # Apple GPU support
```

#### Windows Long Path Issues

Enable long paths in Windows Registry:
```powershell
# Run as Administrator
New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" `
    -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force
```

### Getting Help

If you encounter issues not covered here:

1. Check [GitHub Issues](https://github.com/stridelabs/mouse-locomotor-tracker/issues)
2. Search existing issues before creating new ones
3. Provide:
   - Operating system and version
   - Python version (`python --version`)
   - Package versions (`pip freeze`)
   - Full error traceback
   - Minimal code to reproduce

## Updating

### Update to Latest Version

```bash
# pip
pip install --upgrade mouse-locomotor-tracker

# From GitHub
pip install --upgrade git+https://github.com/stridelabs/mouse-locomotor-tracker.git

# Development version
cd mouse-locomotor-tracker
git pull origin main
pip install -e .
```

### Changelog

See [CHANGELOG.md](../CHANGELOG.md) for version history and migration guides.
