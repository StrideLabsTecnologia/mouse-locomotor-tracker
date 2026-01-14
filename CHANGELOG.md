# Changelog

All notable changes to Mouse Locomotor Tracker will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

---

## [1.1.0] - 2025-01-14

### Added
- **Video Auto-Extraction**: Dashboard ahora extrae frames automáticamente al iniciar
- **assets/video.mp4**: Video de demostración incluido en el proyecto (3.2 MB)
- Función `extract_video_frames()` con cache de Streamlit para evitar re-extracción
- Mensajes de estado si falta OpenCV o el video

### Changed
- Dashboard usa rutas dinámicas (`VIDEO_PATH`, `VIDEO_FRAMES_DIR`) en lugar de hardcoded
- Mejor manejo de errores cuando el video no está disponible

### Fixed
- Video no se mostraba porque los frames no existían

---

## [1.0.0] - 2025-01-12

### Added
- **Core Analysis Engine**
  - Velocity calculation with configurable smoothing
  - Stride detection using peak finding algorithms
  - Gait pattern classification (walk, trot, gallop)
  - Biomechanical metrics (cadence, stride length, symmetry)

- **Tracking System**
  - Motion-based tracking using frame differencing
  - ROI-constrained detection for treadmill experiments
  - Multi-tracker architecture (motion, anatomical, DLC)
  - DeepLabCut integration for pose estimation

- **Visualization Suite**
  - Real-time tracking overlay on video
  - Trajectory heatmaps with density estimation
  - Velocity/acceleration time series plots
  - Gait phase diagrams
  - Publication-ready figure export

- **Export Formats**
  - CSV export with configurable columns
  - JSON export with full metadata
  - HDF5 export with compression (scientific format)
  - NWB export (Neurodata Without Borders standard)
  - Automatic format detection by file extension

- **Professional CLI**
  - Typer-based command interface
  - Rich terminal output with progress bars
  - Multiple output format flags
  - Preview mode for quick checks

- **Docker Support**
  - Multi-stage Dockerfile for optimized images
  - Docker Compose for development/testing/production
  - Jupyter notebook service for exploration

- **Quality Assurance**
  - Pre-commit hooks (black, isort, ruff, mypy, bandit)
  - GitHub Actions CI/CD pipeline
  - Multi-Python version testing (3.9-3.12)
  - Code coverage with Codecov integration

- **Documentation**
  - Comprehensive README with badges
  - API documentation with examples
  - Contributing guidelines
  - MIT License

### Technical Details
- Python 3.9+ compatibility
- OpenCV-based video processing
- NumPy/SciPy for numerical computations
- Pandas for data manipulation
- Type hints throughout codebase

---

## [0.2.0] - 2025-01-11

### Added
- Motion-only tracking processor
- ROI constraints for treadmill area
- Frame differencing algorithm
- Basic velocity calculations

### Fixed
- False positive detection of static objects (screws)
- Tracking accuracy improved from 70% to 100%

---

## [0.1.0] - 2025-01-10

### Added
- Initial project structure
- Basic video loading and processing
- Simple centroid tracking
- CSV export functionality

### Known Issues
- Static object detection causing false positives
- No ROI constraints

---

[Unreleased]: https://github.com/stridelabs/mouse-locomotor-tracker/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/stridelabs/mouse-locomotor-tracker/compare/v0.2.0...v1.0.0
[0.2.0]: https://github.com/stridelabs/mouse-locomotor-tracker/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/stridelabs/mouse-locomotor-tracker/releases/tag/v0.1.0
