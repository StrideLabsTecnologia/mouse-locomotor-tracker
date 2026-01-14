<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/OpenCV-4.5%2B-green?logo=opencv&logoColor=white" alt="OpenCV">
  <img src="https://img.shields.io/badge/Licencia-MIT-yellow" alt="Licencia">
  <img src="https://img.shields.io/badge/code%20style-black-000000.svg" alt="Estilo de Código">
  <img src="https://img.shields.io/badge/pre--commit-habilitado-brightgreen?logo=pre-commit" alt="Pre-commit">
</p>

<h1 align="center">
  <br>
  Mouse Locomotor Tracker
  <br>
</h1>

<h4 align="center">Análisis biomecánico profesional para investigación de locomoción en roedores</h4>

<p align="center">
  <a href="#características">Características</a> •
  <a href="#instalación">Instalación</a> •
  <a href="#inicio-rápido">Inicio Rápido</a> •
  <a href="#uso-del-cli">CLI</a> •
  <a href="#api">API</a> •
  <a href="#citación">Citación</a>
</p>

---

## Descripción General

**Mouse Locomotor Tracker (MLT)** es un toolkit profesional en Python para tracking automatizado y análisis biomecánico de locomoción en roedores. Diseñado para investigación en neurociencia, proporciona:

- **Tracking basado en movimiento** optimizado para experimentos en cinta
- **Métricas biomecánicas**: velocidad, aceleración, ciclos de marcha, coordinación
- **Visualizaciones para publicación**: overlays de trayectoria, gráficos polares, perfiles de velocidad
- **Formatos de exportación científicos**: CSV, JSON, HDF5, NWB

```
╔══════════════════════════════════════════════════════════════╗
║   ███╗   ███╗██╗  ████████╗                                  ║
║   ████╗ ████║██║  ╚══██╔══╝  Mouse Locomotor Tracker        ║
║   ██╔████╔██║██║     ██║     Professional Edition v1.1      ║
║   ██║╚██╔╝██║██║     ██║                                     ║
║   ██║ ╚═╝ ██║███████╗██║     Stride Labs                    ║
║   ╚═╝     ╚═╝╚══════╝╚═╝                                     ║
╚══════════════════════════════════════════════════════════════╝
```

---

## Características

### Tracking
- **Detección basada en movimiento**: Diferencia de frames optimizada para ROI de cinta
- **100% de tasa de tracking** en videos estándar de cinta
- **Suavizado temporal**: Promedio móvil exponencial para tracking estable
- **Integración opcional con DeepLabCut**: 27 puntos anatómicos

### Análisis
| Módulo | Métricas |
|--------|----------|
| **Velocidad** | Velocidad instantánea, promedio, máxima; aceleración |
| **Coordinación** | Estadísticas circulares, acoplamiento de fase, test de Rayleigh |
| **Ciclos de Marcha** | Cadencia, longitud de zancada, factor de servicio, simetría |
| **Cinemática** | Ángulos articulares, rango de movimiento, longitudes de extremidades |

### Visualización
- Overlays de trayectoria con trails de gradiente
- Gráficos de coordinación circular (polar)
- Perfiles de velocidad con overlay de aceleración
- Dashboard en tiempo real con medidores
- Exportación de figuras para publicación

### Exportación
- **CSV**: Datos de tracking frame por frame
- **JSON**: Resultados completos con metadatos
- **HDF5**: Formato binario eficiente para grandes datasets
- **NWB**: Neurodata Without Borders (estándar de neurociencia)

---

## Instalación

### Requisitos Previos

- Python 3.9 o superior
- pip (gestor de paquetes de Python)
- Git
- OpenCV (para procesamiento de video)
- FFmpeg (opcional, para extracción de frames)

---

### Linux (Ubuntu/Debian)

```bash
# 1. Actualizar sistema e instalar dependencias
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3 python3-pip python3-venv git ffmpeg

# 2. Instalar dependencias de OpenCV
sudo apt install -y libopencv-dev python3-opencv

# 3. Clonar repositorio
git clone https://github.com/StrideLabsTecnologia/mouse-locomotor-tracker.git
cd mouse-locomotor-tracker

# 4. Crear entorno virtual
python3 -m venv .venv
source .venv/bin/activate

# 5. Instalar dependencias
pip install --upgrade pip
pip install -r requirements.txt

# 6. Verificar instalación
python -c "import cv2; print(f'OpenCV: {cv2.__version__}')"

# 7. Ejecutar dashboard
streamlit run scientific_dashboard.py
```

---

### Arch Linux

```bash
# 1. Actualizar sistema
sudo pacman -Syu

# 2. Instalar dependencias
sudo pacman -S python python-pip git opencv hdf5 ffmpeg

# 3. Instalar dependencias de Python para OpenCV
sudo pacman -S python-numpy python-scipy

# 4. Clonar repositorio
git clone https://github.com/StrideLabsTecnologia/mouse-locomotor-tracker.git
cd mouse-locomotor-tracker

# 5. Crear entorno virtual
python -m venv .venv
source .venv/bin/activate

# 6. Instalar dependencias
pip install --upgrade pip
pip install -r requirements.txt

# 7. Verificar instalación
python -c "import cv2; print(f'OpenCV: {cv2.__version__}')"

# 8. Ejecutar dashboard
streamlit run scientific_dashboard.py
```

**Nota para Arch Linux**: Si usas `yay` o `paru` para AUR:
```bash
# Alternativa con python-opencv desde AUR (más completo)
yay -S python-opencv
```

---

### macOS

```bash
# 1. Instalar Homebrew (si no está instalado)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# 2. Instalar dependencias
brew install python@3.11 git opencv ffmpeg

# 3. Clonar repositorio
git clone https://github.com/StrideLabsTecnologia/mouse-locomotor-tracker.git
cd mouse-locomotor-tracker

# 4. Crear entorno virtual
python3 -m venv .venv
source .venv/bin/activate

# 5. Instalar dependencias
pip install --upgrade pip
pip install -r requirements.txt

# 6. Verificar instalación
python -c "import cv2; print(f'OpenCV: {cv2.__version__}')"

# 7. Ejecutar dashboard
streamlit run scientific_dashboard.py
```

**Nota para Apple Silicon (M1/M2/M3)**:
```bash
# Si hay problemas con OpenCV, instalar con conda
brew install miniforge
conda create -n mlt python=3.11
conda activate mlt
conda install -c conda-forge opencv
pip install -r requirements.txt
```

---

### Windows

#### Opción 1: PowerShell (Recomendado)

```powershell
# 1. Instalar Python desde https://www.python.org/downloads/
# Marcar "Add Python to PATH" durante la instalación

# 2. Instalar Git desde https://git-scm.com/download/win

# 3. Abrir PowerShell como Administrador y clonar repositorio
git clone https://github.com/StrideLabsTecnologia/mouse-locomotor-tracker.git
cd mouse-locomotor-tracker

# 4. Crear entorno virtual
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 5. Instalar dependencias
pip install --upgrade pip
pip install -r requirements.txt

# 6. Verificar instalación
python -c "import cv2; print(f'OpenCV: {cv2.__version__}')"

# 7. Ejecutar dashboard
streamlit run scientific_dashboard.py
```

#### Opción 2: WSL2 (Windows Subsystem for Linux)

```powershell
# 1. Habilitar WSL2 (PowerShell como Administrador)
wsl --install -d Ubuntu

# 2. Reiniciar y abrir Ubuntu

# 3. Seguir instrucciones de Linux (Ubuntu/Debian) arriba
```

#### Opción 3: Anaconda/Miniconda

```powershell
# 1. Instalar Miniconda desde https://docs.conda.io/en/latest/miniconda.html

# 2. Abrir Anaconda Prompt
conda create -n mlt python=3.11
conda activate mlt

# 3. Instalar OpenCV y dependencias
conda install -c conda-forge opencv numpy scipy pandas

# 4. Clonar e instalar
git clone https://github.com/StrideLabsTecnologia/mouse-locomotor-tracker.git
cd mouse-locomotor-tracker
pip install -r requirements.txt

# 5. Ejecutar dashboard
streamlit run scientific_dashboard.py
```

---

### Docker (Multiplataforma)

```bash
# 1. Construir imagen
docker build -t mouse-locomotor-tracker .

# 2. Ejecutar contenedor
docker run -p 8501:8501 mouse-locomotor-tracker

# 3. Abrir en navegador: http://localhost:8501
```

**Docker Compose**:
```bash
docker-compose up
```

---

### Instalación de Desarrollo

```bash
# Clonar con submódulos
git clone --recursive https://github.com/StrideLabsTecnologia/mouse-locomotor-tracker.git
cd mouse-locomotor-tracker

# Crear entorno virtual
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .\.venv\Scripts\Activate.ps1  # Windows

# Instalar en modo desarrollo
pip install -e ".[dev]"

# Configurar pre-commit hooks
pre-commit install

# Ejecutar tests
pytest tests/ -v
```

---

## Inicio Rápido

### Dashboard Científico (Recomendado)

```bash
# Activar entorno virtual
source .venv/bin/activate  # Linux/macOS
# .\.venv\Scripts\Activate.ps1  # Windows

# Ejecutar dashboard
streamlit run scientific_dashboard.py
```

El dashboard se abrirá en `http://localhost:8501` con:
- Video sincronizado del experimento
- Animación de tracking en tiempo real
- Gráficos de velocidad, aceleración y cadencia
- Exportación de datos

### Línea de Comandos

```bash
# Procesar video con configuración por defecto
python cli.py process video.mp4 -o tracked.mp4

# Con exportación CSV y preview
python cli.py process video.mp4 -o tracked.mp4 --csv --preview

# Obtener información del video
python cli.py info video.mp4
```

### API de Python

```python
from analysis import VelocityAnalyzer, GaitCycleDetector
from visualization import TrajectoryVisualizer

# Analizar velocidad
analyzer = VelocityAnalyzer(frame_rate=30, pixel_to_mm=0.1)
metrics = analyzer.analyze(x_coords, y_coords)

print(f"Velocidad Promedio: {metrics.mean_speed:.2f} cm/s")
print(f"Velocidad Máxima: {metrics.max_speed:.2f} cm/s")
print(f"Distancia Total: {metrics.total_distance:.1f} cm")

# Detectar ciclos de marcha
detector = GaitCycleDetector(fps=30)
gait = detector.detect_cycles(stride_signal, x_positions)

print(f"Cadencia: {gait.cadence:.2f} Hz")
print(f"Longitud de Zancada: {gait.mean_stride_length:.2f} cm")
```

---

## Uso del CLI

### Comando Process

```bash
python cli.py process INPUT [OPCIONES]

Argumentos:
  INPUT                 Ruta del archivo de video de entrada

Opciones:
  -o, --output PATH     Ruta del video de salida
  -p, --preview         Mostrar ventana de preview
  -m, --max-frames INT  Limitar frames a procesar
  --csv                 Exportar datos de tracking a CSV
  --json                Exportar resultados a JSON
  --help                Mostrar ayuda
```

### Ejemplos

```bash
# Procesamiento básico
python cli.py process experimento_001.mp4

# Análisis completo con exportaciones
python cli.py process experimento_001.mp4 \
    --output resultados/tracked.mp4 \
    --csv \
    --json \
    --max-frames 1000

# Preview rápido
python cli.py process video.mp4 --preview --max-frames 100
```

---

## API

### Módulo de Análisis

```python
from analysis import (
    VelocityAnalyzer,
    CircularCoordinationAnalyzer,
    GaitCycleDetector,
    JointAngleAnalyzer,
)

# Análisis de Velocidad
analyzer = VelocityAnalyzer(frame_rate=30, pixel_to_mm=0.1)
metrics = analyzer.analyze(x, y)
# Retorna: VelocityMetrics(mean_speed, max_speed, acceleration, ...)

# Análisis de Coordinación
coord = CircularCoordinationAnalyzer()
stats = coord.analyze_pair(phases_a, phases_b)
# Retorna: CircularStatistics(mean_angle, R, rayleigh_z, p_value)

# Detección de Ciclos de Marcha
detector = GaitCycleDetector(fps=30, min_cycle_duration=0.1)
gait = detector.detect_cycles(stride_signal, x_positions)
# Retorna: GaitMetrics(cadence, stride_length, duty_factor, ...)
```

### Módulo de Visualización

```python
from visualization import (
    TrajectoryVisualizer,
    CoordinationPlotter,
    SpeedProfilePlotter,
    VideoGenerator,
)

# Overlay de trayectoria
viz = TrajectoryVisualizer(trail_length=50, color_scheme='velocity')
frame_with_overlay = viz.draw(frame, positions, velocities)

# Gráfico de coordinación polar
plotter = CoordinationPlotter()
fig = plotter.plot_pair(phases, title="Coordinación LH-RH")
fig.savefig("coordinacion.png", dpi=300)

# Generar video anotado
generator = VideoGenerator(fps=30, codec='h264')
generator.process_video(input_path, output_path, tracking_data)
```

---

## Estructura del Proyecto

```
mouse-locomotor-tracker/
├── analysis/              # Análisis biomecánico
│   ├── velocity.py        # Velocidad y aceleración
│   ├── coordination.py    # Estadísticas circulares
│   ├── gait_cycles.py     # Detección de ciclos
│   ├── kinematics.py      # Ángulos articulares
│   └── metrics.py         # Estructuras de datos
├── visualization/         # Gráficos y video
│   ├── trajectory_overlay.py
│   ├── circular_plots.py
│   ├── speed_plots.py
│   ├── dashboard.py
│   └── video_generator.py
├── tracking/              # Estimación de pose
│   ├── dlc_adapter.py     # Wrapper de DeepLabCut
│   ├── marker_config.py   # Definiciones de keypoints
│   └── track_processor.py # Post-procesamiento
├── export/                # Exportación de datos
│   ├── csv_exporter.py
│   ├── json_exporter.py
│   └── report_generator.py
├── assets/                # Recursos
│   └── video.mp4          # Video de demostración
├── tests/                 # Suite de tests
├── docs/                  # Documentación
├── cli.py                 # Interfaz de línea de comandos
├── scientific_dashboard.py # Dashboard Streamlit
└── pyproject.toml         # Configuración del proyecto
```

---

## Referencia de Métricas

### Métricas de Velocidad

| Métrica | Unidad | Descripción |
|---------|--------|-------------|
| `mean_speed` | cm/s | Velocidad instantánea promedio |
| `max_speed` | cm/s | Velocidad máxima registrada |
| `min_speed` | cm/s | Velocidad mínima (excluyendo paradas) |
| `total_distance` | cm | Distancia total recorrida |
| `acceleration` | cm/s² | Tasa de cambio de velocidad |

### Métricas de Coordinación

| Métrica | Rango | Descripción |
|---------|-------|-------------|
| `mean_angle` | -180° a 180° | Relación de fase media |
| `R` | 0 a 1 | Longitud del vector resultante (fuerza de coordinación) |
| `rayleigh_z` | ≥0 | Estadístico del test de Rayleigh |
| `p_value` | 0 a 1 | Significancia estadística |

### Métricas de Marcha

| Métrica | Unidad | Descripción |
|---------|--------|-------------|
| `cadence` | Hz | Pasos por segundo |
| `stride_length` | cm | Distancia por ciclo |
| `duty_factor` | ratio | Duración stance/ciclo |
| `symmetry_index` | % | Simetría izquierda-derecha |

---

## Solución de Problemas

### Error: "OpenCV not found"

```bash
# Linux
sudo apt install python3-opencv

# macOS
brew install opencv
pip install opencv-python

# Windows
pip install opencv-python
```

### Error: "Streamlit command not found"

```bash
pip install streamlit
# o
python -m streamlit run scientific_dashboard.py
```

### El video no se muestra en el dashboard

El dashboard extrae automáticamente los frames del video `assets/video.mp4`. Si no existe:

```bash
# Verificar que existe el video
ls -la assets/video.mp4

# Si usas tu propio video, copiarlo a assets/
cp tu_video.mp4 assets/video.mp4
```

---

## Citación

Si usas MLT en tu investigación, por favor cita:

```bibtex
@software{mlt2025,
  author = {Stride Labs},
  title = {Mouse Locomotor Tracker: Análisis Biomecánico Profesional},
  year = {2025},
  url = {https://github.com/StrideLabsTecnologia/mouse-locomotor-tracker},
  version = {1.1.0}
}
```

---

## Contribuir

¡Las contribuciones son bienvenidas! Por favor consulta [CONTRIBUTING.md](CONTRIBUTING.md) para las guías.

1. Fork del repositorio
2. Crear rama de feature (`git checkout -b feature/nueva-caracteristica`)
3. Commit de cambios (`git commit -m 'feat: agregar nueva característica'`)
4. Push a la rama (`git push origin feature/nueva-caracteristica`)
5. Abrir Pull Request

---

## Licencia

Licencia MIT - ver [LICENSE](LICENSE) para detalles.

---

<p align="center">
  Hecho con cariño por <a href="https://stridelabs.cl">Stride Labs</a>
</p>
