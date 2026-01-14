# Mouse Locomotor Tracker

Una librería completa en Python para analizar la locomoción de ratones desde grabaciones de video usando estimación de pose con DeepLabCut.

## Descripción General

Mouse Locomotor Tracker proporciona análisis automatizado de la locomoción de roedores, incluyendo:

- **Análisis de Velocidad**: Velocidad, aceleración y detección de arrastre
- **Coordinación de Extremidades**: Relaciones de fase entre extremidades usando estadística circular
- **Detección de Ciclo de Marcha**: Cadencia, longitud de zancada y métricas de regularidad de marcha
- **Exportación**: Formatos de salida JSON, CSV y HDF5

```
+------------------+     +------------------+     +------------------+
|                  |     |                  |     |                  |
|  Entrada Video   +---->+  DeepLabCut      +---->+  Pipeline de     |
|  (.avi, .mp4)    |     |  Tracking Pose   |     |  Análisis        |
|                  |     |                  |     |                  |
+------------------+     +------------------+     +--------+---------+
                                                         |
                        +--------------------------------+
                        |
         +--------------+---------------+----------------+
         |              |               |                |
         v              v               v                v
   +-----------+  +-----------+  +------------+  +------------+
   | Análisis  |  | Análisis  |  | Ciclos     |  | Módulo     |
   | Velocidad |  | Coord.    |  | Marcha     |  | Export     |
   +-----------+  +-----------+  +------------+  +------------+
```

## Características

### Análisis de Velocidad
- Cálculo de velocidad instantánea y promedio
- Detección de aceleración y desaceleración
- Identificación y cuantificación de eventos de arrastre
- Múltiples opciones de filtros de suavizado

### Coordinación de Extremidades
- Estadística circular para análisis de fase
- Todas las combinaciones estándar de pares de extremidades
- Clasificación de patrones de marcha (trote, paso, galope)
- Longitud del vector resultante (R) para fuerza de coordinación

### Detección de Ciclo de Marcha
- Detección automática de ciclos usando búsqueda de picos
- Cálculo de cadencia (frecuencia de paso)
- Estimación de longitud de zancada
- Métricas de regularidad de marcha (CV de duración de ciclo)

### Opciones de Exportación
- JSON para resultados completos con metadatos
- CSV para estadísticas resumidas
- HDF5 para grandes conjuntos de datos
- Visualizaciones con Matplotlib

## Inicio Rápido

### Instalación

```bash
# Clonar el repositorio
git clone https://github.com/stridelabs/mouse-locomotor-tracker.git
cd mouse-locomotor-tracker

# Crear entorno virtual
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt

# Instalar en modo desarrollo
pip install -e .
```

### Uso Básico

```python
from mouse_locomotor_tracker import LocomotorPipeline
from mouse_locomotor_tracker.tracking import VideoMetadata, MarkerSet

# Configurar marcadores
markers = MarkerSet(
    name="mouse_ventral",
    markers=["snout", "foreL", "foreR", "hindL", "hindR", "torso", "tail"],
    limb_pairs={
        "LH_RH": ("hindL", "hindR"),
        "diagonal": ("foreL", "hindR"),
    },
    speed_markers=["snout", "torso", "tail"]
)

# Cargar datos de tracking (salida de DeepLabCut)
import pandas as pd
tracks = pd.read_hdf("tracking_results.h5")

# Crear metadatos
metadata = VideoMetadata(
    duration=30.0,      # segundos
    fps=100,            # cuadros por segundo
    n_frames=3000,
    width=640,
    height=480,
    pixel_width_mm=0.3125
)

# Ejecutar análisis
pipeline = LocomotorPipeline()
results = pipeline.process_tracks(
    tracks=tracks,
    metadata=metadata,
    model_name="DLC_model",
    markers=markers.markers,
    limb_pairs=markers.limb_pairs,
    speed_markers=markers.speed_markers
)

# Exportar resultados
pipeline.export_results("results.json", format="json")

# Acceder a métricas específicas
print(f"Velocidad Media: {results['summary']['mean_speed_cm_s']:.2f} cm/s")
print(f"Cadencia Media: {results['summary']['mean_cadence_hz']:.2f} Hz")
print(f"Coordinación (R): {results['summary']['mean_coordination_R']:.3f}")
```

## Arquitectura

```
mouse-locomotor-tracker/
|
+-- tracking/                  # Módulo de tracking
|   +-- __init__.py
|   +-- dlc_adapter.py        # Interfaz DeepLabCut
|   +-- marker_config.py      # Configuración de marcadores
|   +-- video_metadata.py     # Metadatos de archivo de video
|   +-- mock_tracker.py       # Datos sintéticos para testing
|   +-- track_processor.py    # Post-procesamiento de tracks
|
+-- analysis/                  # Módulos de análisis
|   +-- __init__.py
|   +-- velocity.py           # VelocityAnalyzer
|   +-- coordination.py       # CircularCoordinationAnalyzer
|   +-- gait_cycles.py        # GaitCycleDetector
|   +-- pipeline.py           # LocomotorPipeline
|
+-- visualization/             # Herramientas de visualización
|   +-- __init__.py
|   +-- plotter.py            # Gráficos estáticos
|   +-- video_overlay.py      # Anotación de video
|
+-- export/                    # Funcionalidad de exportación
|   +-- __init__.py
|   +-- json_export.py
|   +-- csv_export.py
|   +-- hdf5_export.py
|
+-- config/                    # Archivos de configuración
|   +-- default_config.yaml
|   +-- markers_ventral.yaml
|   +-- markers_lateral.yaml
|
+-- tests/                     # Suite de tests
|   +-- conftest.py
|   +-- test_velocity.py
|   +-- test_coordination.py
|   +-- test_gait_cycles.py
|   +-- test_integration.py
|
+-- docs/                      # Documentación
    +-- README.md
    +-- INSTALLATION.md
    +-- USER_GUIDE.md
    +-- API.md
    +-- METRICS.md
```

## Diagrama de Flujo de Trabajo

```
                    +-------------------+
                    |   Video Raw       |
                    |   (.avi, .mp4)    |
                    +---------+---------+
                              |
                              v
                    +-------------------+
                    |   DeepLabCut      |
                    |   Estimación Pose |
                    +---------+---------+
                              |
                              v
                    +-------------------+
                    |   Datos Tracking  |
                    |   (HDF5/CSV)      |
                    +---------+---------+
                              |
              +---------------+---------------+
              |               |               |
              v               v               v
       +------+------+ +------+------+ +------+------+
       |  Analizador | | Analizador  | | Detector    |
       |  Velocidad  | | Coordinación| | Ciclo Marcha|
       +------+------+ +------+------+ +------+------+
              |               |               |
              +---------------+---------------+
                              |
                              v
                    +-------------------+
                    |   Agregación      |
                    |   Resultados      |
                    +---------+---------+
                              |
              +---------------+---------------+
              |               |               |
              v               v               v
       +------+------+ +------+------+ +------+------+
       |    JSON     | |    CSV      | |   Gráficos  |
       |   Export    | |   Export    | |   & Video   |
       +-------------+ +-------------+ +-------------+
```

## Referencia de API

### Clases Principales

| Clase | Descripción |
|-------|-------------|
| `LocomotorPipeline` | Pipeline principal que orquesta todos los módulos de análisis |
| `VelocityAnalyzer` | Cálculo de velocidad y aceleración |
| `CircularCoordinationAnalyzer` | Análisis de relación de fase entre extremidades |
| `GaitCycleDetector` | Detección de ciclo de marcha y métricas de zancada |
| `VideoMetadata` | Contenedor de metadatos de archivo de video |
| `MarkerSet` | Configuración de marcadores para tracking |

### Referencia Rápida

```python
# Análisis de Velocidad
from analysis.velocity import VelocityAnalyzer
analyzer = VelocityAnalyzer(smoothing_factor=10)
speed = analyzer.compute_speed(positions, fps, pixel_to_mm)
accel = analyzer.compute_acceleration(speed, fps)
drag, recovery, stats = analyzer.detect_drag_events(accel, fps)

# Análisis de Coordinación
from analysis.coordination import CircularCoordinationAnalyzer
coord = CircularCoordinationAnalyzer()
mean_phi, R = coord.circular_mean(phase_angles)
results = coord.analyze_all_limb_pairs(tracks, limb_pairs, duration)

# Detección de Ciclo de Marcha
from analysis.gait_cycles import GaitCycleDetector
detector = GaitCycleDetector()
n_cycles, peaks, troughs = detector.detect_cycles(stride, fps)
cadence = detector.compute_cadence(stride, duration)
stride_length = detector.compute_stride_length(cadence, avg_speed)
```

## Documentación

- [Guía de Instalación](INSTALLATION.md) - Instrucciones detalladas de instalación
- [Guía de Usuario](USER_GUIDE.md) - Tutorial de uso paso a paso
- [Referencia de API](API.md) - Documentación completa de API
- [Guía de Métricas](METRICS.md) - Explicación de métricas calculadas

## Requisitos

- Python 3.8+
- NumPy >= 1.20
- Pandas >= 1.3
- SciPy >= 1.7
- Matplotlib >= 3.4
- DeepLabCut >= 2.2 (opcional, para estimación de pose)

## Testing

```bash
# Ejecutar todos los tests
pytest tests/

# Ejecutar con cobertura
pytest tests/ --cov=. --cov-report=html

# Ejecutar módulo de test específico
pytest tests/test_velocity.py -v

# Ejecutar solo tests rápidos (saltar tests de rendimiento lentos)
pytest tests/ -m "not slow"
```

## Objetivos de Cobertura

| Módulo | Objetivo | Rutas Críticas |
|--------|----------|----------------|
| Dominio/Lógica de Negocio | 90%+ | 100% |
| Módulos de Análisis | 90%+ | 100% |
| Integración | 70%+ | 80% |

## Contribuir

1. Haz fork del repositorio
2. Crea una rama de funcionalidad (`git checkout -b feature/funcionalidad-increible`)
3. Haz commit de tus cambios (`git commit -m 'Agregar funcionalidad increíble'`)
4. Push a la rama (`git push origin feature/funcionalidad-increible`)
5. Abre un Pull Request

## Licencia

Licencia MIT - ver [LICENSE](../LICENSE) para detalles.

## Citación

Si usas este software en tu investigación, por favor cita:

```bibtex
@software{mouse_locomotor_tracker,
  title = {Mouse Locomotor Tracker},
  author = {Stride Labs},
  year = {2024},
  url = {https://github.com/stridelabs/mouse-locomotor-tracker}
}
```

## Agradecimientos

Este proyecto se basa en metodologías de:

- Allodi et al. (2021) - Métodos de análisis de locomoción
- DeepLabCut - Framework de estimación de pose
- Implementaciones de estadística circular

## Contacto

- **Autor**: Stride Labs
- **Email**: contact@stridelabs.cl
- **Sitio Web**: https://stridelabs.cl
