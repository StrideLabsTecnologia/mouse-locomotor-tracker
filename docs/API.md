# Referencia de API

Documentación completa de API para Mouse Locomotor Tracker.

## Tabla de Contenidos

1. [Módulo de Tracking](#módulo-de-tracking)
2. [Módulo de Análisis](#módulo-de-análisis)
3. [Módulo de Visualización](#módulo-de-visualización)
4. [Módulo de Exportación](#módulo-de-exportación)

---

## Módulo de Tracking

### `tracking.VideoMetadata`

Contenedor para metadatos de archivo de video.

```python
from mouse_locomotor_tracker.tracking import VideoMetadata
```

#### Constructor

```python
VideoMetadata(
    duration: float,       # Duración total en segundos
    fps: int,              # Cuadros por segundo
    n_frames: int,         # Número total de cuadros
    width: int,            # Ancho de imagen en píxeles
    height: int,           # Alto de imagen en píxeles
    pixel_width_mm: float  # Ancho físico por píxel (mm)
)
```

#### Atributos

| Atributo | Tipo | Descripción |
|----------|------|-------------|
| `duration` | `float` | Duración del video en segundos |
| `fps` | `int` | Cuadros por segundo |
| `n_frames` | `int` | Conteo total de cuadros |
| `width` | `int` | Ancho del cuadro en píxeles |
| `height` | `int` | Alto del cuadro en píxeles |
| `pixel_width_mm` | `float` | Tamaño físico por píxel (mm) |

#### Métodos

##### `to_dict() -> dict`

Convertir metadatos a formato diccionario.

```python
metadata = VideoMetadata(duration=30.0, fps=100, n_frames=3000,
                        width=640, height=480, pixel_width_mm=0.3125)
d = metadata.to_dict()
# {'dur': 30.0, 'fps': 100, 'nFrame': 3000, 'imW': 640, 'imH': 480, 'xPixW': 0.3125}
```

##### `from_dict(d: dict) -> VideoMetadata` (classmethod)

Crear VideoMetadata desde diccionario.

```python
d = {'dur': 30.0, 'fps': 100, 'nFrame': 3000, 'imW': 640, 'imH': 480, 'xPixW': 0.3125}
metadata = VideoMetadata.from_dict(d)
```

---

### `tracking.MarkerSet`

Configuración para marcadores de tracking.

```python
from mouse_locomotor_tracker.tracking import MarkerSet
```

#### Constructor

```python
MarkerSet(
    name: str,                              # Nombre de configuración
    markers: List[str],                     # Lista de nombres de marcadores
    limb_pairs: Dict[str, Tuple[str, str]], # Definiciones de pares de extremidades
    speed_markers: List[str]                # Marcadores para cálculo de velocidad
)
```

#### Atributos

| Atributo | Tipo | Descripción |
|----------|------|-------------|
| `name` | `str` | Identificador de configuración |
| `markers` | `List[str]` | Todos los nombres de marcadores |
| `limb_pairs` | `Dict` | Mapeo de nombres de pares a tuplas de extremidades |
| `speed_markers` | `List[str]` | Marcadores usados para velocidad corporal |

#### Métodos

##### `get_all_markers() -> List[str]`

Retorna lista de todos los nombres de marcadores.

```python
markers = marker_set.get_all_markers()
# ['snout', 'foreL', 'foreR', 'hindL', 'hindR', 'torso', 'tail']
```

##### `get_limb_pair(pair_name: str) -> Tuple[str, str]`

Obtener tupla de marcadores para un par de extremidades.

```python
pair = marker_set.get_limb_pair("LH_RH")
# ('hindL', 'hindR')
```

#### Conjuntos de Marcadores Predefinidos

```python
from mouse_locomotor_tracker.tracking import MOUSE_VENTRAL, MOUSE_LATERAL

# Vista ventral (cámara inferior)
MOUSE_VENTRAL  # 11 marcadores, 6 pares de extremidades

# Vista lateral (cámara lateral)
MOUSE_LATERAL  # 6 marcadores para ángulos articulares
```

---

### `tracking.MockTracker`

Generar datos de tracking sintéticos para testing.

```python
from mouse_locomotor_tracker.tracking import MockTracker
```

#### Constructor

```python
MockTracker(
    markers: List[str],            # Nombres de marcadores
    model_name: str = "DLC_mock"   # Identificador de modelo
)
```

#### Métodos

##### `generate_tracks(metadata, gait_frequency, speed_cm_s, noise_level) -> pd.DataFrame`

Generar datos de tracking sintéticos.

```python
tracker = MockTracker(markers=["snout", "hindL", "hindR", "tail"])
tracks = tracker.generate_tracks(
    metadata=metadata,
    gait_frequency=4.0,    # Hz
    speed_cm_s=15.0,       # cm/s
    noise_level=1.0        # píxeles
)
```

**Parámetros:**

| Parámetro | Tipo | Default | Descripción |
|-----------|------|---------|-------------|
| `metadata` | `VideoMetadata` | requerido | Metadatos de video |
| `gait_frequency` | `float` | `4.0` | Frecuencia del ciclo de marcha (Hz) |
| `speed_cm_s` | `float` | `15.0` | Velocidad de locomoción (cm/s) |
| `noise_level` | `float` | `1.0` | Ruido de tracking (píxeles) |

**Retorna:** `pd.DataFrame` con columnas MultiIndex que coinciden con formato DeepLabCut.

---

## Módulo de Análisis

### `analysis.VelocityAnalyzer`

Analizar velocidad y aceleración desde datos de tracking.

```python
from mouse_locomotor_tracker.analysis import VelocityAnalyzer
```

#### Constructor

```python
VelocityAnalyzer(
    smoothing_factor: int = 10,        # Ventana de suavizado de velocidad
    accel_smoothing_factor: int = 12,  # Ventana de suavizado de aceleración
    speed_threshold: float = 5.0,      # Umbral de movimiento (cm/s)
    drag_threshold: float = 0.25       # Umbral de evento de arrastre (segundos)
)
```

#### Métodos

##### `compute_speed(positions, fps, pixel_to_mm) -> np.ndarray`

Calcular velocidad instantánea desde array de posiciones.

```python
analyzer = VelocityAnalyzer()
positions = np.array([[x1, y1], [x2, y2], ...])  # Forma: (n_frames, 2)
speed = analyzer.compute_speed(
    positions=positions,
    fps=100,
    pixel_to_mm=0.3125
)
# Retorna: Array 1D de valores de velocidad en cm/s
```

##### `compute_speed_from_markers(tracks, model_name, speed_markers, fps, pixel_to_mm) -> np.ndarray`

Calcular velocidad corporal desde múltiples marcadores.

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

Calcular aceleración desde perfil de velocidad.

```python
accel = analyzer.compute_acceleration(speed=speed, fps=100)
# Retorna: Array 1D de valores de aceleración en cm/s^2
```

##### `detect_drag_events(acceleration, fps) -> Tuple[np.ndarray, np.ndarray, dict]`

Detectar eventos de arrastre y recuperación.

```python
drag_idx, recovery_idx, stats = analyzer.detect_drag_events(
    acceleration=accel,
    fps=100
)

# stats contiene:
# - drag_count: Número de eventos de arrastre
# - recovery_count: Número de eventos de recuperación
# - drag_duration: Duración total de arrastre (segundos)
# - recovery_duration: Duración total de recuperación (segundos)
# - peak_acceleration: Aceleración máxima
# - min_acceleration: Aceleración mínima
```

##### `apply_smoothing(data, method, window_size) -> np.ndarray`

Aplicar filtro de suavizado a los datos.

```python
smoothed = analyzer.apply_smoothing(
    data=noisy_data,
    method='moving_average',  # o 'savgol'
    window_size=10
)
```

---

### `analysis.CircularCoordinationAnalyzer`

Analizar coordinación de extremidades usando estadística circular.

```python
from mouse_locomotor_tracker.analysis import CircularCoordinationAnalyzer
```

#### Constructor

```python
CircularCoordinationAnalyzer(
    interpolation_factor: int = 4,  # Factor de interpolación de datos
    smoothing_factor: int = 10      # Ventana de suavizado
)
```

#### Métodos

##### `circular_mean(phi) -> Tuple[float, float]`

Calcular media circular y longitud del vector resultante.

```python
analyzer = CircularCoordinationAnalyzer()
mean_angle, R = analyzer.circular_mean(phi=phase_angles)
# mean_angle: Dirección media en radianes
# R: Longitud resultante [0, 1]
```

**Definición Matemática:**

```
X = mean(cos(phi))
Y = mean(sin(phi))
R = sqrt(X^2 + Y^2)
mean_phi = atan2(Y, X)
```

##### `compute_limb_coordination(stride_0, stride_1, mov_duration) -> Tuple`

Calcular coordinación entre dos extremidades.

```python
phi, R, mean_phase_deg, n_steps = analyzer.compute_limb_coordination(
    stride_0=left_hind_stride,
    stride_1=right_hind_stride,
    mov_duration=30.0
)
# phi: Array de ángulos de fase por ciclo
# R: Longitud resultante (fuerza de coordinación)
# mean_phase_deg: Fase media en grados
# n_steps: Número de pasos detectados
```

##### `analyze_all_limb_pairs(tracks_dict, limb_pairs, mov_duration) -> dict`

Analizar todos los pares de extremidades definidos.

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

Interpretar patrón de coordinación.

```python
pattern = analyzer.interpret_coordination(R=0.9, mean_phase=180.0)
# Retorna: 'alternating'

# Posibles retornos:
# - 'synchronized': En fase (0 grados)
# - 'alternating': Anti-fase (180 grados)
# - 'leading': ~90 grados adelante
# - 'lagging': ~90 grados atrás
# - 'weak_coordination': 0.3 < R < 0.7
# - 'no_coordination': R < 0.3
```

##### `measure_cycles(stride) -> Tuple[int, np.ndarray]`

Detectar ciclos de marcha en datos de zancada.

```python
n_cycles, peak_indices = analyzer.measure_cycles(stride=stride_array)
```

##### `iqr_mean(data) -> float`

Calcular media después de remover outliers via método IQR.

```python
robust_mean = analyzer.iqr_mean(data=noisy_array)
```

---

### `analysis.GaitCycleDetector`

Detectar y analizar ciclos de marcha.

```python
from mouse_locomotor_tracker.analysis import GaitCycleDetector
```

#### Constructor

```python
GaitCycleDetector(
    min_peak_distance: int = None,     # Cuadros mín entre picos (auto si None)
    interpolation_factor: int = 4,     # Factor de interpolación
    smoothing_factor: int = 10         # Ventana de suavizado
)
```

#### Métodos

##### `detect_cycles(stride, fps) -> Tuple[int, np.ndarray, np.ndarray]`

Detectar ciclos de marcha usando detección de picos.

```python
detector = GaitCycleDetector()
n_cycles, peaks, troughs = detector.detect_cycles(
    stride=stride_array,
    fps=100
)
# n_cycles: Número de ciclos detectados
# peaks: Índices de máximos de zancada
# troughs: Índices de mínimos de zancada
```

##### `compute_cadence(stride, duration) -> float`

Calcular frecuencia de paso.

```python
cadence = detector.compute_cadence(
    stride=stride_array,
    duration=30.0  # segundos
)
# Retorna: Pasos por segundo (Hz)
```

##### `compute_stride_length(cadence, avg_speed) -> float`

Calcular longitud promedio de zancada.

```python
stride_length = detector.compute_stride_length(
    cadence=4.0,      # Hz
    avg_speed=16.0    # cm/s
)
# Retorna: 4.0 cm (stride_length = speed / cadence)
```

##### `analyze_gait_regularity(stride, fps) -> dict`

Analizar regularidad del ciclo de marcha.

```python
regularity = detector.analyze_gait_regularity(
    stride=stride_array,
    fps=100
)

# Retorna:
# {
#     'mean_cycle_duration': 0.25,      # segundos
#     'std_cycle_duration': 0.02,       # segundos
#     'cv_cycle_duration': 0.08,        # coeficiente de variación
#     'mean_stride_amplitude': 15.3,    # píxeles
#     'std_stride_amplitude': 2.1       # píxeles
# }
```

##### `interpolate_stride(stride, duration) -> np.ndarray`

Interpolar datos de zancada para mejor detección.

```python
interpolated = detector.interpolate_stride(
    stride=stride_array,
    duration=30.0
)
# Retorna: Array con longitud = original * interpolation_factor
```

##### `compute_all_limb_metrics(limb_strides, duration, avg_speed, fps) -> dict`

Calcular métricas para todas las extremidades.

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

Detectar eventos específicos de marcha (contacto, despegue).

```python
events = detector.detect_gait_events(stride=stride_array, fps=100)

# Retorna:
# {
#     'peaks': np.array([...]),        # Índices de máxima extensión
#     'troughs': np.array([...]),      # Índices de máxima flexión
#     'stance_start': np.array([...]), # Índices de contacto de pata
#     'swing_start': np.array([...])   # Índices de despegue de pata
# }
```

---

### `analysis.LocomotorPipeline`

Pipeline de análisis principal que integra todos los módulos.

```python
from mouse_locomotor_tracker import LocomotorPipeline
```

#### Constructor

```python
LocomotorPipeline(config: dict = None)
```

**Opciones de Config:**

| Clave | Tipo | Default | Descripción |
|-------|------|---------|-------------|
| `smoothing_factor` | `int` | `10` | Ventana de suavizado de velocidad |
| `interpolation_factor` | `int` | `4` | Interpolación de zancada |
| `speed_threshold` | `float` | `5.0` | Umbral de movimiento (cm/s) |
| `drag_threshold` | `float` | `0.25` | Umbral de evento de arrastre (s) |

#### Métodos

##### `process_tracks(tracks, metadata, model_name, markers, limb_pairs, speed_markers) -> dict`

Ejecutar pipeline de análisis completo.

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

**Retorna:** Diccionario con estructura:

```python
{
    'metadata': {...},      # Metadatos de video
    'velocity': {...},      # Resultados de velocidad y aceleración
    'coordination': {...},  # Resultados de coordinación de extremidades
    'gait_cycles': {...},   # Métricas de ciclo de marcha
    'summary': {...}        # Estadísticas agregadas
}
```

##### `export_results(output_path, format) -> None`

Exportar resultados a archivo.

```python
pipeline.export_results("results.json", format="json")
pipeline.export_results("summary.csv", format="csv")
```

**Formatos Soportados:**

| Formato | Contenido | Extensión de Archivo |
|---------|-----------|---------------------|
| `json` | Resultados completos | `.json` |
| `csv` | Solo resumen | `.csv` |

---

## Módulo de Visualización

### `visualization.Plotter`

Utilidades de visualización estática.

```python
from mouse_locomotor_tracker.visualization import Plotter
```

#### Métodos

##### `plot_speed_profile(speed, fps, ax=None) -> matplotlib.axes.Axes`

Graficar velocidad en el tiempo.

```python
plotter = Plotter()
ax = plotter.plot_speed_profile(speed=speed_array, fps=100)
```

##### `plot_coordination_polar(coordination_results, ax=None) -> matplotlib.axes.Axes`

Crear gráfico polar de coordinación.

```python
ax = plotter.plot_coordination_polar(
    coordination_results=results['coordination']
)
```

##### `plot_gait_metrics(gait_results, ax=None) -> matplotlib.axes.Axes`

Graficar comparación de métricas de marcha.

```python
ax = plotter.plot_gait_metrics(
    gait_results=results['gait_cycles']
)
```

##### `create_summary_figure(results, output_path=None)`

Crear figura de resumen completa.

```python
plotter.create_summary_figure(
    results=results,
    output_path="analysis_summary.png"
)
```

---

## Módulo de Exportación

### `export.JSONExporter`

Exportar resultados a formato JSON.

```python
from mouse_locomotor_tracker.export import JSONExporter
```

#### Métodos

##### `export(results, output_path) -> None`

Exportar resultados a archivo JSON.

```python
exporter = JSONExporter()
exporter.export(results=results, output_path="output.json")
```

---

### `export.CSVExporter`

Exportar resultados a formato CSV.

```python
from mouse_locomotor_tracker.export import CSVExporter
```

#### Métodos

##### `export(results, output_path) -> None`

Exportar estadísticas de resumen a CSV.

```python
exporter = CSVExporter()
exporter.export(results=results, output_path="summary.csv")
```

##### `export_detailed(results, output_dir) -> None`

Exportar resultados detallados a múltiples archivos CSV.

```python
exporter.export_detailed(
    results=results,
    output_dir="results/"
)
# Crea:
# - results/velocity.csv
# - results/coordination.csv
# - results/gait_cycles.csv
# - results/summary.csv
```

---

## Definiciones de Tipos

### Tipos Comunes

```python
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd

# Datos de posición
Positions = np.ndarray  # Forma: (n_frames, 2)

# Datos de zancada
StrideArray = np.ndarray  # Forma: (n_frames,)

# Ángulos de fase
PhaseArray = np.ndarray  # Forma: (n_cycles,)

# DataFrame de Tracking
TrackingData = pd.DataFrame  # Columnas MultiIndex

# Diccionario de resultados
AnalysisResults = Dict[str, Any]
```

### Esquemas de Tipos de Resultado

```python
# Resultados de Velocidad
VelocityResults = {
    'mean_speed': float,        # cm/s
    'max_speed': float,         # cm/s
    'min_speed': float,         # cm/s
    'std_speed': float,         # cm/s
    'speed_profile': List[float]
}

# Resultados de Coordinación
CoordinationResults = {
    'pair_name': {
        'R': float,              # [0, 1]
        'mean_phase_deg': float, # [-180, 180]
        'n_steps': int
    }
}

# Resultados de Marcha
GaitResults = {
    'limb_name': {
        'cadence': float,           # Hz
        'stride_length': float,     # cm
        'n_cycles': int,
        'mean_cycle_duration': float,  # segundos
        'cv_cycle_duration': float     # [0, 1]
    }
}

# Resultados de Resumen
SummaryResults = {
    'duration': float,              # segundos
    'mean_speed_cm_s': float,
    'mean_coordination_R': float,
    'mean_cadence_hz': float,
    'mean_stride_length_cm': float
}
```

---

## Manejo de Errores

### Excepciones Comunes

```python
# ValueError para entrada inválida
try:
    analyzer.compute_speed(empty_array, fps=100, pixel_to_mm=0.3)
except ValueError as e:
    print(f"Entrada inválida: {e}")

# KeyError para marcadores faltantes
try:
    speed = analyzer.compute_speed_from_markers(
        tracks, model_name, ['nonexistent_marker'], fps, pixel_to_mm
    )
except KeyError as e:
    print(f"Marcador faltante: {e}")
```

### Funciones de Validación

```python
from mouse_locomotor_tracker.utils import validate_tracks, validate_metadata

# Validar estructura de datos de tracking
is_valid, errors = validate_tracks(tracks)
if not is_valid:
    print(f"Datos de tracking inválidos: {errors}")

# Validar metadatos
is_valid, errors = validate_metadata(metadata)
```

---

## Información de Versión

```python
import mouse_locomotor_tracker

print(mouse_locomotor_tracker.__version__)  # '0.1.0'
print(mouse_locomotor_tracker.__author__)   # 'Stride Labs'
```
