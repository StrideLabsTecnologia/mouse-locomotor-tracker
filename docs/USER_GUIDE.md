# Guía de Usuario

Una guía completa para usar Mouse Locomotor Tracker para analizar la locomoción de roedores.

## Tabla de Contenidos

1. [Comenzando](#comenzando)
2. [Procesando un Video](#procesando-un-video)
3. [Configurando Marcadores](#configurando-marcadores)
4. [Entendiendo los Resultados](#entendiendo-los-resultados)
5. [Interpretando Métricas](#interpretando-métricas)
6. [Uso Avanzado](#uso-avanzado)
7. [Ejemplos](#ejemplos)

## Comenzando

### Flujo de Trabajo Básico

```
+-------------+     +-------------+     +-------------+     +-------------+
|   Grabar    |     |   Seguir    |     |   Analizar  |     |   Exportar  |
|   Video     +---->+   Poses     +---->+   Movimiento+---->+   Resultados|
+-------------+     +-------------+     +-------------+     +-------------+
     ^                    |                   |                   |
     |                    v                   v                   v
  Config.            DeepLabCut          Esta Librería        JSON/CSV
  Cámara              o Manual                                Reportes
```

### Prerrequisitos

1. **Grabaciones de video** de locomoción de ratón
2. **Datos de seguimiento de pose** de DeepLabCut (o equivalente)
3. **Entorno Python** con Mouse Locomotor Tracker instalado

## Procesando un Video

### Paso 1: Preparar Tus Datos

Tus datos de seguimiento deben estar en formato HDF5 o CSV de DeepLabCut:

```
tracking_data/
+-- video_01DLC_resnet50_mouseJan1shuffle1_500000.h5
+-- video_01DLC_resnet50_mouseJan1shuffle1_500000_labeled.mp4
+-- video_01.avi
```

### Paso 2: Cargar Datos de Seguimiento

```python
import pandas as pd
from mouse_locomotor_tracker import LocomotorPipeline
from mouse_locomotor_tracker.tracking import VideoMetadata, MarkerSet

# Cargar datos de seguimiento DeepLabCut
tracks = pd.read_hdf("tracking_data/video_01DLC_resnet50_mouseJan1shuffle1_500000.h5")

# Verificar la estructura
print("Columnas:", tracks.columns.levels)
print("Forma:", tracks.shape)
```

### Paso 3: Configurar Metadatos del Video

```python
# Opción 1: Configuración manual
metadata = VideoMetadata(
    duration=30.0,        # Duración total en segundos
    fps=100,              # Cuadros por segundo
    n_frames=3000,        # Número total de cuadros
    width=640,            # Ancho de imagen en píxeles
    height=480,           # Alto de imagen en píxeles
    pixel_width_mm=0.3125 # Tamaño físico de un píxel (mm)
)

# Opción 2: Extraer del archivo de video
from mouse_locomotor_tracker.tracking import extract_video_metadata
metadata = extract_video_metadata("tracking_data/video_01.avi")
```

### Paso 4: Configurar Marcadores

```python
# Marcadores estándar de vista ventral
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
        "LH_RH": ("hindL", "hindR"),      # Traseras izquierda-derecha
        "LH_LF": ("hindL", "foreL"),       # Ipsilateral izquierda
        "RH_RF": ("hindR", "foreR"),       # Ipsilateral derecha
        "LF_RH": ("foreL", "hindR"),       # Diagonal
        "RF_LH": ("foreR", "hindL"),       # Diagonal
        "LF_RF": ("foreL", "foreR"),       # Delanteras izquierda-derecha
    },
    speed_markers=["snout", "torso", "torsoL", "torsoR", "tail"]
)
```

### Paso 5: Ejecutar Análisis

```python
# Crear pipeline
pipeline = LocomotorPipeline(config={
    'smoothing_factor': 10,
    'speed_threshold': 5.0,
    'interpolation_factor': 4
})

# Obtener nombre del modelo de los datos de seguimiento
model_name = tracks.columns.get_level_values(0)[0]

# Procesar seguimiento
results = pipeline.process_tracks(
    tracks=tracks,
    metadata=metadata,
    model_name=model_name,
    markers=markers.markers,
    limb_pairs=markers.limb_pairs,
    speed_markers=markers.speed_markers
)

# Imprimir resumen
print(f"\n=== Resumen del Análisis ===")
print(f"Duración: {results['summary']['duration']:.1f} s")
print(f"Velocidad Media: {results['summary']['mean_speed_cm_s']:.2f} cm/s")
print(f"Cadencia Media: {results['summary']['mean_cadence_hz']:.2f} Hz")
print(f"Coordinación (R): {results['summary']['mean_coordination_R']:.3f}")
```

### Paso 6: Exportar Resultados

```python
# Exportar a JSON (resultados completos)
pipeline.export_results("results/video_01_analysis.json", format="json")

# Exportar a CSV (solo resumen)
pipeline.export_results("results/video_01_summary.csv", format="csv")
```

## Configurando Marcadores

### Conjuntos de Marcadores Estándar

#### Vista Ventral (Cámara Inferior)

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

#### Vista Lateral (Cámara de Costado)

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
    limb_pairs={},  # Ángulos articulares en lugar de pares de extremidades
    speed_markers=["hip", "crest"]
)
```

### Configuración Personalizada de Marcadores

```python
# Crear conjunto de marcadores personalizado
custom_markers = MarkerSet(
    name="custom_experiment",
    markers=["head", "body", "leftPaw", "rightPaw", "tailBase"],
    limb_pairs={
        "left_right": ("leftPaw", "rightPaw"),
    },
    speed_markers=["head", "body", "tailBase"]
)
```

### Cargando desde YAML

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

## Entendiendo los Resultados

### Estructura de Resultados

```python
results = {
    'metadata': {
        'dur': 30.0,           # Duración (segundos)
        'fps': 100,            # Cuadros por segundo
        'nFrame': 3000,        # Total de cuadros
        'imW': 640,            # Ancho de imagen
        'imH': 480,            # Alto de imagen
        'xPixW': 0.3125        # Ancho de píxel (mm)
    },
    'velocity': {
        'mean_speed': 15.2,    # Velocidad media (cm/s)
        'max_speed': 45.8,     # Velocidad máxima (cm/s)
        'min_speed': 0.0,      # Velocidad mínima (cm/s)
        'std_speed': 8.3,      # Desviación estándar de velocidad
        'speed_profile': [...]  # Serie temporal de velocidad
    },
    'coordination': {
        'LH_RH': {
            'R': 0.92,              # Longitud resultante (0-1)
            'mean_phase_deg': 175.3, # Fase media (grados)
            'n_steps': 45           # Número de pasos
        },
        # ... otros pares de extremidades
    },
    'gait_cycles': {
        'hindL': {
            'cadence': 4.2,         # Pasos por segundo (Hz)
            'stride_length': 3.6,   # Longitud de zancada (cm)
            'n_cycles': 126         # Ciclos totales detectados
        },
        # ... otras extremidades
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

### Accediendo a Resultados Específicos

```python
# Velocidad
mean_speed = results['velocity']['mean_speed']
speed_profile = results['velocity']['speed_profile']

# Coordinación para par específico de extremidades
lh_rh_R = results['coordination']['LH_RH']['R']
lh_rh_phase = results['coordination']['LH_RH']['mean_phase_deg']

# Métricas de marcha para extremidad específica
hindL_cadence = results['gait_cycles']['hindL']['cadence']
hindL_stride = results['gait_cycles']['hindL']['stride_length']

# Estadísticas de resumen
summary = results['summary']
```

## Interpretando Métricas

### Métricas de Velocidad

| Métrica | Descripción | Rango Típico | Notas |
|---------|-------------|--------------|-------|
| Velocidad Media | Velocidad promedio de locomoción | 5-30 cm/s | Depende del paradigma |
| Velocidad Máx | Velocidad instantánea pico | 30-80 cm/s | Durante aceleración |
| Std Velocidad | Variabilidad de velocidad | 3-15 cm/s | Mayor = más variable |

### Métricas de Coordinación

#### Longitud Resultante (R)

| Valor R | Interpretación |
|---------|----------------|
| 0.9-1.0 | Coordinación fuerte |
| 0.7-0.9 | Coordinación buena |
| 0.5-0.7 | Coordinación moderada |
| 0.3-0.5 | Coordinación débil |
| 0.0-0.3 | Sin coordinación |

#### Ángulo de Fase

| Fase | Patrón | Descripción |
|------|--------|-------------|
| 0 grados | En fase | Las extremidades se mueven juntas |
| 180 grados | Anti-fase | Las extremidades alternan |
| 90 grados | Cuarto de fase | Una extremidad adelanta 1/4 de ciclo |

#### Patrones de Marcha

```
TROTE (diagonal alternando):
    LH_RH: ~180 grados (alternando)
    LF_RH: ~0 grados (sincronizado)
    RF_LH: ~0 grados (sincronizado)

PASO (ipsilateral):
    LH_LF: ~0 grados (sincronizado)
    RH_RF: ~0 grados (sincronizado)
    LH_RH: ~180 grados (alternando)

SALTO/GALOPE:
    LH_RH: ~0 grados (sincronizado)
    LF_RF: ~0 grados (sincronizado)
    Delante-atrás: ~180 grados
```

### Métricas de Ciclo de Marcha

| Métrica | Descripción | Rango Típico |
|---------|-------------|--------------|
| Cadencia | Pasos por segundo | 2-8 Hz |
| Longitud de Zancada | Distancia por paso | 2-6 cm |
| Duración de Ciclo | Tiempo por ciclo | 0.1-0.5 s |

## Uso Avanzado

### Procesamiento por Lotes

```python
import os
from pathlib import Path

def process_batch(data_dir: str, output_dir: str):
    """Procesar todos los archivos de seguimiento en un directorio."""

    data_path = Path(data_dir)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Encontrar todos los archivos de seguimiento HDF5
    tracking_files = list(data_path.glob("*DLC*.h5"))

    results_list = []

    for track_file in tracking_files:
        print(f"Procesando: {track_file.name}")

        # Cargar datos de seguimiento
        tracks = pd.read_hdf(track_file)
        model_name = tracks.columns.get_level_values(0)[0]

        # Encontrar video correspondiente
        video_name = track_file.stem.split("DLC")[0] + ".avi"
        video_path = data_path / video_name

        # Obtener metadatos
        if video_path.exists():
            metadata = extract_video_metadata(str(video_path))
        else:
            # Usar valores por defecto
            metadata = VideoMetadata(
                duration=len(tracks) / 100,
                fps=100,
                n_frames=len(tracks),
                width=640,
                height=480,
                pixel_width_mm=0.3125
            )

        # Procesar
        pipeline = LocomotorPipeline()
        results = pipeline.process_tracks(
            tracks=tracks,
            metadata=metadata,
            model_name=model_name,
            markers=MOUSE_VENTRAL.markers,
            limb_pairs=MOUSE_VENTRAL.limb_pairs,
            speed_markers=MOUSE_VENTRAL.speed_markers
        )

        # Agregar nombre de archivo a resultados
        results['filename'] = track_file.name
        results_list.append(results)

        # Exportar resultados individuales
        output_file = output_path / f"{track_file.stem}_analysis.json"
        pipeline.export_results(str(output_file), format="json")

    # Crear resumen agregado
    summaries = [r['summary'] | {'filename': r['filename']} for r in results_list]
    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(output_path / "batch_summary.csv", index=False)

    return results_list

# Ejecutar procesamiento por lotes
results = process_batch("data/tracking/", "results/")
```

### Pipeline de Análisis Personalizado

```python
from mouse_locomotor_tracker.analysis import (
    VelocityAnalyzer,
    CircularCoordinationAnalyzer,
    GaitCycleDetector
)

class CustomAnalysisPipeline:
    """Pipeline personalizado con análisis especializado."""

    def __init__(self, config: dict = None):
        self.config = config or {}

        # Inicializar analizadores
        self.velocity_analyzer = VelocityAnalyzer(
            smoothing_factor=self.config.get('smoothing_factor', 10),
            speed_threshold=self.config.get('speed_threshold', 5.0)
        )
        self.coord_analyzer = CircularCoordinationAnalyzer(
            interpolation_factor=self.config.get('interpolation_factor', 4)
        )
        self.gait_detector = GaitCycleDetector()

    def analyze_episode(self, tracks, metadata, start_frame, end_frame):
        """Analizar un episodio específico dentro de la grabación."""

        # Extraer episodio
        episode_tracks = tracks.iloc[start_frame:end_frame]
        episode_duration = (end_frame - start_frame) / metadata.fps

        # Ejecutar análisis en episodio
        # ... código de análisis personalizado

        return episode_results

    def detect_movement_bouts(self, speed_profile, threshold=5.0):
        """Detectar ráfagas de movimiento basadas en umbral de velocidad."""

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

### Visualización

```python
import matplotlib.pyplot as plt
import numpy as np

def plot_analysis_results(results: dict, output_path: str = None):
    """Crear visualización completa de resultados de análisis."""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Perfil de velocidad
    ax1 = axes[0, 0]
    speed = results['velocity'].get('speed_profile', [])
    if speed:
        ax1.plot(speed[:500], 'b-', alpha=0.7)
        ax1.axhline(results['velocity']['mean_speed'], color='r', linestyle='--',
                    label=f"Media: {results['velocity']['mean_speed']:.1f} cm/s")
        ax1.set_xlabel('Cuadro')
        ax1.set_ylabel('Velocidad (cm/s)')
        ax1.set_title('Perfil de Velocidad')
        ax1.legend()

    # 2. Gráfico polar de coordinación
    ax2 = axes[0, 1]
    ax2 = plt.subplot(222, projection='polar')

    for pair_name, data in results['coordination'].items():
        angle = np.deg2rad(data['mean_phase_deg'])
        r = data['R']
        ax2.annotate('', xy=(angle, r), xytext=(0, 0),
                    arrowprops=dict(arrowstyle='->', color='blue', lw=2))
        ax2.text(angle, r + 0.1, pair_name, fontsize=8, ha='center')

    ax2.set_title('Coordinación de Extremidades')
    ax2.set_ylim(0, 1.2)

    # 3. Comparación de cadencia
    ax3 = axes[1, 0]
    limbs = list(results['gait_cycles'].keys())
    cadences = [results['gait_cycles'][l]['cadence'] for l in limbs]
    ax3.bar(limbs, cadences, color='steelblue')
    ax3.set_ylabel('Cadencia (Hz)')
    ax3.set_title('Cadencia por Extremidad')
    ax3.tick_params(axis='x', rotation=45)

    # 4. Comparación de longitud de zancada
    ax4 = axes[1, 1]
    stride_lengths = [results['gait_cycles'][l]['stride_length'] for l in limbs]
    ax4.bar(limbs, stride_lengths, color='coral')
    ax4.set_ylabel('Longitud de Zancada (cm)')
    ax4.set_title('Longitud de Zancada')
    ax4.tick_params(axis='x', rotation=45)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

# Uso
plot_analysis_results(results, "results/analysis_plot.png")
```

## Ejemplos

### Ejemplo 1: Análisis Básico

```python
"""Ejemplo de análisis de locomoción básico."""

import pandas as pd
from mouse_locomotor_tracker import LocomotorPipeline
from mouse_locomotor_tracker.tracking import VideoMetadata, MOUSE_VENTRAL

# Cargar datos
tracks = pd.read_hdf("data/example_tracking.h5")
model_name = tracks.columns.get_level_values(0)[0]

# Configurar
metadata = VideoMetadata(
    duration=30.0, fps=100, n_frames=3000,
    width=640, height=480, pixel_width_mm=0.3125
)

# Analizar
pipeline = LocomotorPipeline()
results = pipeline.process_tracks(
    tracks, metadata, model_name,
    MOUSE_VENTRAL.markers,
    MOUSE_VENTRAL.limb_pairs,
    MOUSE_VENTRAL.speed_markers
)

# Reportar
print(f"Velocidad: {results['summary']['mean_speed_cm_s']:.1f} cm/s")
print(f"Cadencia: {results['summary']['mean_cadence_hz']:.1f} Hz")
print(f"Coordinación: {results['summary']['mean_coordination_R']:.2f}")
```

### Ejemplo 2: Comparando Grupos

```python
"""Comparar locomoción entre grupos experimentales."""

import pandas as pd
from scipy import stats

def compare_groups(control_files: list, treatment_files: list):
    """Comparar métricas de locomoción entre grupos control y tratamiento."""

    control_results = [analyze_file(f) for f in control_files]
    treatment_results = [analyze_file(f) for f in treatment_files]

    # Extraer métricas
    control_speeds = [r['summary']['mean_speed_cm_s'] for r in control_results]
    treatment_speeds = [r['summary']['mean_speed_cm_s'] for r in treatment_results]

    control_cadences = [r['summary']['mean_cadence_hz'] for r in control_results]
    treatment_cadences = [r['summary']['mean_cadence_hz'] for r in treatment_results]

    # Comparación estadística
    speed_stat, speed_p = stats.ttest_ind(control_speeds, treatment_speeds)
    cadence_stat, cadence_p = stats.ttest_ind(control_cadences, treatment_cadences)

    print("=== Comparación de Grupos ===")
    print(f"\nVelocidad (cm/s):")
    print(f"  Control: {np.mean(control_speeds):.2f} +/- {np.std(control_speeds):.2f}")
    print(f"  Tratamiento: {np.mean(treatment_speeds):.2f} +/- {np.std(treatment_speeds):.2f}")
    print(f"  p-valor: {speed_p:.4f}")

    print(f"\nCadencia (Hz):")
    print(f"  Control: {np.mean(control_cadences):.2f} +/- {np.std(control_cadences):.2f}")
    print(f"  Tratamiento: {np.mean(treatment_cadences):.2f} +/- {np.std(treatment_cadences):.2f}")
    print(f"  p-valor: {cadence_p:.4f}")
```

### Ejemplo 3: Análisis de Series Temporales

```python
"""Analizar cambios de locomoción en el tiempo dentro de una sesión."""

def analyze_time_windows(results: dict, window_size: int = 500):
    """Analizar métricas en ventanas de tiempo."""

    speed_profile = results['velocity']['speed_profile']

    windows = []
    for i in range(0, len(speed_profile), window_size):
        window = speed_profile[i:i+window_size]
        if len(window) > 100:  # Tamaño mínimo de ventana
            windows.append({
                'start_frame': i,
                'end_frame': i + len(window),
                'mean_speed': np.mean(window),
                'std_speed': np.std(window),
                'max_speed': np.max(window)
            })

    return pd.DataFrame(windows)

# Uso
time_analysis = analyze_time_windows(results)
print(time_analysis)
```

## Siguientes Pasos

- Lee la [Referencia de API](API.md) para documentación detallada de clases
- Consulta la [Guía de Métricas](METRICS.md) para detalles matemáticos
- Revisa [Ejemplos](../examples/) para más casos de uso
