# Guía de Instalación

Instrucciones completas de instalación para Mouse Locomotor Tracker.

## Tabla de Contenidos

1. [Requisitos del Sistema](#requisitos-del-sistema)
2. [Métodos de Instalación](#métodos-de-instalación)
   - [Instalación con pip](#instalación-con-pip)
   - [Instalación con Conda](#instalación-con-conda)
   - [Instalación de Desarrollo](#instalación-de-desarrollo)
3. [Configuración de GPU](#configuración-de-gpu-para-deeplabcut)
4. [Solución de Problemas](#solución-de-problemas)

## Requisitos del Sistema

### Requisitos Mínimos

| Componente | Requisito |
|-----------|-----------|
| SO | Windows 10+, macOS 10.14+, Ubuntu 18.04+ |
| Python | 3.8, 3.9, 3.10, o 3.11 |
| RAM | 8 GB mínimo, 16 GB recomendado |
| Almacenamiento | 2 GB para instalación base |
| GPU | Opcional (requerido para inferencia DeepLabCut) |

### Recomendado para Aceleración GPU

| Componente | Requisito |
|-----------|-----------|
| GPU | NVIDIA con soporte CUDA |
| CUDA | 11.2+ |
| cuDNN | 8.1+ |
| VRAM | 6 GB mínimo |

## Métodos de Instalación

### Instalación con pip

El método más simple para la mayoría de usuarios:

```bash
# Crear y activar entorno virtual (recomendado)
python -m venv locomotor-env
source locomotor-env/bin/activate  # Linux/macOS
# O
locomotor-env\Scripts\activate  # Windows

# Instalar desde PyPI (cuando esté disponible)
pip install mouse-locomotor-tracker

# O instalar desde GitHub
pip install git+https://github.com/stridelabs/mouse-locomotor-tracker.git
```

### Instalación con Conda

Recomendado para usuarios que necesitan integración con DeepLabCut:

```bash
# Crear entorno conda
conda create -n locomotor python=3.10 -y
conda activate locomotor

# Instalar dependencias principales
conda install numpy pandas scipy matplotlib h5py -y

# Instalar DeepLabCut (opcional, para estimación de pose)
conda install -c conda-forge deeplabcut -y

# Clonar e instalar Mouse Locomotor Tracker
git clone https://github.com/stridelabs/mouse-locomotor-tracker.git
cd mouse-locomotor-tracker
pip install -e .
```

### Instalación de Desarrollo

Para contribuidores y desarrolladores:

```bash
# Clonar repositorio
git clone https://github.com/stridelabs/mouse-locomotor-tracker.git
cd mouse-locomotor-tracker

# Crear entorno virtual
python -m venv venv
source venv/bin/activate

# Instalar en modo desarrollo con todas las dependencias
pip install -e ".[dev]"

# Instalar pre-commit hooks
pre-commit install

# Verificar instalación
pytest tests/ -v
```

## Dependencias

### Dependencias Principales

```
numpy>=1.20.0
pandas>=1.3.0
scipy>=1.7.0
matplotlib>=3.4.0
h5py>=3.0.0
PyYAML>=6.0
```

### Dependencias Opcionales

```
# Para integración DeepLabCut
deeplabcut>=2.2.0

# Para procesamiento de video
opencv-python>=4.5.0
ffmpeg-python>=0.2.0

# Para desarrollo
pytest>=7.0.0
pytest-cov>=3.0.0
black>=22.0.0
flake8>=4.0.0
mypy>=0.950
```

## Configuración de GPU para DeepLabCut

DeepLabCut requiere soporte GPU para estimación de pose eficiente. Sigue estos pasos para configuración GPU:

### Configuración GPU NVIDIA (Linux)

```bash
# Verificar driver NVIDIA
nvidia-smi

# Instalar CUDA toolkit (ejemplo Ubuntu)
# Visita: https://developer.nvidia.com/cuda-downloads

# Instalar cuDNN
# Visita: https://developer.nvidia.com/cudnn

# Verificar instalación CUDA
nvcc --version
```

### Configuración GPU NVIDIA (Windows)

1. Descarga e instala drivers NVIDIA desde [nvidia.com/drivers](https://www.nvidia.com/drivers)
2. Instala CUDA Toolkit desde [developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads)
3. Instala cuDNN desde [developer.nvidia.com/cudnn](https://developer.nvidia.com/cudnn)
4. Agrega CUDA al PATH:
   ```
   C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.x\bin
   C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.x\libnvvp
   ```

### Configuración TensorFlow GPU

```bash
# Instalar TensorFlow con soporte GPU
pip install tensorflow-gpu>=2.5.0

# Verificar detección GPU
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

### Instalación Solo CPU

Si no tienes GPU o solo necesitas ejecutar análisis (no estimación de pose):

```bash
# Instalar sin dependencias GPU
pip install mouse-locomotor-tracker[cpu]

# O excluir manualmente paquetes GPU
pip install tensorflow-cpu>=2.5.0
```

## Variables de Entorno

Configura estas variables de entorno para rendimiento óptimo:

```bash
# Linux/macOS (.bashrc o .zshrc)
export MLT_DATA_DIR="$HOME/locomotor_data"
export MLT_CONFIG_DIR="$HOME/.config/locomotor"
export TF_CPP_MIN_LOG_LEVEL=2  # Reducir verbosidad TensorFlow

# Windows (PowerShell)
$env:MLT_DATA_DIR = "$HOME\locomotor_data"
$env:MLT_CONFIG_DIR = "$HOME\.config\locomotor"
$env:TF_CPP_MIN_LOG_LEVEL = 2
```

## Verificación

Verifica tu instalación con estos tests:

```bash
# Verificar versión
python -c "import mouse_locomotor_tracker; print(mouse_locomotor_tracker.__version__)"

# Ejecutar test básico de importación
python -c "
from mouse_locomotor_tracker import LocomotorPipeline
from mouse_locomotor_tracker.analysis import VelocityAnalyzer
from mouse_locomotor_tracker.tracking import VideoMetadata, MarkerSet
print('¡Todas las importaciones exitosas!')
"

# Ejecutar suite de tests
pytest tests/ -v

# Verificar disponibilidad DeepLabCut (opcional)
python -c "
try:
    import deeplabcut
    print(f'Versión DeepLabCut: {deeplabcut.__version__}')
except ImportError:
    print('DeepLabCut no instalado (opcional)')
"
```

## Instalación Docker

Para entornos aislados usando Docker:

```dockerfile
# Dockerfile
FROM python:3.10-slim

WORKDIR /app

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copiar e instalar paquete
COPY . .
RUN pip install --no-cache-dir -e .

# Ejecutar tests
CMD ["pytest", "tests/", "-v"]
```

Construir y ejecutar:

```bash
# Construir imagen
docker build -t locomotor-tracker .

# Ejecutar contenedor
docker run -it locomotor-tracker

# Ejecutar con datos montados
docker run -v /ruta/a/datos:/data locomotor-tracker python analyze.py
```

## Solución de Problemas

### Problemas Comunes

#### Error de Importación: No module named 'mouse_locomotor_tracker'

```bash
# Asegurar que el paquete está instalado
pip list | grep mouse-locomotor

# Reinstalar en modo desarrollo
pip install -e .
```

#### Conflictos de Versión NumPy/SciPy

```bash
# Reinstalar versiones compatibles
pip uninstall numpy scipy -y
pip install numpy>=1.20.0 scipy>=1.7.0
```

#### Errores de Importación DeepLabCut

```bash
# Instalar con conda (recomendado para DLC)
conda install -c conda-forge deeplabcut

# Verificar compatibilidad TensorFlow
python -c "import tensorflow as tf; print(tf.__version__)"
```

#### GPU No Detectada

```bash
# Verificar instalación CUDA
nvcc --version
nvidia-smi

# Verificar GPU TensorFlow
python -c "
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f'Encontradas {len(gpus)} GPU(s)')
else:
    print('No se encontró GPU')
"
```

#### Errores de Memoria con Videos Grandes

```python
# Procesar videos en chunks
from mouse_locomotor_tracker import LocomotorPipeline

pipeline = LocomotorPipeline(config={
    'chunk_size': 10000,  # Procesar 10k frames a la vez
    'memory_efficient': True
})
```

### Problemas Específicos por Plataforma

#### macOS Apple Silicon (M1/M2)

```bash
# Instalar miniforge para ARM64
brew install miniforge

# Crear entorno
conda create -n locomotor python=3.10
conda activate locomotor

# Instalar con paquetes optimizados ARM64
conda install numpy pandas scipy matplotlib -y
pip install tensorflow-macos tensorflow-metal  # Soporte GPU Apple
```

#### Problemas de Rutas Largas en Windows

Habilita rutas largas en el Registro de Windows:
```powershell
# Ejecutar como Administrador
New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" `
    -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force
```

### Obtener Ayuda

Si encuentras problemas no cubiertos aquí:

1. Revisa [GitHub Issues](https://github.com/stridelabs/mouse-locomotor-tracker/issues)
2. Busca issues existentes antes de crear nuevos
3. Proporciona:
   - Sistema operativo y versión
   - Versión de Python (`python --version`)
   - Versiones de paquetes (`pip freeze`)
   - Traceback completo del error
   - Código mínimo para reproducir

## Actualización

### Actualizar a la Última Versión

```bash
# pip
pip install --upgrade mouse-locomotor-tracker

# Desde GitHub
pip install --upgrade git+https://github.com/stridelabs/mouse-locomotor-tracker.git

# Versión de desarrollo
cd mouse-locomotor-tracker
git pull origin main
pip install -e .
```

### Changelog

Ver [CHANGELOG.md](../CHANGELOG.md) para historial de versiones y guías de migración.
