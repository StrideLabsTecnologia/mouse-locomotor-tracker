# Registro de Cambios

Todos los cambios notables en Mouse Locomotor Tracker serán documentados en este archivo.

El formato está basado en [Keep a Changelog](https://keepachangelog.com/es-ES/1.1.0/),
y este proyecto adhiere a [Versionado Semántico](https://semver.org/lang/es/).

## [Sin Publicar]

---

## [1.1.0] - 2025-01-14

### Agregado
- **Extracción Automática de Video**: El dashboard ahora extrae frames automáticamente al iniciar
- **assets/video.mp4**: Video de demostración incluido en el proyecto (3.2 MB)
- Función `extract_video_frames()` con cache de Streamlit para evitar re-extracción
- Mensajes de estado si falta OpenCV o el video

### Cambiado
- Dashboard usa rutas dinámicas (`VIDEO_PATH`, `VIDEO_FRAMES_DIR`) en lugar de hardcoded
- Mejor manejo de errores cuando el video no está disponible

### Corregido
- Video no se mostraba porque los frames no existían

---

## [1.0.0] - 2025-01-12

### Agregado
- **Motor de Análisis Principal**
  - Cálculo de velocidad con suavizado configurable
  - Detección de zancadas usando algoritmos de búsqueda de picos
  - Clasificación de patrones de marcha (caminar, trote, galope)
  - Métricas biomecánicas (cadencia, longitud de zancada, simetría)

- **Sistema de Tracking**
  - Tracking basado en movimiento usando diferencia de frames
  - Detección con ROI restringido para experimentos en cinta
  - Arquitectura multi-tracker (movimiento, anatómico, DLC)
  - Integración con DeepLabCut para estimación de pose

- **Suite de Visualización**
  - Overlay de tracking en tiempo real sobre video
  - Mapas de calor de trayectoria con estimación de densidad
  - Gráficos de series temporales de velocidad/aceleración
  - Diagramas de fase de marcha
  - Exportación de figuras para publicación

- **Formatos de Exportación**
  - Exportación CSV con columnas configurables
  - Exportación JSON con metadatos completos
  - Exportación HDF5 con compresión (formato científico)
  - Exportación NWB (estándar Neurodata Without Borders)
  - Detección automática de formato por extensión de archivo

- **CLI Profesional**
  - Interfaz de comandos basada en Typer
  - Salida de terminal enriquecida con barras de progreso
  - Múltiples flags de formato de salida
  - Modo preview para verificaciones rápidas

- **Soporte Docker**
  - Dockerfile multi-etapa para imágenes optimizadas
  - Docker Compose para desarrollo/testing/producción
  - Servicio de Jupyter notebook para exploración

- **Aseguramiento de Calidad**
  - Pre-commit hooks (black, isort, ruff, mypy, bandit)
  - Pipeline CI/CD con GitHub Actions
  - Testing multi-versión de Python (3.9-3.12)
  - Cobertura de código con integración Codecov

- **Documentación**
  - README completo con badges
  - Documentación de API con ejemplos
  - Guías de contribución
  - Licencia MIT

### Detalles Técnicos
- Compatibilidad con Python 3.9+
- Procesamiento de video basado en OpenCV
- NumPy/SciPy para cálculos numéricos
- Pandas para manipulación de datos
- Type hints en todo el código

---

## [0.2.0] - 2025-01-11

### Agregado
- Procesador de tracking solo por movimiento
- Restricciones de ROI para área de cinta
- Algoritmo de diferencia de frames
- Cálculos básicos de velocidad

### Corregido
- Detección de falsos positivos en objetos estáticos (tornillos)
- Precisión de tracking mejorada de 70% a 100%

---

## [0.1.0] - 2025-01-10

### Agregado
- Estructura inicial del proyecto
- Carga y procesamiento básico de video
- Tracking simple de centroide
- Funcionalidad de exportación CSV

### Problemas Conocidos
- Detección de objetos estáticos causando falsos positivos
- Sin restricciones de ROI

---

[Sin Publicar]: https://github.com/StrideLabsTecnologia/mouse-locomotor-tracker/compare/v1.1.0...HEAD
[1.1.0]: https://github.com/StrideLabsTecnologia/mouse-locomotor-tracker/compare/v1.0.0...v1.1.0
[1.0.0]: https://github.com/StrideLabsTecnologia/mouse-locomotor-tracker/compare/v0.2.0...v1.0.0
[0.2.0]: https://github.com/StrideLabsTecnologia/mouse-locomotor-tracker/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/StrideLabsTecnologia/mouse-locomotor-tracker/releases/tag/v0.1.0
