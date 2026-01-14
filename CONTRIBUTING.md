# Contribuir a Mouse Locomotor Tracker

¡Gracias por tu interés en contribuir a MLT! Este documento proporciona guías e instrucciones para contribuir.

## Código de Conducta

Al participar en este proyecto, aceptas mantener un ambiente respetuoso e inclusivo.

## Cómo Contribuir

### Reportar Errores

1. Verifica los issues existentes para evitar duplicados
2. Usa la plantilla de reporte de errores
3. Incluye:
   - Versión de Python
   - Sistema operativo y versión
   - Pasos para reproducir
   - Comportamiento esperado vs actual
   - Mensajes de error/stack traces

### Sugerir Funcionalidades

1. Revisa los issues/discusiones existentes
2. Usa la plantilla de solicitud de funcionalidad
3. Describe:
   - Caso de uso
   - Solución propuesta
   - Alternativas consideradas

### Pull Requests

1. Haz fork del repositorio
2. Crea una rama de funcionalidad: `git checkout -b feature/tu-funcionalidad`
3. Realiza tus cambios
4. Ejecuta tests: `pytest tests/`
5. Ejecuta linters: `pre-commit run --all-files`
6. Haz commit con conventional commits: `git commit -m 'feat: agregar funcionalidad'`
7. Push: `git push origin feature/tu-funcionalidad`
8. Abre un Pull Request

## Configuración de Desarrollo

```bash
# Clonar tu fork
git clone https://github.com/TU_USUARIO/mouse-locomotor-tracker.git
cd mouse-locomotor-tracker

# Crear entorno virtual
python -m venv .venv
source .venv/bin/activate  # o .venv\Scripts\activate en Windows

# Instalar dependencias de desarrollo
pip install -e ".[dev]"

# Instalar pre-commit hooks
pre-commit install
```

## Estilo de Código

- **Python**: Seguir PEP 8, aplicado por Black y Ruff
- **Imports**: Ordenados con isort
- **Tipos**: Agregar type hints donde sea posible
- **Docstrings**: Estilo Google

### Ejemplo

```python
def calculate_velocity(
    positions: np.ndarray,
    timestamps: np.ndarray,
) -> np.ndarray:
    """
    Calcular velocidad instantánea desde datos de posición.

    Args:
        positions: Array de coordenadas (x, y), forma (N, 2).
        timestamps: Array de marcas de tiempo en segundos, forma (N,).

    Returns:
        Array de velocidades en cm/s, forma (N-1,).

    Raises:
        ValueError: Si los arrays tienen longitudes diferentes.
    """
    if len(positions) != len(timestamps):
        raise ValueError("Longitudes de array no coinciden")

    dx = np.diff(positions[:, 0])
    dy = np.diff(positions[:, 1])
    dt = np.diff(timestamps)

    return np.sqrt(dx**2 + dy**2) / dt
```

## Mensajes de Commit

Usa [Conventional Commits](https://www.conventionalcommits.org/):

- `feat:` Nueva funcionalidad
- `fix:` Corrección de error
- `docs:` Documentación
- `style:` Estilo de código (sin cambio de lógica)
- `refactor:` Refactorización de código
- `test:` Tests
- `chore:` Mantenimiento

## Testing

```bash
# Ejecutar todos los tests
pytest tests/

# Con cobertura
pytest tests/ --cov=analysis --cov=tracking --cov-report=html

# Test específico
pytest tests/test_velocity.py -v
```

## Documentación

- Actualiza docstrings para cualquier cambio de API
- Actualiza README.md si agregas funcionalidades
- Agrega ejemplos para nueva funcionalidad

## ¿Preguntas?

Abre una discusión o issue. ¡Estamos encantados de ayudar!

---

¡Gracias por contribuir!
