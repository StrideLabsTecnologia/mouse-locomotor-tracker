# Política de Seguridad

## Versiones Soportadas

| Versión | Soportada          |
| ------- | ------------------ |
| 1.1.x   | :white_check_mark: |
| 1.0.x   | :white_check_mark: |
| < 1.0   | :x:                |

## Reportar una Vulnerabilidad

Tomamos la seguridad muy en serio en Mouse Locomotor Tracker. Si descubres una vulnerabilidad de seguridad, por favor sigue estos pasos:

### 1. NO Crees un Issue Público

Las vulnerabilidades de seguridad no deben reportarse a través de issues públicos de GitHub.

### 2. Reporta de Forma Privada

Envía un correo electrónico a: **security@stridelabs.cl**

Incluye:
- Descripción de la vulnerabilidad
- Pasos para reproducir
- Impacto potencial
- Corrección sugerida (si la hay)

### 3. Cronograma de Respuesta

| Fase | Plazo |
|------|-------|
| Respuesta inicial | Dentro de 48 horas |
| Evaluación de vulnerabilidad | Dentro de 7 días |
| Desarrollo de corrección | Dentro de 30 días |
| Divulgación pública | Después de liberar la corrección |

### 4. Qué Esperar

- Reconocimiento de tu reporte dentro de 48 horas
- Actualizaciones regulares sobre el progreso
- Crédito en el aviso de seguridad (si lo deseas)
- NO tomaremos acciones legales contra investigadores que sigan la divulgación responsable

## Medidas de Seguridad

### Seguridad del Código

- **Análisis Estático**: Linter de seguridad Bandit en CI/CD
- **Escaneo de Dependencias**: Auditorías regulares con `pip-audit`
- **Verificación de Tipos**: MyPy para seguridad de tipos
- **Pre-commit Hooks**: Verificaciones de seguridad automatizadas

### Seguridad de Docker

- Ejecución con usuario no-root
- Builds multi-etapa (superficie de ataque mínima)
- Sin secretos en las imágenes
- Sistemas de archivos de solo lectura donde sea posible

### Manejo de Datos

- Sin recolección de datos personales
- Archivos de video procesados solo localmente
- Sin llamadas de red durante el análisis
- Los archivos de salida contienen solo datos de tracking

## Consideraciones de Seguridad Conocidas

### Validación de Entrada

Todas las entradas de video son validadas para:
- Compatibilidad de formato de archivo
- Tamaño máximo de archivo (configurable)
- Verificaciones de seguridad de códec

### Sanitización de Salida

- Las rutas de archivos son sanitizadas
- Sin ejecución de código arbitrario en las salidas
- Las exportaciones JSON/CSV son correctamente escapadas

## Dependencias

Monitoreamos y actualizamos dependencias regularmente:

```bash
# Verificar vulnerabilidades conocidas
pip-audit

# Actualizar dependencias
pip install --upgrade -r requirements.txt
```

### Dependencias Clave

| Paquete | Notas de Seguridad |
|---------|-------------------|
| OpenCV | Solo releases oficiales |
| NumPy | Sin vulnerabilidades conocidas |
| Pandas | Sin vulnerabilidades conocidas |
| h5py | Seguro solo para archivos locales |

## Mejores Prácticas para Usuarios

1. **Mantente Actualizado**: Siempre usa la última versión
2. **Verifica Descargas**: Comprueba checksums SHA256
3. **Docker**: Usa solo imágenes oficiales
4. **Archivos de Entrada**: Solo procesa archivos de video confiables
5. **Directorio de Salida**: Usa directorios de salida dedicados

## Badges de Seguridad

- Pre-commit: Linter de seguridad Bandit
- CI/CD: Escaneo de seguridad automatizado
- Dependencias: Auditorías regulares

---

¡Gracias por ayudar a mantener Mouse Locomotor Tracker seguro!
