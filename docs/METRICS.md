# Guía de Métricas

Explicación detallada de todas las métricas calculadas por Mouse Locomotor Tracker, incluyendo fórmulas matemáticas e interpretación biológica.

## Tabla de Contenidos

1. [Métricas de Velocidad](#métricas-de-velocidad)
2. [Métricas de Coordinación](#métricas-de-coordinación)
3. [Métricas de Ciclo de Marcha](#métricas-de-ciclo-de-marcha)
4. [Interpretación Biológica](#interpretación-biológica)

---

## Métricas de Velocidad

### Velocidad Instantánea

**Definición:** La tasa de cambio de posición entre cuadros consecutivos.

**Fórmula:**

```
                    sqrt((x[i+1] - x[i])^2 + (y[i+1] - y[i])^2) * pixel_to_mm
speed[i] (cm/s) = ------------------------------------------------------------ * fps / 10
                                            1
```

Donde:
- `x[i], y[i]` = Coordenadas de posición en el cuadro `i` (píxeles)
- `pixel_to_mm` = Tamaño físico por píxel (mm)
- `fps` = Cuadros por segundo
- División por 10 convierte mm/s a cm/s

**Suavizado:**

La velocidad cruda se suaviza usando un filtro de media móvil:

```
                    sum(speed[i-w:i+w])
smoothed_speed = -------------------------
                       2*w + 1
```

Donde `w` es el tamaño de media ventana (por defecto: 5 cuadros para ventana=10).

**Valores Típicos:**

| Condición | Rango de Velocidad | Notas |
|-----------|-------------------|-------|
| Estacionario | 0-2 cm/s | Acicalamiento, descanso |
| Caminando | 5-15 cm/s | Locomoción normal |
| Caminata rápida | 15-30 cm/s | Movimiento motivado |
| Corriendo | 30-60 cm/s | Comportamiento de escape |

---

### Velocidad Promedio

**Definición:** Media de la velocidad instantánea durante el período de grabación.

**Fórmula:**

```
                    1   n
mean_speed = --- * SUM speed[i]
                    n  i=1
```

**Casos de Uso:**
- Comparar niveles de actividad general entre grupos
- Evaluar efectos de intervenciones en la locomoción
- Medición base para normalización

---

### Aceleración

**Definición:** Tasa de cambio de velocidad en el tiempo.

**Fórmula:**

```
acceleration[i] (cm/s^2) = (speed[i+1] - speed[i]) * fps
```

**Interpretación:**
- Aceleración positiva = acelerando (recuperación)
- Aceleración negativa = desacelerando (arrastre)

---

### Eventos de Arrastre y Recuperación

**Definiciones:**
- **Evento de Arrastre:** Período sostenido de aceleración negativa (desacelerando)
- **Evento de Recuperación:** Período sostenido de aceleración positiva (acelerando)

**Algoritmo de Detección:**

```
1. Calcular aceleración suavizada
2. Identificar segmentos contiguos donde:
   - Arrastre: aceleración < 0 por duración >= umbral
   - Recuperación: aceleración > 0 por duración >= umbral
3. Filtrar por duración mínima (por defecto: 0.25 segundos)
```

**Métricas:**

| Métrica | Fórmula | Unidad |
|---------|---------|--------|
| Conteo de Arrastre | Número de eventos de arrastre | conteo |
| Conteo de Recuperación | Número de eventos de recuperación | conteo |
| Duración de Arrastre | Suma de duraciones de eventos de arrastre | segundos |
| Duración de Recuperación | Suma de duraciones de eventos de recuperación | segundos |
| Ratio Arrastre/Recuperación | Duración Arrastre / Duración Recuperación | ratio |

**Relevancia Clínica:**
- Eventos de arrastre aumentados pueden indicar deterioro motor
- El ratio arrastre/recuperación puede reflejar fatiga motora
- Útil para evaluar recuperación de lesión medular

---

## Métricas de Coordinación

### Contexto de Estadística Circular

La coordinación de extremidades se mide usando **estadística circular** porque las relaciones de fase son cíclicas (0 = 360 grados).

```
Representación en Espacio de Fase:

           90 grados
              |
              |
    180 grados --+-- 0 grados
              |
              |
           270 grados
```

---

### Longitud del Vector Resultante (R)

**Definición:** Medida de concentración de ángulos de fase. Indica fuerza de coordinación.

**Fórmula:**

```
X = mean(cos(phi))
Y = mean(sin(phi))
R = sqrt(X^2 + Y^2)
```

Donde `phi` es el array de ángulos de fase en radianes.

**Interpretación Geométrica:**

```
Cada ángulo de fase es un vector unitario en el círculo unitario.
R es la longitud del vector medio.

R alto (vectores alineados):      R bajo (vectores dispersos):

    \   |   /                       \  /
     \  |  /                     ----+----
      \ | /                       /  |  \
       \|/                       /   |   \
    ----+----> R ~ 1            R ~ 0
        |
```

**Rango:** [0, 1]
- R = 1: Todas las fases idénticas (coordinación perfecta)
- R = 0: Fases uniformemente distribuidas (sin coordinación)

**Tabla de Interpretación:**

| Valor R | Nivel de Coordinación | Descripción |
|---------|----------------------|-------------|
| 0.9 - 1.0 | Excelente | Relación de fase altamente consistente |
| 0.7 - 0.9 | Buena | Coordinación consistente |
| 0.5 - 0.7 | Moderada | Cierta variabilidad |
| 0.3 - 0.5 | Débil | Alta variabilidad de fase |
| 0.0 - 0.3 | Ninguna | Movimiento aleatorio/independiente |

---

### Ángulo de Fase Medio

**Definición:** Dirección promedio de la relación de fase entre par de extremidades.

**Fórmula:**

```
mean_phi = atan2(Y, X)
```

Donde X e Y se calculan como arriba.

**Rango del Resultado:** [-180, 180] grados (o [-pi, pi] radianes)

**Interpretación del Ángulo de Fase:**

| Fase | Grados | Patrón | Descripción |
|------|--------|--------|-------------|
| En fase | -30 a 30 | Sincronizado | Las extremidades se mueven juntas |
| Anti-fase | 150 a 210 | Alternando | Las extremidades alternan |
| Cuarto de retraso | 60 a 120 | Adelantado | Una extremidad adelanta 1/4 de ciclo |
| Tres cuartos | 240 a 300 | Retrasado | Una extremidad retrasa 1/4 de ciclo |

---

### Estimación de Fase Por Ciclo

**Definición:** Relación de fase estimada para cada ciclo de marcha usando un método basado en integral.

**Fórmula:**

Para cada ciclo delimitado por picos en índices `[i, j]`:

```
1. Normalizar zancada relativa a [-1, 1]:
   y_norm = 2 * (rel_stride - min) / (max - min) - 1

2. Crear eje de fase:
   x = linspace(0, 2*pi, j-i)

3. Calcular fase desde integral:
   phi = (4 - integral(y_norm, x)) * pi / 4
```

**Justificación:**
- La integral de una sinusoide normalizada sobre un período indica el desfase
- Valores alrededor de 4 indican en fase; valores alrededor de 0 indican anti-fase

---

### Definiciones de Pares de Extremidades

Pares estándar de extremidades para locomoción cuadrúpeda:

```
       FRENTE
    foreL    foreR
       |        |
       |        |
    hindL    hindR
       ATRÁS

Abreviaciones de Pares:
- LH = Trasera Izquierda (hindL)
- RH = Trasera Derecha (hindR)
- LF = Delantera Izquierda (foreL)
- RF = Delantera Derecha (foreR)
```

**Pares Estándar:**

| Nombre del Par | Extremidades | Fase Típica | Tipo de Marcha |
|----------------|--------------|-------------|----------------|
| LH_RH | Traseras Izq-Der | ~180 grados | Todas las marchas |
| LF_RF | Delanteras Izq-Der | ~180 grados | Todas las marchas |
| LH_LF | Trasera-Delantera Izq | ~0 o ~180 | Paso o Trote |
| RH_RF | Trasera-Delantera Der | ~0 o ~180 | Paso o Trote |
| LF_RH | Diagonal | ~0 grados | Trote |
| RF_LH | Diagonal | ~0 grados | Trote |

---

### Clasificación de Patrones de Marcha

Diferentes patrones de coordinación indican diferentes tipos de marcha:

```
TROTE (más común en ratones):
+---------------------------+
| Pares diagonales en sync  |
| Ipsilaterales alternando  |
+---------------------------+
  LF_RH: R > 0.8, fase ~ 0
  RF_LH: R > 0.8, fase ~ 0
  LH_RH: R > 0.8, fase ~ 180
  LF_RF: R > 0.8, fase ~ 180


PASO (menos común):
+---------------------------+
| Pares ipsilaterales sync  |
| Contralaterales alternando|
+---------------------------+
  LH_LF: R > 0.8, fase ~ 0
  RH_RF: R > 0.8, fase ~ 0
  LH_RH: R > 0.8, fase ~ 180


SALTO/GALOPE:
+---------------------------+
| Par delantero sincronizado|
| Par trasero sincronizado  |
| Delante-atrás alternando  |
+---------------------------+
  LH_RH: R > 0.8, fase ~ 0
  LF_RF: R > 0.8, fase ~ 0
  LH_LF: R > 0.8, fase ~ 180
```

---

## Métricas de Ciclo de Marcha

### Detección de Ciclos

**Método:** Detección de picos en datos de posición de zancada.

**Algoritmo:**

```
1. Aplicar suavizado a datos de zancada
2. Encontrar todos los máximos locales (picos)
3. Calcular intervalo medio entre picos
4. Re-detectar picos con restricción de distancia mínima (mitad del intervalo medio)
5. Opcionalmente detectar valles (mínimos) para fases de apoyo/balanceo
```

**Definición de Ciclo de Marcha:**

```
                  POSICIÓN DE ZANCADA
                        ^
                        |
    Pico (max extensión)|      Pico
              \         |     /
               \  Balanceo    /
                \       |   /
                 \______|__/
                   |    |
                Fase de apoyo
                   |    |
             Valle (max flexión)

    |<---- Un Ciclo de Marcha ---->|
```

---

### Cadencia

**Definición:** Número de pasos (ciclos) por unidad de tiempo. También llamada frecuencia de paso.

**Fórmula:**

```
cadencia (Hz) = número_de_ciclos / duración (segundos)
```

**Valores Típicos para Ratones:**

| Categoría de Velocidad | Cadencia | Notas |
|------------------------|----------|-------|
| Caminata lenta | 2-3 Hz | Exploratorio |
| Caminata normal | 3-5 Hz | Típica |
| Caminata rápida | 5-7 Hz | Motivada |
| Carrera | 7-10 Hz | Escape |

**Relación con la Velocidad:**

```
velocidad = cadencia * longitud_zancada

Para longitud de zancada constante:
  Mayor cadencia = Mayor velocidad

Para cadencia constante:
  Zancada más larga = Mayor velocidad
```

---

### Longitud de Zancada

**Definición:** Distancia recorrida durante un ciclo de marcha completo.

**Fórmula:**

```
longitud_zancada (cm) = velocidad_promedio (cm/s) / cadencia (Hz)
```

**Componentes:**
- **Longitud de Paso:** Distancia entre mismo punto en extremidades opuestas
- **Longitud de Zancada:** Distancia entre mismo punto en misma extremidad (2x paso para alternancia)

**Valores Típicos:**

| Tamaño/Edad | Longitud de Zancada | Notas |
|-------------|---------------------|-------|
| Ratón joven | 3-5 cm | Mayor relativo al cuerpo |
| Ratón adulto | 4-6 cm | Rango normal |
| Ratón envejecido | 3-4 cm | Puede disminuir |

---

### Métricas de Regularidad de Marcha

**Duración de Ciclo:**

```
cycle_duration[i] = (peak[i+1] - peak[i]) / fps  (segundos)
```

**Coeficiente de Variación (CV):**

```
CV = std(cycle_duration) / mean(cycle_duration)
```

**Interpretación:**
- CV < 0.1: Marcha muy regular
- CV 0.1-0.2: Variación normal
- CV > 0.2: Marcha irregular (puede indicar patología)

**Amplitud de Zancada:**

```
amplitude[i] = |stride[peak[i]] - stride[trough[i]]|
```

Mide el rango de movimiento de la extremidad durante cada ciclo.

---

### Eventos de Marcha

**Inicio de Apoyo (Aterrizaje):**
- Momento cuando la pata contacta el suelo
- Detectado en cruce por cero (zancada yendo hacia atrás)

**Inicio de Balanceo (Despegue):**
- Momento cuando la pata deja el suelo
- Detectado en cruce por cero (zancada yendo hacia adelante)

**Factor de Trabajo:**

```
duty_factor = duración_apoyo / duración_ciclo
```

- < 0.5: Marcha de carrera (fase aérea)
- = 0.5: Transición
- > 0.5: Marcha de caminata (fase de superposición)

---

## Interpretación Biológica

### Patrones Normales vs. Patológicos

**Locomoción Saludable de Ratón:**

| Métrica | Rango Normal | Notas |
|---------|--------------|-------|
| Velocidad Media | 10-25 cm/s | Dependiente del contexto |
| Coordinación R | > 0.7 | Todos los pares |
| Fase (diagonal) | ~0 grados | Patrón de trote |
| Fase (ipsilateral) | ~180 grados | Alternando |
| Cadencia | 3-6 Hz | Dependiente de velocidad |
| CV (regularidad) | < 0.15 | Marcha estable |

**Indicadores de Deterioro:**

| Hallazgo | Posible Interpretación |
|----------|------------------------|
| R bajo (< 0.5) | Pérdida de coordinación |
| Fase irregular | Ataxia |
| CV alto | Marcha inestable |
| Velocidad reducida | Debilidad, dolor |
| Eventos de arrastre aumentados | Fatiga, debilidad |
| Cadencia asimétrica | Deterioro lateralizado |

---

### Aplicaciones Clínicas

**Evaluación de Lesión Medular:**

```
Pre-lesión:      Post-lesión (leve):    Post-lesión (severa):
R = 0.92         R = 0.65               R = 0.30
CV = 0.08        CV = 0.18              CV = 0.35
Fase: estable    Fase: variable         Fase: aleatoria
```

**Evaluación de Efectos de Fármacos:**

Monitorear cambios en:
- Velocidad (sedación/estimulación)
- Coordinación (efectos motores)
- Cadencia (efectos dopaminérgicos)
- Regularidad (efectos cerebelosos)

**Cambios Relacionados con la Edad:**

| Métrica | Joven | Envejecido |
|---------|-------|------------|
| Velocidad | Mayor | Menor |
| Longitud de Zancada | Más larga | Más corta |
| Coordinación R | Mayor | Puede disminuir |
| Regularidad CV | Menor | Puede aumentar |

---

### Consideraciones Estadísticas

**Tamaño de Muestra:**
- Mínimo 10-15 zancadas por par de extremidades para R confiable
- Se recomiendan 30+ segundos de locomoción

**Tamaño de Efecto para R:**

```
Efecto pequeño:  delta_R = 0.1
Efecto mediano:  delta_R = 0.2
Efecto grande:   delta_R = 0.3
```

**Test de Rayleigh para Significancia:**

Probar si R es significativamente diferente de 0:

```
Z = n * R^2
p-valor de distribución chi-cuadrado con df=2
```

Significativo si p < 0.05 indica distribución no uniforme.

---

## Resumen de Fórmulas

### Velocidad

```
speed[i] = sqrt(dx^2 + dy^2) * scale * fps
acceleration = diff(speed) * fps
```

### Estadística Circular

```
R = sqrt(mean(cos(phi))^2 + mean(sin(phi))^2)
mean_angle = atan2(mean(sin(phi)), mean(cos(phi)))
```

### Métricas de Marcha

```
cadence = n_cycles / duration
stride_length = speed / cadence
CV = std(cycle_duration) / mean(cycle_duration)
duty_factor = stance_time / cycle_time
```

---

## Referencias

1. Batka, R. J., et al. (2014). "The need for speed in rodent locomotion analyses." The Anatomical Record.

2. Hamers, F. P., et al. (2006). "CatWalk-assisted gait analysis in the assessment of spinal cord injury." Journal of Neurotrauma.

3. Koopmans, G. C., et al. (2005). "The assessment of locomotor function in spinal cord injured rats: the importance of objective analysis of coordination." Journal of Neurotrauma.

4. Mardia, K. V., & Jupp, P. E. (2000). "Directional Statistics." Wiley.

5. Allodi, I., et al. (2021). "Locomotion analysis in mice." Nature Protocols (metodología adaptada).
