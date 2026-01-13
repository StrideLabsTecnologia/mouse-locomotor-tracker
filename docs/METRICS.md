# Metrics Guide

Detailed explanation of all metrics computed by Mouse Locomotor Tracker, including mathematical formulas and biological interpretation.

## Table of Contents

1. [Velocity Metrics](#velocity-metrics)
2. [Coordination Metrics](#coordination-metrics)
3. [Gait Cycle Metrics](#gait-cycle-metrics)
4. [Biological Interpretation](#biological-interpretation)

---

## Velocity Metrics

### Instantaneous Speed

**Definition:** The rate of position change between consecutive frames.

**Formula:**

```
                    sqrt((x[i+1] - x[i])^2 + (y[i+1] - y[i])^2) * pixel_to_mm
speed[i] (cm/s) = ------------------------------------------------------------ * fps / 10
                                            1
```

Where:
- `x[i], y[i]` = Position coordinates at frame `i` (pixels)
- `pixel_to_mm` = Physical size per pixel (mm)
- `fps` = Frames per second
- Division by 10 converts mm/s to cm/s

**Smoothing:**

The raw speed is smoothed using a moving average filter:

```
                    sum(speed[i-w:i+w])
smoothed_speed = -------------------------
                       2*w + 1
```

Where `w` is the half-window size (default: 5 frames for window=10).

**Typical Values:**

| Condition | Speed Range | Notes |
|-----------|-------------|-------|
| Stationary | 0-2 cm/s | Grooming, resting |
| Walking | 5-15 cm/s | Normal locomotion |
| Fast walking | 15-30 cm/s | Motivated movement |
| Running | 30-60 cm/s | Escape behavior |

---

### Average Speed

**Definition:** Mean instantaneous speed over the recording period.

**Formula:**

```
                    1   n
mean_speed = --- * SUM speed[i]
                    n  i=1
```

**Use Cases:**
- Compare overall activity levels between groups
- Assess effects of interventions on locomotion
- Baseline measurement for normalization

---

### Acceleration

**Definition:** Rate of change of speed over time.

**Formula:**

```
acceleration[i] (cm/s^2) = (speed[i+1] - speed[i]) * fps
```

**Interpretation:**
- Positive acceleration = speeding up (recovery)
- Negative acceleration = slowing down (drag)

---

### Drag and Recovery Events

**Definitions:**
- **Drag Event:** Sustained period of negative acceleration (slowing)
- **Recovery Event:** Sustained period of positive acceleration (speeding up)

**Detection Algorithm:**

```
1. Compute smoothed acceleration
2. Identify contiguous segments where:
   - Drag: acceleration < 0 for duration >= threshold
   - Recovery: acceleration > 0 for duration >= threshold
3. Filter by minimum duration (default: 0.25 seconds)
```

**Metrics:**

| Metric | Formula | Unit |
|--------|---------|------|
| Drag Count | Number of drag events | count |
| Recovery Count | Number of recovery events | count |
| Drag Duration | Sum of all drag event durations | seconds |
| Recovery Duration | Sum of all recovery event durations | seconds |
| Drag/Recovery Ratio | Drag Duration / Recovery Duration | ratio |

**Clinical Relevance:**
- Increased drag events may indicate motor impairment
- Drag/recovery ratio can reflect motor fatigue
- Useful for assessing spinal cord injury recovery

---

## Coordination Metrics

### Circular Statistics Background

Limb coordination is measured using **circular statistics** because phase relationships are cyclic (0 = 360 degrees).

```
Phase Space Representation:

           90 deg
              |
              |
    180 deg --+-- 0 deg
              |
              |
           270 deg
```

---

### Resultant Vector Length (R)

**Definition:** Measure of concentration of phase angles. Indicates coordination strength.

**Formula:**

```
X = mean(cos(phi))
Y = mean(sin(phi))
R = sqrt(X^2 + Y^2)
```

Where `phi` is the array of phase angles in radians.

**Geometric Interpretation:**

```
Each phase angle is a unit vector on the unit circle.
R is the length of the mean vector.

High R (vectors aligned):      Low R (vectors scattered):

    \   |   /                       \  /
     \  |  /                     ----+----
      \ | /                       /  |  \
       \|/                       /   |   \
    ----+----> R ~ 1            R ~ 0
        |
```

**Range:** [0, 1]
- R = 1: All phases identical (perfect coordination)
- R = 0: Phases uniformly distributed (no coordination)

**Interpretation Table:**

| R Value | Coordination Level | Description |
|---------|-------------------|-------------|
| 0.9 - 1.0 | Excellent | Highly consistent phase relationship |
| 0.7 - 0.9 | Good | Consistent coordination |
| 0.5 - 0.7 | Moderate | Some variability |
| 0.3 - 0.5 | Weak | High phase variability |
| 0.0 - 0.3 | None | Random/independent movement |

---

### Mean Phase Angle

**Definition:** Average direction of phase relationship between limb pair.

**Formula:**

```
mean_phi = atan2(Y, X)
```

Where X and Y are computed as above.

**Result Range:** [-180, 180] degrees (or [-pi, pi] radians)

**Phase Angle Interpretation:**

| Phase | Degrees | Pattern | Description |
|-------|---------|---------|-------------|
| In-phase | -30 to 30 | Synchronized | Limbs move together |
| Anti-phase | 150 to 210 | Alternating | Limbs alternate |
| Quarter lag | 60 to 120 | Leading | One limb leads by 1/4 cycle |
| Three-quarter | 240 to 300 | Lagging | One limb lags by 1/4 cycle |

---

### Phase Estimation Per Cycle

**Definition:** Phase relationship estimated for each gait cycle using an integral-based method.

**Formula:**

For each cycle bounded by peaks at indices `[i, j]`:

```
1. Normalize relative stride to [-1, 1]:
   y_norm = 2 * (rel_stride - min) / (max - min) - 1

2. Create phase axis:
   x = linspace(0, 2*pi, j-i)

3. Compute phase from integral:
   phi = (4 - integral(y_norm, x)) * pi / 4
```

**Rationale:**
- The integral of a normalized sinusoid over one period indicates phase offset
- Values around 4 indicate in-phase; values around 0 indicate anti-phase

---

### Limb Pair Definitions

Standard limb pairs for quadrupedal locomotion:

```
       FRONT
    foreL    foreR
       |        |
       |        |
    hindL    hindR
       BACK

Pair Abbreviations:
- LH = Left Hind (hindL)
- RH = Right Hind (hindR)
- LF = Left Fore (foreL)
- RF = Right Fore (foreR)
```

**Standard Pairs:**

| Pair Name | Limbs | Typical Phase | Gait Type |
|-----------|-------|---------------|-----------|
| LH_RH | Left-Right Hind | ~180 deg | All gaits |
| LF_RF | Left-Right Fore | ~180 deg | All gaits |
| LH_LF | Left Hind-Fore | ~0 or ~180 | Pace or Trot |
| RH_RF | Right Hind-Fore | ~0 or ~180 | Pace or Trot |
| LF_RH | Diagonal | ~0 deg | Trot |
| RF_LH | Diagonal | ~0 deg | Trot |

---

### Gait Pattern Classification

Different coordination patterns indicate different gait types:

```
TROT (most common in mice):
+---------------------------+
| Diagonal pairs in sync    |
| Ipsilateral alternating   |
+---------------------------+
  LF_RH: R > 0.8, phase ~ 0
  RF_LH: R > 0.8, phase ~ 0
  LH_RH: R > 0.8, phase ~ 180
  LF_RF: R > 0.8, phase ~ 180


PACE (less common):
+---------------------------+
| Ipsilateral pairs in sync |
| Contralateral alternating |
+---------------------------+
  LH_LF: R > 0.8, phase ~ 0
  RH_RF: R > 0.8, phase ~ 0
  LH_RH: R > 0.8, phase ~ 180


BOUND/GALLOP:
+---------------------------+
| Front pair synchronized   |
| Hind pair synchronized    |
| Front-hind alternating    |
+---------------------------+
  LH_RH: R > 0.8, phase ~ 0
  LF_RF: R > 0.8, phase ~ 0
  LH_LF: R > 0.8, phase ~ 180
```

---

## Gait Cycle Metrics

### Cycle Detection

**Method:** Peak detection on stride position data.

**Algorithm:**

```
1. Apply smoothing to stride data
2. Find all local maxima (peaks)
3. Compute mean inter-peak interval
4. Re-detect peaks with minimum distance constraint (half mean interval)
5. Optionally detect troughs (minima) for stance/swing phases
```

**Gait Cycle Definition:**

```
                  STRIDE POSITION
                        ^
                        |
    Peak (max extension)|      Peak
              \         |     /
               \  Swing |    /
                \       |   /
                 \______|__/
                   |    |
                Stance phase
                   |    |
             Trough (max flexion)

    |<---- One Gait Cycle ---->|
```

---

### Cadence

**Definition:** Number of steps (cycles) per unit time. Also called step frequency.

**Formula:**

```
cadence (Hz) = number_of_cycles / duration (seconds)
```

**Typical Values for Mice:**

| Speed Category | Cadence | Notes |
|----------------|---------|-------|
| Slow walk | 2-3 Hz | Exploratory |
| Normal walk | 3-5 Hz | Typical |
| Fast walk | 5-7 Hz | Motivated |
| Run | 7-10 Hz | Escape |

**Relationship to Speed:**

```
speed = cadence * stride_length

For constant stride length:
  Higher cadence = Higher speed

For constant cadence:
  Longer stride = Higher speed
```

---

### Stride Length

**Definition:** Distance traveled during one complete gait cycle.

**Formula:**

```
stride_length (cm) = average_speed (cm/s) / cadence (Hz)
```

**Components:**
- **Step Length:** Distance between same point on opposite limbs
- **Stride Length:** Distance between same point on same limb (2x step for alternating)

**Typical Values:**

| Size/Age | Stride Length | Notes |
|----------|---------------|-------|
| Young mouse | 3-5 cm | Higher relative to body |
| Adult mouse | 4-6 cm | Normal range |
| Aged mouse | 3-4 cm | May decrease |

---

### Gait Regularity Metrics

**Cycle Duration:**

```
cycle_duration[i] = (peak[i+1] - peak[i]) / fps  (seconds)
```

**Coefficient of Variation (CV):**

```
CV = std(cycle_duration) / mean(cycle_duration)
```

**Interpretation:**
- CV < 0.1: Very regular gait
- CV 0.1-0.2: Normal variation
- CV > 0.2: Irregular gait (may indicate pathology)

**Stride Amplitude:**

```
amplitude[i] = |stride[peak[i]] - stride[trough[i]]|
```

Measures the range of limb movement during each cycle.

---

### Gait Events

**Stance Start (Touchdown):**
- Moment when paw contacts ground
- Detected at zero-crossing (stride going backward)

**Swing Start (Liftoff):**
- Moment when paw leaves ground
- Detected at zero-crossing (stride going forward)

**Duty Factor:**

```
duty_factor = stance_duration / cycle_duration
```

- < 0.5: Running gait (aerial phase)
- = 0.5: Transition
- > 0.5: Walking gait (overlap phase)

---

## Biological Interpretation

### Normal vs. Pathological Patterns

**Healthy Mouse Locomotion:**

| Metric | Normal Range | Notes |
|--------|--------------|-------|
| Mean Speed | 10-25 cm/s | Context dependent |
| Coordination R | > 0.7 | All pairs |
| Phase (diagonal) | ~0 deg | Trot pattern |
| Phase (ipsilateral) | ~180 deg | Alternating |
| Cadence | 3-6 Hz | Speed dependent |
| CV (regularity) | < 0.15 | Stable gait |

**Indicators of Impairment:**

| Finding | Possible Interpretation |
|---------|------------------------|
| Low R (< 0.5) | Loss of coordination |
| Irregular phase | Ataxia |
| High CV | Unstable gait |
| Reduced speed | Weakness, pain |
| Increased drag events | Fatigue, weakness |
| Asymmetric cadence | Lateralized impairment |

---

### Clinical Applications

**Spinal Cord Injury Assessment:**

```
Pre-injury:     Post-injury (mild):    Post-injury (severe):
R = 0.92        R = 0.65               R = 0.30
CV = 0.08       CV = 0.18              CV = 0.35
Phase: stable   Phase: variable        Phase: random
```

**Drug Effect Evaluation:**

Monitor changes in:
- Speed (sedation/stimulation)
- Coordination (motor effects)
- Cadence (dopaminergic effects)
- Regularity (cerebellar effects)

**Age-Related Changes:**

| Metric | Young | Aged |
|--------|-------|------|
| Speed | Higher | Lower |
| Stride Length | Longer | Shorter |
| Coordination R | Higher | May decrease |
| Regularity CV | Lower | May increase |

---

### Statistical Considerations

**Sample Size:**
- Minimum 10-15 strides per limb pair for reliable R
- 30+ seconds of locomotion recommended

**Effect Size for R:**

```
Small effect:   delta_R = 0.1
Medium effect:  delta_R = 0.2
Large effect:   delta_R = 0.3
```

**Rayleigh Test for Significance:**

Test whether R is significantly different from 0:

```
Z = n * R^2
p-value from chi-squared distribution with df=2
```

Significant if p < 0.05 indicates non-uniform distribution.

---

## Formulas Summary

### Velocity

```
speed[i] = sqrt(dx^2 + dy^2) * scale * fps
acceleration = diff(speed) * fps
```

### Circular Statistics

```
R = sqrt(mean(cos(phi))^2 + mean(sin(phi))^2)
mean_angle = atan2(mean(sin(phi)), mean(cos(phi)))
```

### Gait Metrics

```
cadence = n_cycles / duration
stride_length = speed / cadence
CV = std(cycle_duration) / mean(cycle_duration)
duty_factor = stance_time / cycle_time
```

---

## References

1. Batka, R. J., et al. (2014). "The need for speed in rodent locomotion analyses." The Anatomical Record.

2. Hamers, F. P., et al. (2006). "CatWalk-assisted gait analysis in the assessment of spinal cord injury." Journal of Neurotrauma.

3. Koopmans, G. C., et al. (2005). "The assessment of locomotor function in spinal cord injured rats: the importance of objective analysis of coordination." Journal of Neurotrauma.

4. Mardia, K. V., & Jupp, P. E. (2000). "Directional Statistics." Wiley.

5. Allodi, I., et al. (2021). "Locomotion analysis in mice." Nature Protocols (methodology adapted).
