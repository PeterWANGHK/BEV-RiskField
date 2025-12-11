# Corrected exiD SVO Visualization: Issues Analysis and Fixes

## Executive Summary

This document details the issues identified in the original `exid_enhanced_svo_visualization.py` and the corrections implemented. The main problems were:

1. **No change in mutual SVO** - Static SVO values over time
2. **Abrupt disappearance of surrounding cars** - Vehicles vanishing suddenly
3. **Missing bidirectional SVO plots** - No per-pair interaction tracking
4. **Missing SVO evolution plots** - No analysis vs distance/position
5. **Missing SVO heatmap matrix** - No interaction matrix visualization

---

## Issue 1: No Change in Mutual SVO

### Root Cause Analysis

**Problem A: Over-aggressive normalization**

The `potential` was calculated using fixed reference values (`V_REF=30`, `DIST_REF=30`) that were too high, causing most aggressiveness values to normalize to nearly the same value.

**Problem B: Static weighting**

The weights didn't account for relative motion between vehicles, which is critical for dynamic SVO changes.

**Problem C: Excessive smoothing**

A 15-frame smoothing window with max delta of 2.5 degrees severely dampened any SVO changes.

### Fixes Implemented

1. **Context-dependent normalization**: Normalization factor now adapts to current speeds and distances
2. **Added relative motion component**: New weight (0.15) for approach/recede behavior
3. **Reduced smoothing**: Window reduced to 11 frames, max delta increased to 3.5 degrees

---

## Issue 2: Abrupt Disappearance of Surrounding Cars

### Root Cause Analysis

**Problem A: Short hysteresis window** - Only 10 frames (0.4 seconds)

**Problem B: Limited interpolation** - Only 3 frames interpolation

**Problem C: No confidence tracking** - Sudden on/off behavior

### Fixes Implemented

1. **Extended hysteresis**: 25 frames (~1 second)
2. **Velocity-based prediction**: Uses velocity history for stable extrapolation
3. **Confidence scoring**: Tracks reliability of each vehicle's position

---

## Issue 3: Missing Bidirectional SVO Plots

### Problem

The original only showed aggregate SVO, not per-pair interactions.

### Solution

New `PairwiseSVO` dataclass tracks:
- Ego-to-other SVO
- Other-to-ego SVO
- Aggressiveness in both directions
- Relative state (distance, delta_x, delta_y)
- Merge phase

New plot: `bidirectional_svo_{ego_id}.png` showing mutual SVO for top 4 interacting pairs.

---

## Issue 4: Missing SVO Evolution Plots

### Problem

No visualization of how SVO changes with spatial variables.

### Solution

New `svo_evolution_{ego_id}.png` plot showing:
- SVO vs Distance
- SVO vs Delta-X (longitudinal gap)
- SVO vs Delta-Y (lateral gap)
- SVO vs Relative Velocity
- SVO by Merge Phase (bar chart)

---

## Issue 5: Missing SVO Heatmap Matrix

### Problem

No overview of all truck-car interactions.

### Solution

New `svo_heatmap_{ego_id}.png` showing:
- Truck to Cars (mean SVO)
- Cars to Truck (mean SVO)
- Asymmetry (difference)

---

## Key Parameter Changes

| Parameter | Original | Corrected | Reason |
|-----------|----------|-----------|--------|
| TRACKING_HYSTERESIS | 10 | 25 | Prevent abrupt disappearances |
| INTERPOLATION_WINDOW | 3 | 8 | Smoother interpolation |
| INTERPOLATION_DECAY | 0.85 | 0.92 | More gradual confidence decay |
| V_REF | 30.0 | 25.0 | More realistic reference |
| DIST_REF | 30.0 | 25.0 | More sensitive to distance |
| SMOOTH_WINDOW | 15 | 11 | Faster response |
| MAX_SVO_DELTA | 2.5 | 3.5 | Allow more dynamic changes |
| EMA_ALPHA | - | 0.2 | New EMA smoothing |
| WEIGHT_RELATIVE_MOTION | - | 0.15 | New motion component |

---

## Usage

```bash
python exid_corrected_svo_visualization.py \
    --data_dir /path/to/exid/data \
    --recording 25 \
    --output_dir ./output_corrected
```

## Output Files

1. `bidirectional_svo_{id}.png` - Mutual SVO for top pairs
2. `svo_evolution_{id}.png` - SVO vs spatial variables
3. `svo_heatmap_{id}.png` - Interaction matrix
4. `summary_analysis_{id}.png` - Overview with tracking stats

---

## Mathematical Logic Review

### SVO Angle Formula (Corrected)

The SVO angle is computed from four components:

1. **Aggressiveness Component** (45%):
   - θ_aggr = 90° - 135° × normalized_aggr
   - Range: -45° (aggressive) to 90° (passive)

2. **Deceleration Component** (30%):
   - θ_decel = 45° × tanh(decel/2)
   - Range: 0° to ~45°

3. **Yielding Component** (25%):
   - θ_yield = -22.5° + 67.5° × (1 - speed_ratio)
   - Range: -22.5° to 45°

4. **Relative Motion Component** (15% - NEW):
   - θ_approach = -20° × normalized_approach_rate
   - Range: -20° to 20°

Final: θ_SVO = phase_modifier × (w1×θ_aggr + w2×θ_decel + w3×θ_yield + w4×θ_approach)

### Interpretation

- θ > 60°: Altruistic
- 30° < θ < 60°: Cooperative
- 0° < θ < 30°: Individualistic
- θ < 0°: Competitive/Aggressive
