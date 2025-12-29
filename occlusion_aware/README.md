# Occlusion-Aware Interaction Field Modeling

## 1. Conceptual Framework

### 1.1 The Problem

In highway merging scenarios with trucks, **occlusion creates epistemic uncertainty**:

```
                    OCCLUDER (Truck T)
                         🚛
                        /  \
                       /    \  ← SHADOW ZONE
                      /      \
    EGO (Observer)            OCCLUDED VEHICLE (O)
         🚗 ─────────────X────────→ 🚙
              CAN'T SEE!           IS HIDDEN
```

![The example occlusion scenarios in highway merging scenarios](occlusion_merging.png)

**Standard GVF/APF Problem**: Traditional fields treat all vehicles equally, assuming perfect observation. But the ego vehicle **cannot accurately assess** the state of occluded vehicles.

### 1.2 Key Insight

> **The field contribution from an occluded vehicle should reflect our UNCERTAINTY about it, not just its physical presence.**

This means:
- **Fully visible vehicle** → Full field weight (w = 1.0)
- **Partially occluded** → Reduced weight (0.4 < w ≤ 0.8)
- **Mostly occluded** → Minimal weight (w ≥ 0.1, never zero for safety)

---

## 2. Mathematical Formulation

### 2.1 Standard GVF (Baseline)

The standard Gaussian Velocity Field is:

$$\Phi(x) = \sum_{i=1}^{N} K(x, x_i) \cdot v_i$$

where:
- $K(x, x_i)$ is an RBF kernel centered at vehicle $i$'s position
- $v_i$ is vehicle $i$'s relative velocity
- All vehicles contribute equally

### 2.2 Occlusion-Aware GVF (Proposed)

We introduce a **visibility weight** $w_i$ for each vehicle:

$$\Phi_{occ}(x) = \sum_{i=1}^{N} w_i(t) \cdot K(x, x_i) \cdot v_i$$

where $w_i(t) \in [w_{min}, 1.0]$ depends on:

1. **Geometric occlusion ratio** $\rho_i$
2. **Observer's acceleration** $a_{obs}$
3. **Temporal history** (smoothing)

### 2.3 Visibility Weight Computation

```
w_i(t) = max(w_min, smooth(w_base × w_accel))

where:
  w_base = 1 - ρ_i                    (geometric factor)
  w_accel = 1 + α × a_obs             (dynamic factor)
  smooth(·) = exponential moving avg  (temporal factor)
```

| Factor | Symbol | Description |
|--------|--------|-------------|
| Occlusion ratio | ρ | Fraction of vehicle angularly blocked by occluder |
| Base visibility | w_base | 1 - ρ (purely geometric) |
| Acceleration sensitivity | α | How much ego acceleration affects visibility |
| Minimum visibility | w_min | Safety floor (never completely ignore a vehicle) |

---

## 3. Dynamic Visibility Model

### 3.1 Why Dynamic?

Visibility isn't static—it changes as vehicles move:

```
Time t:   Ego 🚗 ──────🚛 Truck────────🚙 Occluded (w = 0.3)
                   ↓ Ego accelerates ↓
Time t+Δt: Ego 🚗────🚛 Truck──────🚙 Occluded (w = 0.5)
                   ↓ Ego passes truck ↓  
Time t+2Δt:     🚛 Truck──🚗 Ego──────🚙 Now Visible (w = 1.0)
```

### 3.2 Acceleration Effect

When the **observer (ego) accelerates**:
- They may **gain visibility** by passing the occluder
- The field weight of previously occluded vehicles **increases**

When the **observer decelerates**:
- They may **lose visibility** by falling behind
- The field weight of vehicles ahead **decreases**

**Mathematical form**:
$$w_{accel} = 1 + \alpha \cdot a_{longitudinal}$$

where $\alpha \approx 0.1$ (tunable parameter)

### 3.3 Temporal Smoothing

To prevent jitter in visibility weights:

$$w_{smooth}(t) = \beta \cdot w_{smooth}(t-1) + (1-\beta) \cdot w_{raw}(t)$$

with $\beta \approx 0.7$ (smoothing factor)

---

## 4. Implementation Details

### 4.1 Occlusion Ratio Computation

```python
def compute_occlusion_ratio(observer, target, occluder):
    # Angular extent of target from observer's view
    target_angle = arctan2(target.y - observer.y, target.x - observer.x)
    target_half_width = arctan2(target.width / 2, distance_to_target)
    target_range = (target_angle - target_half_width, 
                    target_angle + target_half_width)
    
    # Shadow cast by occluder
    occluder_angle = arctan2(occluder.y - observer.y, occluder.x - observer.x)
    shadow_half_width = arctan2(effective_occluder_width / 2, distance_to_occluder)
    shadow_range = (occluder_angle - shadow_half_width,
                    occluder_angle + shadow_half_width)
    
    # Overlap
    overlap = compute_angular_overlap(target_range, shadow_range)
    occlusion_ratio = overlap / (target_range[1] - target_range[0])
    
    return occlusion_ratio
```

### 4.2 Visibility Weight Computation

```python
def compute_visibility_weight(observer, target, occluder, 
                               observer_accel, history):
    # Base geometric visibility
    occlusion_ratio = compute_occlusion_ratio(observer, target, occluder)
    w_base = 1.0 - occlusion_ratio
    
    # Dynamic adjustment
    alpha = 0.1  # acceleration sensitivity
    w_accel = 1.0 + alpha * observer_accel
    w_accel = clip(w_accel, 0.8, 1.2)
    
    # Combine
    w_dynamic = w_base * w_accel
    
    # Temporal smoothing
    beta = 0.7
    w_smooth = beta * history[-1] + (1 - beta) * w_dynamic
    
    # Safety floor
    w_min = 0.1
    w_final = max(w_min, min(1.0, w_smooth))
    
    return w_final
```

---

## 5. Field Interpretation

### 5.1 What Reduced Weight Means

| Weight | Interpretation | Field Effect |
|--------|---------------|--------------|
| w = 1.0 | Full visibility | Full contribution to interaction field |
| w = 0.5 | Partial occlusion | Reduced but significant influence |
| w = 0.1 | Heavy occlusion | Minimal influence (uncertainty) |

### 5.2 Safety Consideration

We **never set w = 0** because:
- A hidden vehicle still poses collision risk
- The minimum weight represents "we know something is there but uncertain"
- Planning layers should treat low-visibility vehicles with extra caution


## 6. Visualization Guide

The generated comparison figure shows:

| Panel | Content |
|-------|---------|
| **Standard GVF** | Baseline field (all weights = 1.0) |
| **Occlusion-Aware GVF** | Modified field with visibility weights |
| **Visibility Legend** | Per-vehicle weights and status |
| **Occlusion Geometry** | Shadow zones and tangent lines |
| **Field Difference** | Where the two fields differ (red = overestimated) |
| **Dynamic Model** | Explanation of acceleration effects |

### Vehicle Styling

| Element | Meaning |
|---------|---------|
| Solid border | Fully visible |
| Dashed border | Occluded (reduced weight) |
| Red dashed lines | Shadow/tangent boundaries |
| Faded color | Low visibility weight |

---

## 7. Extension to Unified Field Model

### 7.1 Integration with Role Classification

Combine visibility weights with role-based risk:

$$\Phi_{unified}(x) = \sum_{i} w_i^{vis}(t) \cdot w_i^{role} \cdot w_i^{mass} \cdot K(x, x_i) \cdot v_i$$

where:
- $w_i^{vis}$ = visibility weight (occlusion)
- $w_i^{role}$ = role urgency (merging vehicles have higher weight)
- $w_i^{mass}$ = mass/size factor (trucks have larger influence)

### 7.2 Bidirectional Occlusion (MERGE_TWO_WAY)

When both vehicles can't see each other:

```
Main lane:   🚗 Car A (can't see B)
             🚛 Truck (mutual occluder)
Merge lane:  🚗 Car B (can't see A)
```

Both A and B have reduced fields **from each other's perspective**:
- A's field contribution to B is reduced
- B's field contribution to A is reduced
- This creates a "mutual uncertainty zone"

## 8. Comparison Metrics

To prove occlusion-awareness improves realism:

| Metric | How to Measure |
|--------|----------------|
| **Prediction ADE in occlusion** | Compare ADE for vehicles that were occluded |
| **False positive rate** | Does standard GVF overestimate risk in shadows? |
| **Uncertainty calibration** | Does low visibility correlate with prediction variance? |
| **Gap acceptance accuracy** | Do merging decisions match when occlusion considered? |

---

## 9. Code Usage

```python
from occlusion_aware_gvf import (
    OcclusionAwareGVFVisualizer,
    OcclusionGVFConfig,
    construct_occlusion_aware_gvf
)

# Create visualizer
config = OcclusionGVFConfig()
viz = OcclusionAwareGVFVisualizer(config)

# Generate comparison
visibility_weights, visibility_info = viz.create_comparison_figure(
    ego=ego_vehicle,
    vehicles=surrounding_vehicles,
    occlusion_events=detected_occlusions,
    output_path='occlusion_comparison.png'
)

# Use weights in your field
X, Y, VX, VY, W = construct_occlusion_aware_gvf(
    ego, vehicles, visibility_weights, x_range, y_range
)
```

---
