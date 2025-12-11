"""
exiD Dataset: Corrected SVO Visualization with Enhanced Features
================================================================
FIXES:
1. Dynamic SVO calculation with proper sensitivity to state changes
2. Improved hysteresis tracking with longer memory and better interpolation
3. Bidirectional SVO tracking for specific vehicle pairs

NEW FEATURES:
1. Bidirectional SVO plots (mutual interactions between truck-car pairs)
2. SVO evolution plots as function of merge distance and Δx/Δy
3. SVO heatmap matrix between truck and car agents
4. Phase detection (pre-merge, during merge, post-merge)

Based on the logic from intersection_visualizer.py for frame/track handling.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.gridspec import GridSpec
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Set, NamedTuple
from collections import defaultdict, deque
import warnings
import argparse
import logging
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


# =============================================================================
# Configuration - CORRECTED PARAMETERS
# =============================================================================

@dataclass
class Config:
    """Configuration with corrected parameters for dynamic SVO."""
    
    HEAVY_VEHICLE_CLASSES: Set[str] = field(default_factory=lambda: {'truck', 'bus', 'van', 'trailer'})
    CAR_CLASSES: Set[str] = field(default_factory=lambda: {'car'})
    
    # Aggressiveness parameters - RECALIBRATED for sensitivity
    MU_1: float = 0.25  # Increased from 0.2 for more sensitivity to ego velocity
    MU_2: float = 0.25  # Increased from 0.21 for more sensitivity to other velocity
    SIGMA: float = 0.08  # Decreased from 0.1 for faster distance decay
    DELTA: float = 0.001  # Increased from 0.0005 for better scaling
    TAU_1: float = 0.25  # Slightly increased for longer longitudinal ellipse
    TAU_2: float = 0.12  # Slightly increased for wider lateral ellipse
    BETA: float = 0.04   # Slightly decreased velocity expansion factor
    
    # Vehicle masses
    MASS_HV: float = 15000.0
    MASS_PC: float = 3000.0
    
    # Reference values - RECALIBRATED
    V_REF: float = 25.0  # Lowered from 30 for more realistic highway reference
    DIST_REF: float = 25.0  # Lowered for more sensitive distance effects
    
    # SVO behavioral weights - REBALANCED
    WEIGHT_AGGR_RATIO: float = 0.45  # Increased aggressiveness weight
    WEIGHT_DECEL: float = 0.30
    WEIGHT_YIELDING: float = 0.25
    WEIGHT_RELATIVE_MOTION: float = 0.15  # NEW: weight for relative motion
    
    # === IMPROVED HYSTERESIS TRACKING ===
    TRACKING_HYSTERESIS: int = 25  # Increased from 10 to 25 frames (~1 second)
    MAX_TRACKING_DISTANCE: float = 120.0  # Increased from 100
    INTERPOLATION_WINDOW: int = 8  # Increased from 3 for smoother interpolation
    INTERPOLATION_DECAY: float = 0.92  # Increased from 0.85 for slower decay
    
    # Velocity-based prediction for interpolation
    USE_VELOCITY_PREDICTION: bool = True
    PREDICTION_CONFIDENCE_DECAY: float = 0.9
    
    # Interaction detection
    ROI_RADIUS: float = 80.0
    MIN_INTERACTION_FRAMES: int = 50
    
    # Smoothing - ADJUSTED
    SMOOTH_WINDOW: int = 11  # Reduced from 15 for faster response
    SMOOTH_POLY: int = 2
    MAX_SVO_DELTA: float = 3.5  # Increased to allow more dynamic changes
    EMA_ALPHA: float = 0.2  # Exponential moving average factor
    
    # Visualization
    TRAIL_LENGTH: int = 75
    FPS: int = 25
    GRID_RESOLUTION: int = 60
    
    # Merge zone detection (estimated x-coordinate range for merge)
    MERGE_ZONE_X_START: float = 100.0
    MERGE_ZONE_X_END: float = 300.0
    
    # Position colors
    POSITION_COLORS: Dict[str, str] = field(default_factory=lambda: {
        'lead': '#E74C3C', 'rear': '#3498DB',
        'leftLead': '#9B59B6', 'leftAlongside': '#E67E22', 'leftRear': '#1ABC9C',
        'rightLead': '#F1C40F', 'rightAlongside': '#2ECC71', 'rightRear': '#34495E',
        'tracked': '#95A5A6',
        'interpolated': '#BDC3C7',
    })


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class VehicleState:
    """Vehicle state at a specific frame."""
    track_id: int
    frame: int
    x: float
    y: float
    heading: float
    width: float
    length: float
    x_velocity: float
    y_velocity: float
    x_acceleration: float
    y_acceleration: float
    vehicle_class: str
    mass: float = 3000.0
    
    @property
    def speed(self) -> float:
        return np.sqrt(self.x_velocity**2 + self.y_velocity**2)
    
    @property
    def acceleration(self) -> float:
        speed = self.speed
        if speed < 0.1:
            return 0.0
        return (self.x_velocity * self.x_acceleration + 
                self.y_velocity * self.y_acceleration) / speed
    
    @property
    def jerk_proxy(self) -> float:
        """Approximate jerk from acceleration magnitude."""
        return np.sqrt(self.x_acceleration**2 + self.y_acceleration**2)


@dataclass
class TrackedVehicle:
    """Vehicle with enhanced tracking metadata."""
    state: VehicleState
    position_label: str
    last_slot_frame: int
    frames_since_slot: int = 0
    is_interpolated: bool = False
    confidence: float = 1.0  # Tracking confidence (decays over time)
    velocity_history: List[Tuple[float, float]] = field(default_factory=list)


@dataclass
class PairwiseSVO:
    """Bidirectional SVO between two specific vehicles."""
    ego_id: int
    other_id: int
    frame: int
    
    # SVO from ego toward other
    ego_to_other_svo: float = 45.0
    # SVO from other toward ego
    other_to_ego_svo: float = 45.0
    
    # Aggressiveness
    ego_to_other_aggr: float = 0.0
    other_to_ego_aggr: float = 0.0
    
    # Relative state
    distance: float = 0.0
    delta_x: float = 0.0  # Other.x - Ego.x
    delta_y: float = 0.0  # Other.y - Ego.y
    relative_velocity: float = 0.0
    
    # Phase
    merge_phase: str = 'unknown'  # 'pre_merge', 'during_merge', 'post_merge'


@dataclass
class SVOState:
    """Complete SVO state for ego vehicle."""
    ego_id: int
    frame: int
    
    # Per-vehicle pairwise SVOs
    pairwise_svos: Dict[int, PairwiseSVO] = field(default_factory=dict)
    
    # Aggregates
    mean_ego_svo: float = 45.0
    mean_others_svo: float = 45.0
    total_exerted: float = 0.0
    total_suffered: float = 0.0
    
    # Position in merge zone
    merge_progress: float = 0.0  # 0 = start, 1 = end of merge zone


# =============================================================================
# Improved Hysteresis Tracker
# =============================================================================

class ImprovedHysteresisTracker:
    """
    Enhanced tracker that prevents abrupt vehicle disappearances.
    
    Key improvements:
    1. Longer hysteresis window (25 frames vs 10)
    2. Velocity-based state prediction
    3. Confidence scoring for tracked vehicles
    4. Smooth transition when vehicles reappear
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.tracked_vehicles: Dict[int, TrackedVehicle] = {}
        self.vehicle_history: Dict[int, deque] = defaultdict(lambda: deque(maxlen=30))
    
    def update(
        self,
        frame: int,
        ego_state: VehicleState,
        slot_vehicles: Dict[str, Optional[VehicleState]],
        all_frame_vehicles: Dict[int, VehicleState]
    ) -> Dict[str, TrackedVehicle]:
        """Update tracking with improved hysteresis."""
        
        result = {}
        seen_ids = set()
        
        # 1. Process slot-detected vehicles
        for slot_name, vehicle in slot_vehicles.items():
            if vehicle is None:
                continue
            
            seen_ids.add(vehicle.track_id)
            
            # Update velocity history
            self.vehicle_history[vehicle.track_id].append(
                (vehicle.x_velocity, vehicle.y_velocity)
            )
            
            tracked = TrackedVehicle(
                state=vehicle,
                position_label=slot_name,
                last_slot_frame=frame,
                frames_since_slot=0,
                is_interpolated=False,
                confidence=1.0,
                velocity_history=list(self.vehicle_history[vehicle.track_id])
            )
            
            self.tracked_vehicles[vehicle.track_id] = tracked
            result[slot_name] = tracked
        
        # 2. Check previously tracked vehicles with improved hysteresis
        vehicles_to_remove = []
        
        for vid, tracked in list(self.tracked_vehicles.items()):
            if vid in seen_ids or vid == ego_state.track_id:
                continue
            
            frames_since = frame - tracked.last_slot_frame
            
            if frames_since > self.config.TRACKING_HYSTERESIS:
                vehicles_to_remove.append(vid)
                continue
            
            # Try to find in all_frame_vehicles first
            if vid in all_frame_vehicles:
                new_state = all_frame_vehicles[vid]
                dist = np.sqrt(
                    (new_state.x - ego_state.x)**2 + 
                    (new_state.y - ego_state.y)**2
                )
                
                if dist <= self.config.MAX_TRACKING_DISTANCE:
                    # Update with actual state
                    self.vehicle_history[vid].append(
                        (new_state.x_velocity, new_state.y_velocity)
                    )
                    
                    # Confidence recovers when we have actual data
                    new_confidence = min(1.0, tracked.confidence + 0.1)
                    
                    tracked.state = new_state
                    tracked.frames_since_slot = frames_since
                    tracked.is_interpolated = False
                    tracked.confidence = new_confidence
                    tracked.velocity_history = list(self.vehicle_history[vid])
                    
                    label = f"tracked_{vid}"
                    result[label] = tracked
                    seen_ids.add(vid)
                    continue
            
            # Use velocity-based prediction if enabled
            if (self.config.USE_VELOCITY_PREDICTION and 
                frames_since <= self.config.INTERPOLATION_WINDOW):
                
                predicted = self._predict_state(tracked, frames_since)
                if predicted:
                    # Check if prediction is still within tracking distance
                    dist = np.sqrt(
                        (predicted.x - ego_state.x)**2 + 
                        (predicted.y - ego_state.y)**2
                    )
                    
                    if dist <= self.config.MAX_TRACKING_DISTANCE:
                        # Decay confidence
                        decay = self.config.PREDICTION_CONFIDENCE_DECAY ** frames_since
                        tracked.state = predicted
                        tracked.frames_since_slot = frames_since
                        tracked.is_interpolated = True
                        tracked.confidence = tracked.confidence * decay
                        
                        label = f"interpolated_{vid}"
                        result[label] = tracked
                        seen_ids.add(vid)
                        continue
            
            # If we get here and frames_since is small, keep with low confidence
            if frames_since <= 5:
                label = f"fading_{vid}"
                tracked.frames_since_slot = frames_since
                tracked.confidence *= 0.7
                result[label] = tracked
                seen_ids.add(vid)
            else:
                vehicles_to_remove.append(vid)
        
        # Clean up
        for vid in vehicles_to_remove:
            if vid in self.tracked_vehicles:
                del self.tracked_vehicles[vid]
            if vid in self.vehicle_history:
                del self.vehicle_history[vid]
        
        return result
    
    def _predict_state(
        self, 
        tracked: TrackedVehicle, 
        frames_elapsed: int
    ) -> Optional[VehicleState]:
        """Predict vehicle state using velocity history."""
        
        old = tracked.state
        dt = frames_elapsed / self.config.FPS
        
        # Use average velocity from history for more stable prediction
        if tracked.velocity_history:
            recent_vels = tracked.velocity_history[-min(5, len(tracked.velocity_history)):]
            avg_vx = np.mean([v[0] for v in recent_vels])
            avg_vy = np.mean([v[1] for v in recent_vels])
        else:
            avg_vx = old.x_velocity
            avg_vy = old.y_velocity
        
        # Apply decay to predicted velocity
        decay = self.config.INTERPOLATION_DECAY ** frames_elapsed
        pred_vx = avg_vx * decay
        pred_vy = avg_vy * decay
        
        # Predict position
        new_x = old.x + avg_vx * dt
        new_y = old.y + avg_vy * dt
        
        return VehicleState(
            track_id=old.track_id,
            frame=old.frame + frames_elapsed,
            x=new_x,
            y=new_y,
            heading=old.heading,
            width=old.width,
            length=old.length,
            x_velocity=pred_vx,
            y_velocity=pred_vy,
            x_acceleration=old.x_acceleration * decay,
            y_acceleration=old.y_acceleration * decay,
            vehicle_class=old.vehicle_class,
            mass=old.mass
        )
    
    def reset(self):
        """Reset tracker state."""
        self.tracked_vehicles = {}
        self.vehicle_history = defaultdict(lambda: deque(maxlen=30))
    
    def get_tracking_stats(self) -> Dict:
        """Get tracking statistics."""
        n_slot = sum(1 for t in self.tracked_vehicles.values() if t.frames_since_slot == 0)
        n_hysteresis = sum(1 for t in self.tracked_vehicles.values() 
                         if t.frames_since_slot > 0 and not t.is_interpolated)
        n_interpolated = sum(1 for t in self.tracked_vehicles.values() if t.is_interpolated)
        avg_confidence = np.mean([t.confidence for t in self.tracked_vehicles.values()]) if self.tracked_vehicles else 0
        
        return {
            'slot_detected': n_slot,
            'hysteresis_tracked': n_hysteresis,
            'interpolated': n_interpolated,
            'total': len(self.tracked_vehicles),
            'avg_confidence': avg_confidence
        }


# =============================================================================
# Dynamic SVO Calculator - FIXED
# =============================================================================

class DynamicSVOCalculator:
    """
    Computes SVO with proper sensitivity to state changes.
    
    Key fixes:
    1. Dynamic normalization based on current interaction context
    2. Relative motion component for responsive SVO
    3. Per-pair history for smooth but responsive transitions
    4. Phase-aware computation (pre/during/post merge)
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.pair_history: Dict[Tuple[int, int], deque] = defaultdict(
            lambda: deque(maxlen=50)
        )
        self.ego_svo_history: deque = deque(maxlen=100)
    
    def compute_svo_state(
        self,
        ego: VehicleState,
        tracked_vehicles: Dict[str, TrackedVehicle],
        merge_zone: Tuple[float, float] = None
    ) -> SVOState:
        """Compute complete SVO state with pairwise details."""
        
        if merge_zone is None:
            merge_zone = (self.config.MERGE_ZONE_X_START, self.config.MERGE_ZONE_X_END)
        
        state = SVOState(ego_id=ego.track_id, frame=ego.frame)
        
        # Compute merge progress
        state.merge_progress = self._compute_merge_progress(ego.x, merge_zone)
        
        ego_svos = []
        others_svos = []
        
        for label, tracked in tracked_vehicles.items():
            other = tracked.state
            if other.track_id == ego.track_id:
                continue
            
            # Weight by tracking confidence
            confidence = tracked.confidence
            
            # Compute pairwise SVO
            pairwise = self._compute_pairwise_svo(ego, other, merge_zone, confidence)
            state.pairwise_svos[other.track_id] = pairwise
            
            # Accumulate
            state.total_exerted += pairwise.ego_to_other_aggr * confidence
            state.total_suffered += pairwise.other_to_ego_aggr * confidence
            
            ego_svos.append(pairwise.ego_to_other_svo * confidence)
            others_svos.append(pairwise.other_to_ego_svo * confidence)
        
        # Compute weighted means
        if ego_svos:
            total_weight = sum(tracked.confidence for tracked in tracked_vehicles.values() 
                             if tracked.state.track_id != ego.track_id)
            if total_weight > 0:
                state.mean_ego_svo = sum(ego_svos) / total_weight
                state.mean_others_svo = sum(others_svos) / total_weight
        
        # Apply EMA smoothing to aggregate SVO
        state.mean_ego_svo = self._apply_ema(state.mean_ego_svo, 'ego')
        state.mean_others_svo = self._apply_ema(state.mean_others_svo, 'others')
        
        return state
    
    def _compute_pairwise_svo(
        self,
        ego: VehicleState,
        other: VehicleState,
        merge_zone: Tuple[float, float],
        confidence: float
    ) -> PairwiseSVO:
        """Compute bidirectional SVO for a specific pair."""
        
        # Relative state
        dx = other.x - ego.x
        dy = other.y - ego.y
        dist = np.sqrt(dx**2 + dy**2)
        
        # Relative velocity
        rel_vx = other.x_velocity - ego.x_velocity
        rel_vy = other.y_velocity - ego.y_velocity
        rel_speed = np.sqrt(rel_vx**2 + rel_vy**2)
        
        # Determine merge phase
        phase = self._determine_phase(ego.x, other.x, merge_zone)
        
        # Compute bidirectional aggressiveness
        aggr_ego_to_other = self._compute_aggressiveness(ego, other, dist, dx, dy)
        aggr_other_to_ego = self._compute_aggressiveness(other, ego, dist, -dx, -dy)
        
        # Compute INDEPENDENT SVOs with dynamic sensitivity
        ego_svo = self._compute_dynamic_svo(
            ego, other, aggr_ego_to_other, dist, rel_speed, phase
        )
        other_svo = self._compute_dynamic_svo(
            other, ego, aggr_other_to_ego, dist, rel_speed, phase
        )
        
        # Apply per-pair smoothing
        pair_key = (ego.track_id, other.track_id)
        ego_svo = self._smooth_pair_svo(pair_key, ego_svo, confidence)
        
        pair_key_rev = (other.track_id, ego.track_id)
        other_svo = self._smooth_pair_svo(pair_key_rev, other_svo, confidence)
        
        return PairwiseSVO(
            ego_id=ego.track_id,
            other_id=other.track_id,
            frame=ego.frame,
            ego_to_other_svo=ego_svo,
            other_to_ego_svo=other_svo,
            ego_to_other_aggr=aggr_ego_to_other,
            other_to_ego_aggr=aggr_other_to_ego,
            distance=dist,
            delta_x=dx,
            delta_y=dy,
            relative_velocity=rel_speed,
            merge_phase=phase
        )
    
    def _compute_dynamic_svo(
        self,
        vehicle: VehicleState,
        other: VehicleState,
        aggr_exerted: float,
        distance: float,
        rel_speed: float,
        phase: str
    ) -> float:
        """
        Compute SVO with proper dynamic sensitivity.
        
        Key changes from original:
        1. Context-dependent normalization
        2. Relative motion component
        3. Phase-aware adjustments
        """
        
        # 1. Dynamic normalization for aggressiveness
        # Instead of fixed potential, use context-dependent normalization
        context_factor = self._compute_context_factor(vehicle, other, distance)
        
        if context_factor > 1e-6:
            normalized_aggr = np.clip(aggr_exerted / context_factor, 0, 1)
        else:
            normalized_aggr = 0.5
        
        # 2. Deceleration behavior (more sensitive)
        decel = max(0, -vehicle.acceleration)
        # Use tanh for smoother mapping
        normalized_decel = np.tanh(decel / 2.0)
        
        # 3. Speed-based yielding
        speed_ratio = vehicle.speed / max(self.config.V_REF, 1)
        normalized_yielding = np.clip(1 - speed_ratio, 0, 1)
        
        # 4. NEW: Relative motion component
        # If closing in on other vehicle rapidly, less cooperative
        approach_rate = -rel_speed if distance < self.config.DIST_REF else 0
        normalized_approach = np.clip(approach_rate / 10.0, -1, 1)
        
        # 5. Phase adjustments
        phase_modifier = 1.0
        if phase == 'during_merge':
            phase_modifier = 1.2  # Interactions matter more during merge
        elif phase == 'pre_merge':
            phase_modifier = 0.9
        
        # Combine components with rebalanced weights
        w1 = self.config.WEIGHT_AGGR_RATIO
        w2 = self.config.WEIGHT_DECEL
        w3 = self.config.WEIGHT_YIELDING
        w4 = self.config.WEIGHT_RELATIVE_MOTION
        
        # Normalize weights
        total_w = w1 + w2 + w3 + w4
        w1, w2, w3, w4 = w1/total_w, w2/total_w, w3/total_w, w4/total_w
        
        # SVO components
        svo_aggr = 90 - 135 * normalized_aggr  # -45 to 90
        svo_decel = 45 * normalized_decel  # 0 to ~45
        svo_yield = -22.5 + 67.5 * normalized_yielding  # -22.5 to 45
        svo_approach = -20 * normalized_approach  # -20 to 20
        
        svo = phase_modifier * (w1 * svo_aggr + w2 * svo_decel + w3 * svo_yield + w4 * svo_approach)
        
        return np.clip(svo, -45, 90)
    
    def _compute_context_factor(
        self,
        vehicle: VehicleState,
        other: VehicleState,
        distance: float
    ) -> float:
        """Compute context-dependent normalization factor."""
        
        # Base on current speeds and distance
        v_ego = max(vehicle.speed, 1.0)
        v_other = max(other.speed, 1.0)
        
        # Normalization increases with speeds and decreases with distance
        speed_factor = (v_ego + v_other) / (2 * self.config.V_REF)
        dist_factor = self.config.DIST_REF / max(distance, 5.0)
        
        # Mass factor
        mass_factor = vehicle.mass / self.config.MASS_PC
        
        base_factor = 100.0 * speed_factor * dist_factor * mass_factor
        
        return max(base_factor, 10.0)
    
    def _compute_aggressiveness(
        self,
        aggressor: VehicleState,
        sufferer: VehicleState,
        dist: float,
        dx: float,
        dy: float
    ) -> float:
        """Compute directional aggressiveness."""
        
        if dist < 0.1:
            return 0.0
        
        pos_unit_x = dx / dist
        pos_unit_y = dy / dist
        
        v_i = aggressor.speed
        v_j = sufferer.speed
        
        # Angular alignment
        if v_i > 0.1:
            cos_theta_i = (aggressor.x_velocity * pos_unit_x + 
                         aggressor.y_velocity * pos_unit_y) / v_i
        else:
            cos_theta_i = 0.0
        
        if v_j > 0.1:
            cos_theta_j = -(sufferer.x_velocity * pos_unit_x + 
                          sufferer.y_velocity * pos_unit_y) / v_j
        else:
            cos_theta_j = 0.0
        
        cos_theta_i = np.clip(cos_theta_i, -1, 1)
        cos_theta_j = np.clip(cos_theta_j, -1, 1)
        
        # Pseudo-distance in local frame
        phi = aggressor.heading
        x_local = dx * np.cos(phi) + dy * np.sin(phi)
        y_local = -dx * np.sin(phi) + dy * np.cos(phi)
        
        # Anisotropic field with velocity expansion
        expansion = np.exp(2 * self.config.BETA * v_i)
        tau_1 = self.config.TAU_1 * expansion
        tau_2 = self.config.TAU_2
        
        if tau_1 > 0 and tau_2 > 0:
            r_pseudo = np.sqrt((x_local**2)/(tau_1**2) + (y_local**2)/(tau_2**2))
            r_pseudo = r_pseudo * min(tau_1, tau_2)
        else:
            r_pseudo = dist
        r_pseudo = max(r_pseudo, 0.1)
        
        # Aggressiveness formula
        xi_1 = self.config.MU_1 * v_i * cos_theta_i + self.config.MU_2 * v_j * cos_theta_j
        xi_2 = -self.config.SIGMA * (aggressor.mass ** -1) * r_pseudo
        
        mass_term = (aggressor.mass * v_i) / (2 * self.config.DELTA * sufferer.mass)
        omega = mass_term * np.exp(xi_1 + xi_2)
        
        return np.clip(omega, 0, 2000)
    
    def _determine_phase(
        self,
        ego_x: float,
        other_x: float,
        merge_zone: Tuple[float, float]
    ) -> str:
        """Determine interaction phase relative to merge zone."""
        
        avg_x = (ego_x + other_x) / 2
        
        if avg_x < merge_zone[0]:
            return 'pre_merge'
        elif avg_x > merge_zone[1]:
            return 'post_merge'
        else:
            return 'during_merge'
    
    def _compute_merge_progress(
        self,
        x: float,
        merge_zone: Tuple[float, float]
    ) -> float:
        """Compute progress through merge zone (0 to 1)."""
        
        if x <= merge_zone[0]:
            return 0.0
        elif x >= merge_zone[1]:
            return 1.0
        else:
            return (x - merge_zone[0]) / (merge_zone[1] - merge_zone[0])
    
    def _smooth_pair_svo(
        self,
        pair_key: Tuple[int, int],
        svo: float,
        confidence: float
    ) -> float:
        """Apply smoothing to per-pair SVO with confidence weighting."""
        
        self.pair_history[pair_key].append(svo)
        history = list(self.pair_history[pair_key])
        
        if len(history) < 3:
            return svo
        
        # Weighted average with exponential decay
        weights = np.array([self.config.EMA_ALPHA ** i for i in range(len(history))])[::-1]
        weights = weights * confidence
        weights = weights / weights.sum()
        
        smoothed = np.average(history, weights=weights)
        
        # Limit rate of change
        if len(history) > 1:
            prev = history[-2]
            max_delta = self.config.MAX_SVO_DELTA
            smoothed = np.clip(smoothed, prev - max_delta, prev + max_delta)
        
        return smoothed
    
    def _apply_ema(self, value: float, key: str) -> float:
        """Apply exponential moving average."""
        
        if not hasattr(self, '_ema_state'):
            self._ema_state = {}
        
        if key not in self._ema_state:
            self._ema_state[key] = value
            return value
        
        alpha = self.config.EMA_ALPHA
        smoothed = alpha * value + (1 - alpha) * self._ema_state[key]
        self._ema_state[key] = smoothed
        
        return smoothed
    
    def reset(self):
        """Reset calculator state."""
        self.pair_history = defaultdict(lambda: deque(maxlen=50))
        self.ego_svo_history = deque(maxlen=100)
        if hasattr(self, '_ema_state'):
            self._ema_state = {}


# =============================================================================
# Data Loader (Same as original with minor fixes)
# =============================================================================

class ExiDLoader:
    """Loads exiD dataset."""
    
    LOCATION_INFO = {
        0: {'name': 'Double consecutive merge', 'recordings': range(0, 19)},
        1: {'name': 'Merge + weaving', 'recordings': range(19, 39)},
        2: {'name': 'Normal single merge', 'recordings': range(39, 53)},
        3: {'name': 'Normal single merge', 'recordings': range(53, 61)},
        4: {'name': 'Incomplete acceleration lane', 'recordings': range(61, 73)},
        5: {'name': 'Normal single merge', 'recordings': range(73, 78)},
        6: {'name': 'Normal single merge', 'recordings': range(78, 93)},
    }
    
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.tracks_df = None
        self.tracks_meta_df = None
        self.recording_meta = None
        self.background_image = None
        self.ortho_px_to_meter = 0.1
        self.config = Config()
        
        # Cache for frame data (like intersection_visualizer's ids_for_frame)
        self.ids_for_frame: Dict[int, List[int]] = {}
    
    def load_recording(self, recording_id: int) -> bool:
        prefix = f"{recording_id:02d}_"
        
        try:
            logger.info(f"Loading recording {recording_id}...")
            
            self.tracks_df = pd.read_csv(self.data_dir / f"{prefix}tracks.csv")
            self.tracks_meta_df = pd.read_csv(self.data_dir / f"{prefix}tracksMeta.csv")
            rec_meta_df = pd.read_csv(self.data_dir / f"{prefix}recordingMeta.csv")
            self.recording_meta = rec_meta_df.iloc[0]
            
            self.ortho_px_to_meter = self.recording_meta.get('orthoPxToMeter', 0.1)
            
            # Load background
            bg_path = self.data_dir / f"{prefix}background.png"
            if bg_path.exists():
                self.background_image = plt.imread(str(bg_path))
            
            # Merge metadata
            self.tracks_df = self.tracks_df.merge(
                self.tracks_meta_df[['trackId', 'class', 'width', 'length']],
                on='trackId', how='left', suffixes=('', '_meta')
            )
            
            if 'width_meta' in self.tracks_df.columns:
                self.tracks_df['width'] = self.tracks_df['width'].fillna(self.tracks_df['width_meta'])
                self.tracks_df['length'] = self.tracks_df['length'].fillna(self.tracks_df['length_meta'])
            
            # Build frame index (like intersection_visualizer)
            self._build_frame_index()
            
            location_id = int(self.recording_meta.get('locationId', -1))
            location_name = self.LOCATION_INFO.get(location_id, {}).get('name', 'Unknown')
            
            logger.info(f"  Location: {location_id} ({location_name})")
            logger.info(f"  Duration: {self.recording_meta.get('duration', 0):.1f}s")
            
            heavy_mask = self.tracks_meta_df['class'].str.lower().isin(self.config.HEAVY_VEHICLE_CLASSES)
            logger.info(f"  Heavy vehicles: {heavy_mask.sum()}")
            
            return True
        except Exception as e:
            logger.error(f"Error loading recording: {e}")
            return False
    
    def _build_frame_index(self):
        """Build index of which vehicles are present in each frame."""
        max_frame = int(self.tracks_df['frame'].max())
        
        for frame in range(max_frame + 1):
            frame_data = self.tracks_df[self.tracks_df['frame'] == frame]
            self.ids_for_frame[frame] = frame_data['trackId'].tolist()
    
    def get_vehicle_state(self, track_id: int, frame: int) -> Optional[VehicleState]:
        row = self.tracks_df[
            (self.tracks_df['trackId'] == track_id) & 
            (self.tracks_df['frame'] == frame)
        ]
        
        if row.empty:
            return None
        
        row = row.iloc[0]
        vehicle_class = str(row.get('class', 'car')).lower()
        mass = self.config.MASS_HV if vehicle_class in self.config.HEAVY_VEHICLE_CLASSES else self.config.MASS_PC
        
        return VehicleState(
            track_id=int(row['trackId']),
            frame=int(row['frame']),
            x=row['xCenter'],
            y=row['yCenter'],
            heading=np.radians(row.get('heading', 0)),
            width=row.get('width', 2.0),
            length=row.get('length', 5.0),
            x_velocity=row.get('xVelocity', 0.0),
            y_velocity=row.get('yVelocity', 0.0),
            x_acceleration=row.get('xAcceleration', 0.0),
            y_acceleration=row.get('yAcceleration', 0.0),
            vehicle_class=vehicle_class,
            mass=mass
        )
    
    def get_surrounding_vehicles_by_slot(self, track_id: int, frame: int) -> Dict[str, Optional[VehicleState]]:
        """Get surrounding vehicles by slot (original method)."""
        row = self.tracks_df[
            (self.tracks_df['trackId'] == track_id) & 
            (self.tracks_df['frame'] == frame)
        ]
        
        if row.empty:
            return {}
        
        row = row.iloc[0]
        surrounding = {}
        
        slot_mapping = {
            'leadId': 'lead', 'rearId': 'rear',
            'leftLeadId': 'leftLead', 'leftAlongsideId': 'leftAlongside', 
            'leftRearId': 'leftRear', 'rightLeadId': 'rightLead',
            'rightAlongsideId': 'rightAlongside', 'rightRearId': 'rightRear',
        }
        
        for col, label in slot_mapping.items():
            if col in row.index:
                other_id = row[col]
                try:
                    if pd.notna(other_id):
                        other_id_num = pd.to_numeric(other_id, errors='coerce')
                        if pd.notna(other_id_num) and other_id_num > 0:
                            surrounding[label] = self.get_vehicle_state(int(other_id_num), frame)
                        else:
                            surrounding[label] = None
                    else:
                        surrounding[label] = None
                except:
                    surrounding[label] = None
            else:
                surrounding[label] = None
        
        return surrounding
    
    def get_all_vehicles_in_frame(self, frame: int) -> Dict[int, VehicleState]:
        """Get ALL vehicles in a frame."""
        if frame not in self.ids_for_frame:
            return {}
        
        vehicles = {}
        for vid in self.ids_for_frame[frame]:
            state = self.get_vehicle_state(vid, frame)
            if state:
                vehicles[vid] = state
        
        return vehicles
    
    def get_heavy_vehicles(self) -> pd.DataFrame:
        mask = self.tracks_meta_df['class'].str.lower().isin(self.config.HEAVY_VEHICLE_CLASSES)
        return self.tracks_meta_df[mask].copy()
    
    def get_track_trajectory(self, track_id: int) -> pd.DataFrame:
        return self.tracks_df[self.tracks_df['trackId'] == track_id].sort_values('frame')
    
    def get_background_extent(self) -> List[float]:
        if self.background_image is None:
            return [0, 500, -400, 0]
        h, w = self.background_image.shape[:2]
        return [0, w * self.ortho_px_to_meter, -h * self.ortho_px_to_meter, 0]


# =============================================================================
# Interaction Detector
# =============================================================================

class InteractionDetector:
    def __init__(self, loader: ExiDLoader):
        self.loader = loader
        self.config = Config()
    
    def find_interactions(self) -> List[Dict]:
        heavy_vehicles = self.loader.get_heavy_vehicles()
        interactions = []
        
        for _, hv_row in heavy_vehicles.iterrows():
            hv_id = hv_row['trackId']
            hv_track = self.loader.get_track_trajectory(hv_id)
            
            if len(hv_track) < self.config.MIN_INTERACTION_FRAMES:
                continue
            
            frames = hv_track['frame'].values
            dense_frames = []
            
            for frame in frames:
                surrounding = self.loader.get_surrounding_vehicles_by_slot(hv_id, frame)
                n_vehicles = sum(1 for v in surrounding.values() if v is not None)
                if n_vehicles >= 2:
                    dense_frames.append(frame)
            
            if len(dense_frames) >= self.config.MIN_INTERACTION_FRAMES:
                interactions.append({
                    'ego_id': hv_id,
                    'ego_class': hv_row['class'],
                    'frames': dense_frames,
                })
        
        interactions.sort(key=lambda x: len(x['frames']), reverse=True)
        return interactions


# =============================================================================
# Enhanced Visualizer with New Features
# =============================================================================

class EnhancedSVOVisualizer:
    """
    Complete visualizer with:
    1. Animation
    2. Bidirectional SVO plots
    3. SVO evolution vs distance/Δx/Δy
    4. SVO heatmap matrix
    """
    
    def __init__(self, loader: ExiDLoader):
        self.loader = loader
        self.config = Config()
        self.tracker = ImprovedHysteresisTracker(self.config)
        self.svo_calc = DynamicSVOCalculator(self.config)
        
        self.fig = None
        self.axes = {}
        self.elements = {}
    
    def create_complete_analysis(
        self,
        interaction: Dict,
        output_dir: str = './output'
    ):
        """Create all analysis outputs."""
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Creating complete analysis for ego vehicle {interaction['ego_id']}...")
        
        # Reset
        self.tracker.reset()
        self.svo_calc.reset()
        
        # Prepare frame data
        frames_data = self._prepare_frames(interaction)
        
        if not frames_data:
            logger.error("No valid frame data.")
            return
        
        # 1. Create bidirectional SVO plot
        logger.info("Creating bidirectional SVO plot...")
        self._create_bidirectional_svo_plot(frames_data, interaction, output_path)
        
        # 2. Create SVO evolution plots
        logger.info("Creating SVO evolution plots...")
        self._create_svo_evolution_plots(frames_data, interaction, output_path)
        
        # 3. Create SVO heatmap matrix
        logger.info("Creating SVO heatmap matrix...")
        self._create_svo_heatmap(frames_data, interaction, output_path)
        
        # 4. Create summary statistics
        logger.info("Creating summary analysis...")
        self._create_summary_analysis(frames_data, interaction, output_path)
        
        logger.info(f"\n✓ All outputs saved to: {output_path}")
    
    def _prepare_frames(self, interaction: Dict) -> List[Dict]:
        """Prepare data for each frame."""
        
        frames_data = []
        ego_id = interaction['ego_id']
        
        # Accumulators for pairwise tracking
        pairwise_history: Dict[int, List[Dict]] = defaultdict(list)
        
        for i, frame in enumerate(interaction['frames']):
            ego_state = self.loader.get_vehicle_state(ego_id, frame)
            if ego_state is None:
                continue
            
            slot_vehicles = self.loader.get_surrounding_vehicles_by_slot(ego_id, frame)
            all_frame_vehicles = self.loader.get_all_vehicles_in_frame(frame)
            
            tracked_vehicles = self.tracker.update(
                frame, ego_state, slot_vehicles, all_frame_vehicles
            )
            
            tracking_stats = self.tracker.get_tracking_stats()
            svo_state = self.svo_calc.compute_svo_state(ego_state, tracked_vehicles)
            
            time_s = i / self.config.FPS
            
            # Track pairwise SVOs
            for other_id, pairwise in svo_state.pairwise_svos.items():
                pairwise_history[other_id].append({
                    'time': time_s,
                    'frame': frame,
                    'ego_to_other_svo': pairwise.ego_to_other_svo,
                    'other_to_ego_svo': pairwise.other_to_ego_svo,
                    'distance': pairwise.distance,
                    'delta_x': pairwise.delta_x,
                    'delta_y': pairwise.delta_y,
                    'relative_velocity': pairwise.relative_velocity,
                    'merge_phase': pairwise.merge_phase,
                    'ego_to_other_aggr': pairwise.ego_to_other_aggr,
                    'other_to_ego_aggr': pairwise.other_to_ego_aggr,
                })
            
            # Get trajectory history
            ego_track = self.loader.get_track_trajectory(ego_id)
            ego_history = ego_track[
                (ego_track['frame'] <= frame) &
                (ego_track['frame'] > frame - self.config.TRAIL_LENGTH)
            ][['xCenter', 'yCenter']].values
            
            frames_data.append({
                'frame': frame,
                'frame_idx': i,
                'time': time_s,
                'ego': ego_state,
                'tracked_vehicles': tracked_vehicles,
                'tracking_stats': tracking_stats,
                'svo_state': svo_state,
                'ego_history': ego_history,
            })
        
        # Attach pairwise history to last frame
        if frames_data:
            frames_data[-1]['pairwise_history'] = dict(pairwise_history)
        
        return frames_data
    
    def _create_bidirectional_svo_plot(
        self,
        frames_data: List[Dict],
        interaction: Dict,
        output_path: Path
    ):
        """Create bidirectional SVO plot for top interacting pairs."""
        
        pairwise_history = frames_data[-1].get('pairwise_history', {})
        
        if not pairwise_history:
            logger.warning("No pairwise history available.")
            return
        
        # Find top 4 vehicles by interaction duration
        sorted_pairs = sorted(
            pairwise_history.items(),
            key=lambda x: len(x[1]),
            reverse=True
        )[:4]
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.patch.set_facecolor('#0D1117')
        
        for idx, (other_id, history) in enumerate(sorted_pairs):
            ax = axes[idx // 2, idx % 2]
            ax.set_facecolor('#1A1A2E')
            
            if not history:
                continue
            
            times = [h['time'] for h in history]
            ego_svo = [h['ego_to_other_svo'] for h in history]
            other_svo = [h['other_to_ego_svo'] for h in history]
            
            # Background zones
            ax.axhspan(60, 100, alpha=0.15, color='#27AE60')
            ax.axhspan(30, 60, alpha=0.15, color='#3498DB')
            ax.axhspan(0, 30, alpha=0.15, color='#F39C12')
            ax.axhspan(-50, 0, alpha=0.15, color='#E74C3C')
            
            # Plot SVOs
            ax.plot(times, ego_svo, color='#E74C3C', linewidth=2, 
                   label=f'Truck → Car {other_id}')
            ax.plot(times, other_svo, color='#3498DB', linewidth=2, 
                   label=f'Car {other_id} → Truck')
            
            # Fill between to show asymmetry
            ax.fill_between(times, ego_svo, other_svo, alpha=0.2, color='#9B59B6')
            
            ax.axhline(45, color='white', linestyle='--', alpha=0.3)
            
            ax.set_xlabel('Time (s)', color='white')
            ax.set_ylabel('SVO Angle (°)', color='white')
            ax.set_title(f'Bidirectional SVO: Truck ↔ Car {other_id}\n'
                        f'({len(history)} frames)',
                        fontsize=11, fontweight='bold', color='white')
            ax.tick_params(colors='white')
            ax.set_ylim(-50, 100)
            ax.legend(loc='best', fontsize=9)
            ax.grid(True, alpha=0.2)
            
            for spine in ax.spines.values():
                spine.set_color('#4A4A6A')
        
        fig.suptitle(
            f'Mutual SVO Analysis: {interaction["ego_class"].title()} (ID: {interaction["ego_id"]})\n'
            f'Independent Bidirectional SVO for Top 4 Interacting Vehicles',
            fontsize=14, fontweight='bold', color='white', y=0.98
        )
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        save_path = output_path / f'bidirectional_svo_{interaction["ego_id"]}.png'
        fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
        plt.close(fig)
        logger.info(f"  Saved: {save_path.name}")
    
    def _create_svo_evolution_plots(
        self,
        frames_data: List[Dict],
        interaction: Dict,
        output_path: Path
    ):
        """Create SVO evolution plots vs distance, Δx, Δy, merge progress."""
        
        pairwise_history = frames_data[-1].get('pairwise_history', {})
        
        if not pairwise_history:
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.patch.set_facecolor('#0D1117')
        
        # Collect all data
        all_distances = []
        all_delta_x = []
        all_delta_y = []
        all_rel_vel = []
        all_ego_svo = []
        all_other_svo = []
        all_phases = []
        
        for other_id, history in pairwise_history.items():
            for h in history:
                all_distances.append(h['distance'])
                all_delta_x.append(h['delta_x'])
                all_delta_y.append(h['delta_y'])
                all_rel_vel.append(h['relative_velocity'])
                all_ego_svo.append(h['ego_to_other_svo'])
                all_other_svo.append(h['other_to_ego_svo'])
                all_phases.append(h['merge_phase'])
        
        all_distances = np.array(all_distances)
        all_delta_x = np.array(all_delta_x)
        all_delta_y = np.array(all_delta_y)
        all_rel_vel = np.array(all_rel_vel)
        all_ego_svo = np.array(all_ego_svo)
        all_other_svo = np.array(all_other_svo)
        
        # Phase colors
        phase_colors = {
            'pre_merge': '#3498DB',
            'during_merge': '#E74C3C',
            'post_merge': '#27AE60'
        }
        colors = [phase_colors.get(p, '#7F8C8D') for p in all_phases]
        
        # Plot 1: SVO vs Distance
        ax = axes[0, 0]
        ax.set_facecolor('#1A1A2E')
        scatter1 = ax.scatter(all_distances, all_ego_svo, c=colors, alpha=0.5, s=10, label='Truck SVO')
        ax.scatter(all_distances, all_other_svo, c=colors, alpha=0.3, s=10, marker='s', label='Car SVO')
        ax.set_xlabel('Distance (m)', color='white')
        ax.set_ylabel('SVO Angle (°)', color='white')
        ax.set_title('SVO vs Distance to Surrounding Vehicle', fontsize=11, fontweight='bold', color='white')
        ax.tick_params(colors='white')
        ax.axhline(45, color='white', linestyle='--', alpha=0.3)
        ax.grid(True, alpha=0.2)
        for spine in ax.spines.values():
            spine.set_color('#4A4A6A')
        
        # Plot 2: SVO vs Δx
        ax = axes[0, 1]
        ax.set_facecolor('#1A1A2E')
        ax.scatter(all_delta_x, all_ego_svo, c=colors, alpha=0.5, s=10)
        ax.scatter(all_delta_x, all_other_svo, c=colors, alpha=0.3, s=10, marker='s')
        ax.set_xlabel('Δx (m) [Other - Ego]', color='white')
        ax.set_ylabel('SVO Angle (°)', color='white')
        ax.set_title('SVO vs Longitudinal Gap (Δx)', fontsize=11, fontweight='bold', color='white')
        ax.tick_params(colors='white')
        ax.axhline(45, color='white', linestyle='--', alpha=0.3)
        ax.axvline(0, color='white', linestyle=':', alpha=0.3)
        ax.grid(True, alpha=0.2)
        for spine in ax.spines.values():
            spine.set_color('#4A4A6A')
        
        # Plot 3: SVO vs Δy
        ax = axes[0, 2]
        ax.set_facecolor('#1A1A2E')
        ax.scatter(all_delta_y, all_ego_svo, c=colors, alpha=0.5, s=10)
        ax.scatter(all_delta_y, all_other_svo, c=colors, alpha=0.3, s=10, marker='s')
        ax.set_xlabel('Δy (m) [Other - Ego]', color='white')
        ax.set_ylabel('SVO Angle (°)', color='white')
        ax.set_title('SVO vs Lateral Gap (Δy)', fontsize=11, fontweight='bold', color='white')
        ax.tick_params(colors='white')
        ax.axhline(45, color='white', linestyle='--', alpha=0.3)
        ax.axvline(0, color='white', linestyle=':', alpha=0.3)
        ax.grid(True, alpha=0.2)
        for spine in ax.spines.values():
            spine.set_color('#4A4A6A')
        
        # Plot 4: SVO vs Relative Velocity
        ax = axes[1, 0]
        ax.set_facecolor('#1A1A2E')
        ax.scatter(all_rel_vel, all_ego_svo, c=colors, alpha=0.5, s=10)
        ax.scatter(all_rel_vel, all_other_svo, c=colors, alpha=0.3, s=10, marker='s')
        ax.set_xlabel('Relative Velocity (m/s)', color='white')
        ax.set_ylabel('SVO Angle (°)', color='white')
        ax.set_title('SVO vs Relative Velocity', fontsize=11, fontweight='bold', color='white')
        ax.tick_params(colors='white')
        ax.axhline(45, color='white', linestyle='--', alpha=0.3)
        ax.grid(True, alpha=0.2)
        for spine in ax.spines.values():
            spine.set_color('#4A4A6A')
        
        # Plot 5: SVO by Phase (box plot style)
        ax = axes[1, 1]
        ax.set_facecolor('#1A1A2E')
        
        phases = ['pre_merge', 'during_merge', 'post_merge']
        phase_labels = ['Pre-Merge', 'During Merge', 'Post-Merge']
        
        ego_by_phase = {p: [] for p in phases}
        other_by_phase = {p: [] for p in phases}
        
        for i, phase in enumerate(all_phases):
            if phase in ego_by_phase:
                ego_by_phase[phase].append(all_ego_svo[i])
                other_by_phase[phase].append(all_other_svo[i])
        
        x_pos = np.arange(len(phases))
        width = 0.35
        
        ego_means = [np.mean(ego_by_phase[p]) if ego_by_phase[p] else 0 for p in phases]
        ego_stds = [np.std(ego_by_phase[p]) if ego_by_phase[p] else 0 for p in phases]
        other_means = [np.mean(other_by_phase[p]) if other_by_phase[p] else 0 for p in phases]
        other_stds = [np.std(other_by_phase[p]) if other_by_phase[p] else 0 for p in phases]
        
        ax.bar(x_pos - width/2, ego_means, width, yerr=ego_stds, 
              color='#E74C3C', alpha=0.7, label='Truck SVO', capsize=5)
        ax.bar(x_pos + width/2, other_means, width, yerr=other_stds,
              color='#3498DB', alpha=0.7, label='Car SVO', capsize=5)
        
        ax.set_xticks(x_pos)
        ax.set_xticklabels(phase_labels, color='white')
        ax.set_ylabel('SVO Angle (°)', color='white')
        ax.set_title('SVO by Merge Phase', fontsize=11, fontweight='bold', color='white')
        ax.tick_params(colors='white')
        ax.axhline(45, color='white', linestyle='--', alpha=0.3)
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.2, axis='y')
        for spine in ax.spines.values():
            spine.set_color('#4A4A6A')
        
        # Plot 6: Legend/Info
        ax = axes[1, 2]
        ax.set_facecolor('#1A1A2E')
        ax.axis('off')
        
        legend_text = (
            "═══════ LEGEND ═══════\n\n"
            "Point Colors by Phase:\n"
            "  ● Blue: Pre-Merge\n"
            "  ● Red: During Merge\n"
            "  ● Green: Post-Merge\n\n"
            "Point Shapes:\n"
            "  ● Circle: Truck SVO (toward car)\n"
            "  ■ Square: Car SVO (toward truck)\n\n"
            "═══════ INTERPRETATION ═══════\n\n"
            "Δx > 0: Other vehicle ahead\n"
            "Δx < 0: Other vehicle behind\n\n"
            "Δy > 0: Other vehicle to left\n"
            "Δy < 0: Other vehicle to right\n\n"
            "SVO > 45°: Cooperative/Altruistic\n"
            "SVO < 45°: Individualistic/Competitive"
        )
        
        ax.text(0.1, 0.9, legend_text, transform=ax.transAxes,
               fontsize=10, color='white', family='monospace',
               verticalalignment='top')
        
        for spine in ax.spines.values():
            spine.set_color('#4A4A6A')
        
        fig.suptitle(
            f'SVO Evolution Analysis: {interaction["ego_class"].title()} (ID: {interaction["ego_id"]})',
            fontsize=14, fontweight='bold', color='white', y=0.98
        )
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        save_path = output_path / f'svo_evolution_{interaction["ego_id"]}.png'
        fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
        plt.close(fig)
        logger.info(f"  Saved: {save_path.name}")
    
    def _create_svo_heatmap(
        self,
        frames_data: List[Dict],
        interaction: Dict,
        output_path: Path
    ):
        """Create SVO heatmap matrix between truck and cars."""
        
        pairwise_history = frames_data[-1].get('pairwise_history', {})
        
        if not pairwise_history:
            return
        
        # Get all unique vehicle IDs with sufficient data
        ego_id = interaction['ego_id']
        other_ids = sorted([oid for oid, hist in pairwise_history.items() if len(hist) >= 10])
        
        if len(other_ids) < 2:
            logger.warning("Not enough vehicles for heatmap.")
            return
        
        # Limit to top 10 vehicles
        other_ids = other_ids[:10]
        
        # Create matrices
        n = len(other_ids)
        ego_to_others = np.zeros(n)
        others_to_ego = np.zeros(n)
        interaction_duration = np.zeros(n)
        
        for i, oid in enumerate(other_ids):
            hist = pairwise_history[oid]
            ego_to_others[i] = np.mean([h['ego_to_other_svo'] for h in hist])
            others_to_ego[i] = np.mean([h['other_to_ego_svo'] for h in hist])
            interaction_duration[i] = len(hist) / self.config.FPS
        
        # Create figure
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.patch.set_facecolor('#0D1117')
        
        # Custom colormap
        cmap = LinearSegmentedColormap.from_list(
            'svo_cmap',
            ['#E74C3C', '#F39C12', '#3498DB', '#27AE60']
        )
        
        # Plot 1: Truck → Cars
        ax = axes[0]
        ax.set_facecolor('#1A1A2E')
        
        bars1 = ax.barh(range(n), ego_to_others, color=[cmap((v + 45) / 135) for v in ego_to_others])
        ax.set_yticks(range(n))
        ax.set_yticklabels([f'Car {oid}' for oid in other_ids], color='white')
        ax.set_xlabel('SVO Angle (°)', color='white')
        ax.set_title(f'Truck (ID: {ego_id}) → Cars\nMean SVO', 
                    fontsize=11, fontweight='bold', color='white')
        ax.tick_params(colors='white')
        ax.axvline(45, color='white', linestyle='--', alpha=0.5)
        ax.set_xlim(-50, 100)
        ax.grid(True, alpha=0.2, axis='x')
        
        for spine in ax.spines.values():
            spine.set_color('#4A4A6A')
        
        # Add value labels
        for i, v in enumerate(ego_to_others):
            ax.text(v + 2, i, f'{v:.1f}°', va='center', color='white', fontsize=9)
        
        # Plot 2: Cars → Truck
        ax = axes[1]
        ax.set_facecolor('#1A1A2E')
        
        bars2 = ax.barh(range(n), others_to_ego, color=[cmap((v + 45) / 135) for v in others_to_ego])
        ax.set_yticks(range(n))
        ax.set_yticklabels([f'Car {oid}' for oid in other_ids], color='white')
        ax.set_xlabel('SVO Angle (°)', color='white')
        ax.set_title(f'Cars → Truck (ID: {ego_id})\nMean SVO', 
                    fontsize=11, fontweight='bold', color='white')
        ax.tick_params(colors='white')
        ax.axvline(45, color='white', linestyle='--', alpha=0.5)
        ax.set_xlim(-50, 100)
        ax.grid(True, alpha=0.2, axis='x')
        
        for spine in ax.spines.values():
            spine.set_color('#4A4A6A')
        
        for i, v in enumerate(others_to_ego):
            ax.text(v + 2, i, f'{v:.1f}°', va='center', color='white', fontsize=9)
        
        # Plot 3: Asymmetry (Cars - Truck)
        ax = axes[2]
        ax.set_facecolor('#1A1A2E')
        
        asymmetry = others_to_ego - ego_to_others
        colors_asym = ['#27AE60' if a > 0 else '#E74C3C' for a in asymmetry]
        
        bars3 = ax.barh(range(n), asymmetry, color=colors_asym, alpha=0.8)
        ax.set_yticks(range(n))
        ax.set_yticklabels([f'Car {oid}' for oid in other_ids], color='white')
        ax.set_xlabel('Asymmetry (Cars SVO - Truck SVO) (°)', color='white')
        ax.set_title('SVO Asymmetry\n(Positive = Cars more cooperative)', 
                    fontsize=11, fontweight='bold', color='white')
        ax.tick_params(colors='white')
        ax.axvline(0, color='white', linestyle='--', alpha=0.5)
        ax.grid(True, alpha=0.2, axis='x')
        
        for spine in ax.spines.values():
            spine.set_color('#4A4A6A')
        
        for i, v in enumerate(asymmetry):
            ax.text(v + 1 if v >= 0 else v - 5, i, f'{v:+.1f}°', va='center', color='white', fontsize=9)
        
        fig.suptitle(
            f'SVO Interaction Matrix: {interaction["ego_class"].title()} (ID: {interaction["ego_id"]}) with Surrounding Cars',
            fontsize=14, fontweight='bold', color='white', y=0.98
        )
        
        plt.tight_layout(rect=[0, 0, 1, 0.93])
        
        save_path = output_path / f'svo_heatmap_{interaction["ego_id"]}.png'
        fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
        plt.close(fig)
        logger.info(f"  Saved: {save_path.name}")
    
    def _create_summary_analysis(
        self,
        frames_data: List[Dict],
        interaction: Dict,
        output_path: Path
    ):
        """Create summary analysis plot."""
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.patch.set_facecolor('#0D1117')
        
        times = [fd['time'] for fd in frames_data]
        ego_svo = [fd['svo_state'].mean_ego_svo for fd in frames_data]
        others_svo = [fd['svo_state'].mean_others_svo for fd in frames_data]
        exerted = [fd['svo_state'].total_exerted for fd in frames_data]
        suffered = [fd['svo_state'].total_suffered for fd in frames_data]
        n_slot = [fd['tracking_stats']['slot_detected'] for fd in frames_data]
        n_tracked = [fd['tracking_stats']['total'] for fd in frames_data]
        avg_conf = [fd['tracking_stats']['avg_confidence'] for fd in frames_data]
        
        # Plot 1: SVO Comparison
        ax = axes[0, 0]
        ax.set_facecolor('#1A1A2E')
        ax.axhspan(60, 100, alpha=0.15, color='#27AE60')
        ax.axhspan(30, 60, alpha=0.15, color='#3498DB')
        ax.axhspan(0, 30, alpha=0.15, color='#F39C12')
        ax.axhspan(-50, 0, alpha=0.15, color='#E74C3C')
        ax.plot(times, ego_svo, color='#E74C3C', linewidth=2, label='Truck SVO')
        ax.plot(times, others_svo, color='#3498DB', linewidth=2, label='Cars SVO (mean)')
        ax.axhline(45, color='white', linestyle='--', alpha=0.5)
        ax.set_xlabel('Time (s)', color='white')
        ax.set_ylabel('SVO Angle (°)', color='white')
        ax.set_title('Dynamic Bidirectional SVO', fontsize=11, fontweight='bold', color='white')
        ax.tick_params(colors='white')
        ax.set_ylim(-50, 100)
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.2)
        for spine in ax.spines.values():
            spine.set_color('#4A4A6A')
        
        # Plot 2: SVO Asymmetry
        ax = axes[0, 1]
        ax.set_facecolor('#1A1A2E')
        asymmetry = [o - e for e, o in zip(ego_svo, others_svo)]
        ax.fill_between(times, asymmetry, alpha=0.3, color='#9B59B6')
        ax.plot(times, asymmetry, color='#9B59B6', linewidth=2)
        ax.axhline(0, color='white', linestyle='--', alpha=0.5)
        ax.set_xlabel('Time (s)', color='white')
        ax.set_ylabel('Asymmetry (°)', color='white')
        ax.set_title('SVO Asymmetry (Cars - Truck)', fontsize=11, fontweight='bold', color='white')
        ax.tick_params(colors='white')
        ax.grid(True, alpha=0.2)
        for spine in ax.spines.values():
            spine.set_color('#4A4A6A')
        
        # Plot 3: Aggressiveness
        ax = axes[0, 2]
        ax.set_facecolor('#1A1A2E')
        ax.plot(times, exerted, color='#E74C3C', linewidth=2, label='Exerted')
        ax.plot(times, suffered, color='#3498DB', linewidth=2, label='Suffered')
        ax.fill_between(times, exerted, alpha=0.2, color='#E74C3C')
        ax.fill_between(times, suffered, alpha=0.2, color='#3498DB')
        ax.set_xlabel('Time (s)', color='white')
        ax.set_ylabel('Aggressiveness', color='white')
        ax.set_title('Asymmetric Aggressiveness', fontsize=11, fontweight='bold', color='white')
        ax.tick_params(colors='white')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.2)
        for spine in ax.spines.values():
            spine.set_color('#4A4A6A')
        
        # Plot 4: Tracking Stability
        ax = axes[1, 0]
        ax.set_facecolor('#1A1A2E')
        ax.fill_between(times, n_tracked, alpha=0.3, color='#9B59B6')
        ax.plot(times, n_tracked, color='#9B59B6', linewidth=2, label='Total tracked')
        ax.plot(times, n_slot, color='#27AE60', linewidth=2, linestyle='--', label='Slot-based')
        ax.set_xlabel('Time (s)', color='white')
        ax.set_ylabel('Vehicle Count', color='white')
        ax.set_title('Tracking Stability (Hysteresis)', fontsize=11, fontweight='bold', color='white')
        ax.tick_params(colors='white')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.2)
        for spine in ax.spines.values():
            spine.set_color('#4A4A6A')
        
        # Plot 5: Tracking Confidence
        ax = axes[1, 1]
        ax.set_facecolor('#1A1A2E')
        ax.plot(times, avg_conf, color='#F39C12', linewidth=2)
        ax.fill_between(times, avg_conf, alpha=0.3, color='#F39C12')
        ax.set_xlabel('Time (s)', color='white')
        ax.set_ylabel('Confidence', color='white')
        ax.set_title('Average Tracking Confidence', fontsize=11, fontweight='bold', color='white')
        ax.tick_params(colors='white')
        ax.set_ylim(0, 1.1)
        ax.grid(True, alpha=0.2)
        for spine in ax.spines.values():
            spine.set_color('#4A4A6A')
        
        # Plot 6: Statistics Summary
        ax = axes[1, 2]
        ax.set_facecolor('#1A1A2E')
        ax.axis('off')
        
        stats_text = (
            "═══════ ANALYSIS SUMMARY ═══════\n\n"
            f"Duration: {times[-1]:.1f} seconds\n"
            f"Frames analyzed: {len(frames_data)}\n\n"
            "─── SVO Statistics ───\n"
            f"Truck SVO:\n"
            f"  Mean: {np.mean(ego_svo):.1f}°\n"
            f"  Std:  {np.std(ego_svo):.1f}°\n"
            f"  Range: [{np.min(ego_svo):.1f}°, {np.max(ego_svo):.1f}°]\n\n"
            f"Cars SVO (mean):\n"
            f"  Mean: {np.mean(others_svo):.1f}°\n"
            f"  Std:  {np.std(others_svo):.1f}°\n"
            f"  Range: [{np.min(others_svo):.1f}°, {np.max(others_svo):.1f}°]\n\n"
            "─── Tracking Statistics ───\n"
            f"Avg vehicles tracked: {np.mean(n_tracked):.1f}\n"
            f"Avg slot-based: {np.mean(n_slot):.1f}\n"
            f"Hysteresis benefit: +{np.mean(n_tracked) - np.mean(n_slot):.1f}\n"
            f"Avg confidence: {np.mean(avg_conf):.2f}\n\n"
            "─── Key Finding ───\n"
            f"Mean Asymmetry: {np.mean(asymmetry):.1f}°\n"
            f"Cars {abs(np.mean(asymmetry)):.1f}° more "
            f"{'cooperative' if np.mean(asymmetry) > 0 else 'competitive'}"
        )
        
        ax.text(0.1, 0.95, stats_text, transform=ax.transAxes,
               fontsize=10, color='white', family='monospace',
               verticalalignment='top')
        
        for spine in ax.spines.values():
            spine.set_color('#4A4A6A')
        
        fig.suptitle(
            f'Summary Analysis: {interaction["ego_class"].title()} (ID: {interaction["ego_id"]})',
            fontsize=14, fontweight='bold', color='white', y=0.98
        )
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        save_path = output_path / f'summary_analysis_{interaction["ego_id"]}.png'
        fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
        plt.close(fig)
        logger.info(f"  Saved: {save_path.name}")


# =============================================================================
# Main Entry Point
# =============================================================================

def main(data_dir: str, recording_id: int, output_dir: str = './output_corrected'):
    """Main function to run corrected SVO analysis."""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 70)
    logger.info("CORRECTED exiD SVO Visualization")
    logger.info("  - Fixed: Dynamic SVO calculation with proper sensitivity")
    logger.info("  - Fixed: Improved hysteresis tracking (25 frames)")
    logger.info("  - Added: Bidirectional SVO plots")
    logger.info("  - Added: SVO evolution vs distance/Δx/Δy")
    logger.info("  - Added: SVO heatmap matrix")
    logger.info("=" * 70)
    
    # Load data
    loader = ExiDLoader(data_dir)
    if not loader.load_recording(recording_id):
        return None
    
    # Find interactions
    detector = InteractionDetector(loader)
    interactions = detector.find_interactions()
    
    if not interactions:
        logger.warning("No interactions found.")
        return None
    
    logger.info(f"\nFound {len(interactions)} interactions")
    for i, inter in enumerate(interactions[:5]):
        logger.info(f"  {i+1}. {inter['ego_class']} (ID: {inter['ego_id']}) - "
                   f"{len(inter['frames'])} frames")
    
    # Process best interaction
    best = interactions[0]
    logger.info(f"\n{'='*70}")
    logger.info(f"Analyzing: {best['ego_class']} (ID: {best['ego_id']})")
    logger.info(f"{'='*70}")
    
    visualizer = EnhancedSVOVisualizer(loader)
    visualizer.create_complete_analysis(best, str(output_path))
    
    logger.info(f"\n✓ All outputs saved to: {output_path}")
    logger.info("\nGenerated files:")
    for f in output_path.glob('*.png'):
        logger.info(f"  - {f.name}")
    
    return interactions


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Corrected exiD SVO Visualization')
    parser.add_argument('--data_dir', type=str, default='C:\\exiD-tools\\data',
                       help='Directory containing exiD data files')
    parser.add_argument('--recording', type=int, default=25,
                       help='Recording ID to analyze')
    parser.add_argument('--output_dir', type=str, default='./output_corrected',
                       help='Output directory for plots')
    
    args = parser.parse_args()
    main(args.data_dir, args.recording, args.output_dir)
