"""
exiD Dataset: Enhanced Merging Scenario & Risk Analysis for PINN Field Modeling
================================================================================
Based on: "Unraveling Spatial-Temporal Patterns and Heterogeneity of On-Ramp 
Vehicle Merging Behavior: Evidence from the exiD Dataset" (Wang et al., 2024)

Key Enhancements:
1. Eight merging patterns (A-H) classification per the paper
2. Three merging sections (I, II, III) identification from HD map
3. Enhanced truck-car occlusion scenarios for uncertainty modeling
4. Risk metrics: TTC, THW, merging distance ratio, conflict indicators
5. Benchmark comparison framework for field models (GVF, APF, PINN)
6. Six-vehicle surrounding context extraction for ego truck

For PINN-based interaction field learning with physics PDE constraints.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle, FancyArrowPatch
from matplotlib.animation import FuncAnimation, PillowWriter
import matplotlib.patches as mpatches
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Set, NamedTuple
from collections import defaultdict
from enum import Enum, auto
import warnings
import logging
import json
from scipy.spatial.distance import cdist
from scipy.signal import find_peaks

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


# =============================================================================
# Merging Pattern Definitions (from Wang et al. 2024)
# =============================================================================

class MergingPattern(Enum):
    """Eight merging patterns based on LV/RV presence and transitions.
    
    Reference: Table 4 in Wang et al. 2024
    """
    A = "no_vehicles"           # No LV, No RV
    B = "lead_only"             # LV exists, No RV
    C = "rear_to_lead"          # LV exists (was RV), No RV at merge
    D = "rear_only"             # No LV, RV exists
    E = "lead_and_rear"         # LV exists, RV exists
    F = "rear_to_lead_with_rv"  # LV exists (was RV), RV exists
    G = "lead_to_rear"          # No LV, RV exists (was LV - ego cut in front)
    H = "lead_to_rear_with_lv"  # LV exists, RV exists (was LV - ego cut in front)
    UNKNOWN = "unknown"


class MergingSection(Enum):
    """Three merging sections based on HD map segmentation."""
    SECTION_I = "section_I"       # Before acceleration lane (solid line)
    SECTION_II = "section_II"     # Acceleration lane (dashed line)
    SECTION_III = "section_III"   # After merge completion
    MAINLINE = "mainline"
    UNKNOWN = "unknown"


class AgentRole(Enum):
    """Agent behavioral roles in highway merging scenarios."""
    NORMAL_MAIN = "normal_main"
    MERGING = "merging"
    EXITING = "exiting"
    YIELDING = "yielding"
    PLATOON_MEMBER = "platoon_member"
    UNKNOWN = "unknown"


class OcclusionScenario(Enum):
    """Occlusion scenarios for uncertainty modeling."""
    FRONT_OCCLUSION = "front_occlusion"       # Car behind truck can't see ahead
    LATERAL_OCCLUSION = "lateral_occlusion"   # Car on left can't see merging car on right
    MERGE_CONFLICT = "merge_conflict"         # Potential conflict due to occlusion
    NONE = "none"


class ConflictType(Enum):
    """Types of potential conflicts in merging scenarios."""
    REAR_END = "rear_end"
    SIDE_SWIPE = "side_swipe"
    CUT_IN = "cut_in"
    FORCED_YIELD = "forced_yield"
    NONE = "none"


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class AgentState:
    """Complete agent state with role and merging information."""
    id: int
    x: float
    y: float
    vx: float
    vy: float
    ax: float
    ay: float
    heading: float
    speed: float
    length: float
    width: float
    vehicle_class: str
    mass: float
    lane_id: Optional[int] = None
    role: AgentRole = AgentRole.UNKNOWN
    role_confidence: float = 0.0
    urgency: float = 0.0
    merging_section: MergingSection = MergingSection.UNKNOWN
    lanelet_id: Optional[int] = None


@dataclass
class MergingEvent:
    """Describes a complete merging event with pattern classification."""
    vehicle_id: int
    recording_id: int
    start_frame: int
    merge_frame: int  # Critical moment when crossing lane line
    end_frame: int
    pattern: MergingPattern
    section: MergingSection
    vehicle_class: str
    
    # Spatial metrics
    merge_x: float = 0.0
    merge_y: float = 0.0
    merging_distance: float = 0.0  # Distance along acceleration lane
    merging_distance_ratio: float = 0.0  # Ratio of accel lane used
    
    # Temporal metrics
    merging_duration: float = 0.0  # seconds
    
    # Kinematic metrics at merge moment
    merge_speed: float = 0.0
    merge_vx: float = 0.0
    merge_vy: float = 0.0
    merge_ax: float = 0.0
    merge_ay: float = 0.0
    
    # Risk metrics
    ttc_lead: float = float('inf')
    ttc_rear: float = float('inf')
    thw_lead: float = float('inf')  # Time headway
    thw_rear: float = float('inf')
    min_gap_lead: float = float('inf')
    min_gap_rear: float = float('inf')
    
    # Surrounding vehicle info at merge moment
    lead_vehicle_id: Optional[int] = None
    rear_vehicle_id: Optional[int] = None
    alongside_vehicle_id: Optional[int] = None
    
    # Pattern transition info
    lv_was_alongside: bool = False
    rv_was_alongside: bool = False


@dataclass
class OcclusionEvent:
    """Describes an occlusion relationship for uncertainty modeling."""
    frame: int
    occluder_id: int          # Vehicle blocking view (typically truck)
    occluded_id: int          # Vehicle that is hidden
    blocked_id: int           # Observer who cannot see
    scenario: OcclusionScenario
    occlusion_ratio: float
    
    # Geometric info
    occluder_x: float = 0.0
    occluder_y: float = 0.0
    occluded_x: float = 0.0
    occluded_y: float = 0.0
    blocked_x: float = 0.0
    blocked_y: float = 0.0
    
    # Conflict potential
    conflict_type: ConflictType = ConflictType.NONE
    ttc_if_conflict: float = float('inf')
    
    # Shadow geometry for visualization
    shadow_polygon: Optional[np.ndarray] = None


@dataclass
class SurroundingContext:
    """Six-vehicle surrounding context for ego truck."""
    ego_id: int
    frame: int
    
    # Surrounding vehicles (can be None if not present)
    left_lead: Optional[AgentState] = None
    left_alongside: Optional[AgentState] = None
    left_rear: Optional[AgentState] = None
    right_lead: Optional[AgentState] = None  # Typically merging vehicle
    right_alongside: Optional[AgentState] = None
    right_rear: Optional[AgentState] = None
    
    # Same-lane vehicles
    front: Optional[AgentState] = None
    rear: Optional[AgentState] = None
    
    # Occlusion relationships
    occlusions: List[OcclusionEvent] = field(default_factory=list)


@dataclass
class RiskMetrics:
    """Comprehensive risk metrics for field model comparison."""
    ttc: float = float('inf')              # Time-to-Collision
    thw: float = float('inf')              # Time Headway
    drac: float = 0.0                      # Deceleration Rate to Avoid Collision
    pet: float = float('inf')              # Post-Encroachment Time
    safety_margin: float = float('inf')    # Gap minus safety buffer
    
    # Field model outputs (to be computed)
    apf_value: float = 0.0                 # Artificial Potential Field
    gvf_value: float = 0.0                 # Generalized Velocity Field
    pinn_field_value: float = 0.0          # PINN-computed field
    
    # Risk classification
    is_high_risk: bool = False
    risk_level: str = "low"  # low, medium, high, critical


# =============================================================================
# Configuration
# =============================================================================

@dataclass  
class Config:
    """Configuration for analysis."""
    
    HEAVY_VEHICLE_CLASSES: Set[str] = field(default_factory=lambda: {'truck', 'bus', 'van', 'trailer'})
    CAR_CLASSES: Set[str] = field(default_factory=lambda: {'car'})
    
    # Observation range for surrounding vehicles
    OBS_RANGE_AHEAD: float = 100.0
    OBS_RANGE_BEHIND: float = 50.0
    OBS_RANGE_LEFT: float = 15.0
    OBS_RANGE_RIGHT: float = 15.0
    
    # Merging detection thresholds
    LATERAL_VEL_THRESHOLD: float = 0.3  # m/s for lane change detection
    LANE_CHANGE_Y_DELTA: float = 2.0    # m lateral movement
    LOOKBACK_FRAMES: int = 5            # Frames to look back for pattern detection
    
    # Merging pattern detection distance (100m as in paper)
    PATTERN_DETECTION_RANGE: float = 100.0
    
    # Risk thresholds (from paper: TTC < 2.5s is high risk)
    TTC_HIGH_RISK: float = 2.5
    TTC_CRITICAL: float = 1.5
    THW_HIGH_RISK: float = 1.0
    
    # Occlusion detection
    OCCLUSION_RANGE: float = 80.0
    FOV_RANGE: float = 150.0
    SAME_DIRECTION_THRESHOLD: float = np.pi / 2
    
    # Physical parameters
    MASS_HV: float = 15000.0  # kg for heavy vehicles
    MASS_PC: float = 3000.0   # kg for passenger cars
    FPS: int = 25
    TIMESTEP: float = 0.04    # seconds per frame
    
    # Merging section boundaries (will be set per location)
    # Default values for location 2 from paper Table 3
    SECTION_BOUNDARIES: Dict = field(default_factory=lambda: {
        2: {'I_length': 67.56, 'II_length': 119.67, 'III_length': 40.65,
            'I_lanelets': [1499, 1500], 'II_lanelets': [1502, 1503], 'III_lanelets': [1574]},
        3: {'I_length': 17.90, 'II_length': 168.03, 'III_length': 32.49,
            'I_lanelets': [1414, 1415], 'II_lanelets': [1524, 1527], 'III_lanelets': [1528]},
        5: {'I_length': 66.54, 'II_length': 132.62, 'III_length': 42.19,
            'I_lanelets': [1408, 1409], 'II_lanelets': [1411, 1412], 'III_lanelets': [1414]},
        6: {'I_length': 26.62, 'II_length': 192.32, 'III_length': 27.40,
            'I_lanelets': [1459, 1460], 'II_lanelets': [1514, 1463], 'III_lanelets': [1467]},
    })
    
    # Visualization colors
    COLORS: Dict[str, str] = field(default_factory=lambda: {
        'truck': '#E74C3C',
        'car': '#3498DB',
        'bus': '#F39C12',
        'van': '#9B59B6',
    })
    
    ROLE_COLORS: Dict[str, str] = field(default_factory=lambda: {
        'normal_main': '#3498DB',
        'merging': '#E74C3C',
        'exiting': '#F39C12',
        'yielding': '#9B59B6',
        'platoon_member': '#2ECC71',
        'unknown': '#95A5A6',
    })
    
    PATTERN_COLORS: Dict[str, str] = field(default_factory=lambda: {
        'A': '#3498DB',  # Blue - free merge
        'B': '#2ECC71',  # Green - safe with lead
        'C': '#F39C12',  # Orange - competitive
        'D': '#9B59B6',  # Purple - cut-in
        'E': '#1ABC9C',  # Teal - constrained
        'F': '#E74C3C',  # Red - high risk
        'G': '#E74C3C',  # Red - aggressive cut-in
        'H': '#E74C3C',  # Red - very high risk
    })
    
    # Theme colors
    BG_DARK: str = '#0D1117'
    BG_PANEL: str = '#1A1A2E'


# =============================================================================
# Merging Pattern Classifier
# =============================================================================

class MergingPatternClassifier:
    """Classifies merging patterns based on surrounding vehicle dynamics.
    
    Implements the 8-pattern classification from Wang et al. 2024.
    """
    
    def __init__(self, config: Config = None):
        self.config = config or Config()
    
    def classify_pattern(self, 
                        ego_trajectory: pd.DataFrame,
                        surrounding_history: Dict[str, List[Optional[int]]],
                        merge_frame: int) -> MergingPattern:
        """
        Classify merging pattern based on LV/RV presence and transitions.
        
        Args:
            ego_trajectory: DataFrame with ego vehicle trajectory
            surrounding_history: Dict with 'lv_history', 'rv_history', 'alongside_history'
                                 over LOOKBACK_FRAMES before merge_frame
            merge_frame: Frame at which merge occurs
            
        Returns:
            MergingPattern enum value
        """
        lookback = self.config.LOOKBACK_FRAMES
        
        lv_history = surrounding_history.get('lv_history', [])
        rv_history = surrounding_history.get('rv_history', [])
        alongside_history = surrounding_history.get('alongside_history', [])
        
        # Current state at merge moment
        has_lv = lv_history[-1] is not None if lv_history else False
        has_rv = rv_history[-1] is not None if rv_history else False
        
        # Check for alongside vehicle that became LV or RV
        had_alongside = any(a is not None for a in alongside_history) if alongside_history else False
        
        # Check if LV was previously alongside (Pattern C, F)
        lv_was_alongside = False
        if has_lv and had_alongside:
            current_lv = lv_history[-1]
            for i, alongside in enumerate(alongside_history[:-1]):
                if alongside == current_lv:
                    lv_was_alongside = True
                    break
        
        # Check if RV was previously LV/alongside (Pattern G, H - ego cut in front)
        rv_was_lead = False
        if has_rv:
            current_rv = rv_history[-1]
            for i, lv in enumerate(lv_history[:-1]):
                if lv == current_rv:
                    rv_was_lead = True
                    break
            if not rv_was_lead:
                for i, alongside in enumerate(alongside_history[:-1]):
                    if alongside == current_rv:
                        rv_was_lead = True
                        break
        
        # Classify based on decision tree
        if not has_lv and not has_rv:
            return MergingPattern.A
        elif has_lv and not has_rv:
            if lv_was_alongside:
                return MergingPattern.C
            else:
                return MergingPattern.B
        elif not has_lv and has_rv:
            if rv_was_lead:
                return MergingPattern.G
            else:
                return MergingPattern.D
        else:  # has_lv and has_rv
            if lv_was_alongside:
                return MergingPattern.F
            elif rv_was_lead:
                return MergingPattern.H
            else:
                return MergingPattern.E
    
    def get_pattern_risk_level(self, pattern: MergingPattern) -> str:
        """Get risk level for a merging pattern."""
        high_risk_patterns = {MergingPattern.C, MergingPattern.F, 
                             MergingPattern.G, MergingPattern.H}
        medium_risk_patterns = {MergingPattern.D, MergingPattern.E}
        
        if pattern in high_risk_patterns:
            return "high"
        elif pattern in medium_risk_patterns:
            return "medium"
        else:
            return "low"


# =============================================================================
# Risk Metrics Calculator
# =============================================================================

class RiskMetricsCalculator:
    """Computes various risk metrics for merging scenarios."""
    
    def __init__(self, config: Config = None):
        self.config = config or Config()
    
    def compute_ttc(self, ego: AgentState, target: AgentState) -> float:
        """Compute Time-to-Collision between ego and target vehicle."""
        # Relative position
        dx = target.x - ego.x
        dy = target.y - ego.y
        
        # Relative velocity
        dvx = ego.vx - target.vx
        dvy = ego.vy - target.vy
        
        # Distance considering vehicle dimensions
        dist = np.sqrt(dx**2 + dy**2) - (ego.length/2 + target.length/2)
        
        if dist <= 0:
            return 0.0  # Already in collision
        
        # Closing speed (negative means approaching)
        closing_speed = (dx * dvx + dy * dvy) / (np.sqrt(dx**2 + dy**2) + 1e-6)
        
        if closing_speed <= 0:
            return float('inf')  # Not approaching
        
        ttc = dist / closing_speed
        return max(0, ttc)
    
    def compute_thw(self, follower: AgentState, leader: AgentState) -> float:
        """Compute Time Headway (THW) for following vehicle."""
        # Longitudinal gap
        gap = leader.x - follower.x - (leader.length/2 + follower.length/2)
        
        if gap <= 0 or follower.vx <= 0:
            return 0.0
        
        return gap / follower.vx
    
    def compute_drac(self, ego: AgentState, target: AgentState) -> float:
        """Compute Deceleration Rate to Avoid Collision (DRAC)."""
        dx = target.x - ego.x
        dy = target.y - ego.y
        dist = np.sqrt(dx**2 + dy**2) - (ego.length/2 + target.length/2)
        
        if dist <= 0:
            return float('inf')  # Already in collision
        
        # Relative velocity
        dvx = ego.vx - target.vx
        dvy = ego.vy - target.vy
        relative_speed = np.sqrt(dvx**2 + dvy**2)
        
        if relative_speed <= 0:
            return 0.0
        
        drac = relative_speed**2 / (2 * dist)
        return drac
    
    def compute_apf(self, ego: AgentState, obstacles: List[AgentState],
                    goal: Tuple[float, float] = None) -> float:
        """
        Compute Artificial Potential Field value at ego position.
        
        Traditional APF for comparison with PINN field.
        """
        # Attractive potential to goal (merge completion point)
        if goal is None:
            goal = (ego.x + 50, ego.y)  # Default: ahead on mainline
        
        goal_dist = np.sqrt((ego.x - goal[0])**2 + (ego.y - goal[1])**2)
        U_att = 0.5 * goal_dist**2
        
        # Repulsive potential from obstacles
        U_rep = 0.0
        rho_0 = 10.0  # Influence range
        eta = 100.0   # Repulsive gain
        
        for obs in obstacles:
            if obs.id == ego.id:
                continue
            
            dx = ego.x - obs.x
            dy = ego.y - obs.y
            rho = np.sqrt(dx**2 + dy**2) - (ego.length/2 + obs.length/2)
            
            if rho <= 0:
                rho = 0.1
            
            if rho < rho_0:
                U_rep += 0.5 * eta * (1/rho - 1/rho_0)**2
        
        return U_att + U_rep
    
    def compute_gvf(self, ego: AgentState, obstacles: List[AgentState],
                    lane_center: float = 0.0) -> Tuple[float, float]:
        """
        Compute Generalized Velocity Field at ego position.
        
        Returns desired velocity direction for comparison with PINN.
        """
        # Lane keeping component
        k_lane = 0.5
        lateral_error = ego.y - lane_center
        vy_desired = -k_lane * lateral_error
        
        # Forward velocity component (maintain speed)
        vx_desired = max(ego.vx, 20.0)  # At least 20 m/s
        
        # Obstacle avoidance adjustment
        for obs in obstacles:
            if obs.id == ego.id:
                continue
            
            dx = obs.x - ego.x
            dy = obs.y - ego.y
            dist = np.sqrt(dx**2 + dy**2)
            
            if dist < 30:  # Influence range
                # Push away from obstacle
                influence = 1.0 / (dist + 1)
                vx_desired -= influence * dx / (dist + 1)
                vy_desired -= influence * dy / (dist + 1)
        
        return (vx_desired, vy_desired)
    
    def compute_all_metrics(self, ego: AgentState, 
                           surrounding: SurroundingContext) -> RiskMetrics:
        """Compute comprehensive risk metrics."""
        metrics = RiskMetrics()
        
        # Find closest threats
        ttc_min = float('inf')
        
        # Check front vehicle
        if surrounding.front:
            ttc = self.compute_ttc(ego, surrounding.front)
            thw = self.compute_thw(ego, surrounding.front)
            metrics.thw = thw
            if ttc < ttc_min:
                ttc_min = ttc
        
        # Check merging vehicles (right side typically)
        for vehicle in [surrounding.right_lead, surrounding.right_alongside, 
                       surrounding.right_rear]:
            if vehicle:
                ttc = self.compute_ttc(ego, vehicle)
                if ttc < ttc_min:
                    ttc_min = ttc
        
        metrics.ttc = ttc_min
        
        # Compute DRAC if there's a conflict
        if surrounding.right_alongside:
            metrics.drac = self.compute_drac(ego, surrounding.right_alongside)
        
        # Risk classification
        if ttc_min < self.config.TTC_CRITICAL:
            metrics.risk_level = "critical"
            metrics.is_high_risk = True
        elif ttc_min < self.config.TTC_HIGH_RISK:
            metrics.risk_level = "high"
            metrics.is_high_risk = True
        elif ttc_min < 5.0:
            metrics.risk_level = "medium"
        else:
            metrics.risk_level = "low"
        
        return metrics


# =============================================================================
# Occlusion Detector for Uncertainty Modeling
# =============================================================================

class OcclusionDetector:
    """
    Detects occlusion scenarios relevant to PINN uncertainty modeling.
    
    Focuses on:
    1. Front occlusion: Cars behind truck can't see ahead
    2. Lateral occlusion: Cars can't see merging vehicles on other side of truck
    """
    
    def __init__(self, config: Config = None):
        self.config = config or Config()
    
    def is_same_direction(self, agent1: Dict, agent2: Dict) -> bool:
        """Check if two agents are traveling in the same direction."""
        if agent1.get('vx', 0) != 0 or agent1.get('vy', 0) != 0:
            dir1 = np.arctan2(agent1.get('vy', 0), agent1.get('vx', 0))
        else:
            dir1 = agent1.get('heading', 0)
            
        if agent2.get('vx', 0) != 0 or agent2.get('vy', 0) != 0:
            dir2 = np.arctan2(agent2.get('vy', 0), agent2.get('vx', 0))
        else:
            dir2 = agent2.get('heading', 0)
        
        angle_diff = abs(dir1 - dir2)
        angle_diff = min(angle_diff, 2 * np.pi - angle_diff)
        
        return angle_diff < self.config.SAME_DIRECTION_THRESHOLD
    
    def compute_vehicle_corners(self, agent: Dict) -> np.ndarray:
        """Compute 4 corners of vehicle bounding box."""
        cx, cy = agent['x'], agent['y']
        heading = agent.get('heading', 0)
        length = agent['length']
        width = agent['width']
        
        half_l, half_w = length/2, width/2
        
        corners_local = np.array([
            [-half_l, -half_w],
            [half_l, -half_w],
            [half_l, half_w],
            [-half_l, half_w]
        ])
        
        cos_h = np.cos(heading)
        sin_h = np.sin(heading)
        R = np.array([[cos_h, -sin_h], [sin_h, cos_h]])
        
        return corners_local @ R.T + np.array([cx, cy])
    
    def compute_shadow_polygon(self, observer: Dict, occluder: Dict,
                               fov_range: float = None) -> np.ndarray:
        """Compute the shadow polygon cast by occluder from observer's view."""
        if fov_range is None:
            fov_range = self.config.FOV_RANGE
            
        obs_x, obs_y = observer['x'], observer['y']
        corners = self.compute_vehicle_corners(occluder)
        
        # Find tangent points (extremes of angle from observer)
        angles = []
        for corner in corners:
            dx = corner[0] - obs_x
            dy = corner[1] - obs_y
            angle = np.arctan2(dy, dx)
            angles.append((angle, corner))
        
        angles.sort(key=lambda x: x[0])
        
        # Find widest angular span
        max_span = 0
        left_pt = angles[0][1]
        right_pt = angles[-1][1]
        
        for i in range(len(angles)):
            for j in range(i+1, len(angles)):
                span = angles[j][0] - angles[i][0]
                if span > max_span:
                    max_span = span
                    right_pt = angles[i][1]
                    left_pt = angles[j][1]
        
        # Extend to FOV range
        left_dir = left_pt - np.array([obs_x, obs_y])
        right_dir = right_pt - np.array([obs_x, obs_y])
        
        left_norm = left_dir / (np.linalg.norm(left_dir) + 1e-6)
        right_norm = right_dir / (np.linalg.norm(right_dir) + 1e-6)
        
        left_far = np.array([obs_x, obs_y]) + left_norm * fov_range
        right_far = np.array([obs_x, obs_y]) + right_norm * fov_range
        
        return np.array([right_pt, left_pt, left_far, right_far])
    
    def check_point_in_shadow(self, point: np.ndarray, shadow: np.ndarray) -> bool:
        """Check if a point is inside the shadow polygon."""
        from matplotlib.path import Path
        path = Path(shadow)
        return path.contains_point(point)
    
    def detect_front_occlusion(self, ego_truck: Dict, followers: List[Dict],
                               vehicles_ahead: List[Dict], frame: int) -> List[OcclusionEvent]:
        """
        Detect scenario 1: Cars behind truck can't see vehicles ahead.
        
        This creates uncertainty for following vehicles about traffic conditions.
        """
        occlusions = []
        
        for follower in followers:
            if follower['id'] == ego_truck['id']:
                continue
            if not self.is_same_direction(follower, ego_truck):
                continue
            
            # Check if follower is behind truck
            dx = ego_truck['x'] - follower['x']
            dy = ego_truck['y'] - follower['y']
            
            if dx < 0:  # Truck is behind follower
                continue
            if abs(dy) > 5.0:  # Not in same lane group
                continue
            
            # Compute shadow from follower's perspective
            shadow = self.compute_shadow_polygon(follower, ego_truck)
            
            # Check which vehicles ahead are occluded
            for target in vehicles_ahead:
                if target['id'] in [ego_truck['id'], follower['id']]:
                    continue
                if not self.is_same_direction(follower, target):
                    continue
                
                target_point = np.array([target['x'], target['y']])
                
                if self.check_point_in_shadow(target_point, shadow):
                    # Compute occlusion ratio based on vehicle visibility
                    corners = self.compute_vehicle_corners(target)
                    occluded_corners = sum(1 for c in corners if 
                                          self.check_point_in_shadow(c, shadow))
                    occlusion_ratio = occluded_corners / 4.0
                    
                    if occlusion_ratio > 0.2:
                        event = OcclusionEvent(
                            frame=frame,
                            occluder_id=ego_truck['id'],
                            occluded_id=target['id'],
                            blocked_id=follower['id'],
                            scenario=OcclusionScenario.FRONT_OCCLUSION,
                            occlusion_ratio=occlusion_ratio,
                            occluder_x=ego_truck['x'],
                            occluder_y=ego_truck['y'],
                            occluded_x=target['x'],
                            occluded_y=target['y'],
                            blocked_x=follower['x'],
                            blocked_y=follower['y'],
                            shadow_polygon=shadow
                        )
                        occlusions.append(event)
        
        return occlusions
    
    def detect_lateral_occlusion(self, ego_truck: Dict, 
                                 left_vehicles: List[Dict],
                                 right_vehicles: List[Dict],
                                 frame: int) -> List[OcclusionEvent]:
        """
        Detect scenario 2: Vehicles on one side can't see vehicles on other side.
        
        Critical for merge conflicts where left-side car can't see merging car.
        """
        occlusions = []
        
        for left_v in left_vehicles:
            if left_v['id'] == ego_truck['id']:
                continue
            if not self.is_same_direction(left_v, ego_truck):
                continue
            
            # Compute shadow from left vehicle's perspective
            shadow = self.compute_shadow_polygon(left_v, ego_truck)
            
            for right_v in right_vehicles:
                if right_v['id'] in [ego_truck['id'], left_v['id']]:
                    continue
                if not self.is_same_direction(left_v, right_v):
                    continue
                
                target_point = np.array([right_v['x'], right_v['y']])
                
                if self.check_point_in_shadow(target_point, shadow):
                    corners = self.compute_vehicle_corners(right_v)
                    occluded_corners = sum(1 for c in corners if 
                                          self.check_point_in_shadow(c, shadow))
                    occlusion_ratio = occluded_corners / 4.0
                    
                    if occlusion_ratio > 0.2:
                        # Check if this could lead to merge conflict
                        conflict_type = ConflictType.NONE
                        ttc_conflict = float('inf')
                        
                        # If right vehicle is merging and left vehicle doesn't know
                        if right_v.get('vy', 0) < -0.3:  # Merging left
                            conflict_type = ConflictType.SIDE_SWIPE
                            # Estimate TTC
                            lateral_dist = abs(left_v['y'] - right_v['y'])
                            lateral_speed = abs(right_v.get('vy', 0.1))
                            ttc_conflict = lateral_dist / lateral_speed if lateral_speed > 0.1 else float('inf')
                        
                        event = OcclusionEvent(
                            frame=frame,
                            occluder_id=ego_truck['id'],
                            occluded_id=right_v['id'],
                            blocked_id=left_v['id'],
                            scenario=OcclusionScenario.LATERAL_OCCLUSION,
                            occlusion_ratio=occlusion_ratio,
                            occluder_x=ego_truck['x'],
                            occluder_y=ego_truck['y'],
                            occluded_x=right_v['x'],
                            occluded_y=right_v['y'],
                            blocked_x=left_v['x'],
                            blocked_y=left_v['y'],
                            conflict_type=conflict_type,
                            ttc_if_conflict=ttc_conflict,
                            shadow_polygon=shadow
                        )
                        
                        if conflict_type != ConflictType.NONE:
                            event.scenario = OcclusionScenario.MERGE_CONFLICT
                        
                        occlusions.append(event)
        
        return occlusions
    
    def detect_all_occlusions_for_truck(self, ego_truck: Dict,
                                        all_vehicles: List[Dict],
                                        frame: int) -> List[OcclusionEvent]:
        """
        Comprehensive occlusion detection for truck as ego vehicle.
        
        Returns all occlusion events where the truck causes visibility issues.
        """
        # Categorize surrounding vehicles
        followers = []      # Behind truck
        ahead = []          # Ahead of truck
        left_side = []      # Left of truck
        right_side = []     # Right of truck (acceleration lane side)
        
        for v in all_vehicles:
            if v['id'] == ego_truck['id']:
                continue
            
            dx = v['x'] - ego_truck['x']
            dy = v['y'] - ego_truck['y']
            
            if dx < -5:  # Behind
                followers.append(v)
            elif dx > 5:  # Ahead
                ahead.append(v)
            
            if dy < -2:  # Left (more negative y in German conventions)
                left_side.append(v)
            elif dy > 2:  # Right (acceleration lane side)
                right_side.append(v)
        
        # Detect both types of occlusion
        front_occlusions = self.detect_front_occlusion(
            ego_truck, followers, ahead, frame
        )
        
        lateral_occlusions = self.detect_lateral_occlusion(
            ego_truck, left_side, right_side, frame
        )
        
        return front_occlusions + lateral_occlusions


# =============================================================================
# Six-Vehicle Surrounding Context Extractor
# =============================================================================

class SurroundingContextExtractor:
    """
    Extracts six surrounding vehicles for ego truck.
    
    Layout:
    [left_lead]     [front]     [right_lead]
    [left_alongside] [EGO]      [right_alongside]
    [left_rear]     [rear]      [right_rear]
    """
    
    def __init__(self, config: Config = None):
        self.config = config or Config()
    
    def extract_context(self, ego: AgentState, 
                       all_agents: List[AgentState],
                       frame: int,
                       occlusion_detector: OcclusionDetector = None) -> SurroundingContext:
        """Extract six surrounding vehicles and occlusion relationships."""
        
        context = SurroundingContext(ego_id=ego.id, frame=frame)
        
        # Convert to dict for occlusion detection
        ego_dict = self._agent_to_dict(ego)
        all_dicts = [self._agent_to_dict(a) for a in all_agents]
        
        # Categorize by relative position
        for agent in all_agents:
            if agent.id == ego.id:
                continue
            
            dx = agent.x - ego.x
            dy = agent.y - ego.y
            dist = np.sqrt(dx**2 + dy**2)
            
            # Skip if too far
            if abs(dx) > self.config.OBS_RANGE_AHEAD + self.config.OBS_RANGE_BEHIND:
                continue
            if abs(dy) > max(self.config.OBS_RANGE_LEFT, self.config.OBS_RANGE_RIGHT):
                continue
            
            # Determine position category
            is_ahead = dx > ego.length / 2
            is_behind = dx < -ego.length / 2
            is_alongside = abs(dx) <= max(ego.length, agent.length) / 2 + 2
            
            is_left = dy < -ego.width / 2
            is_right = dy > ego.width / 2
            is_same_lane = abs(dy) <= ego.width + 1
            
            # Assign to closest slot
            if is_same_lane:
                if is_ahead:
                    if context.front is None or dx < (context.front.x - ego.x):
                        context.front = agent
                elif is_behind:
                    if context.rear is None or dx > (context.rear.x - ego.x):
                        context.rear = agent
            elif is_left:
                if is_ahead:
                    if context.left_lead is None or dx < (context.left_lead.x - ego.x):
                        context.left_lead = agent
                elif is_alongside:
                    if context.left_alongside is None or abs(dy) < abs(context.left_alongside.y - ego.y):
                        context.left_alongside = agent
                elif is_behind:
                    if context.left_rear is None or dx > (context.left_rear.x - ego.x):
                        context.left_rear = agent
            elif is_right:
                if is_ahead:
                    if context.right_lead is None or dx < (context.right_lead.x - ego.x):
                        context.right_lead = agent
                elif is_alongside:
                    if context.right_alongside is None or abs(dy) < abs(context.right_alongside.y - ego.y):
                        context.right_alongside = agent
                elif is_behind:
                    if context.right_rear is None or dx > (context.right_rear.x - ego.x):
                        context.right_rear = agent
        
        # Detect occlusions if ego is a truck
        if occlusion_detector and ego.vehicle_class in self.config.HEAVY_VEHICLE_CLASSES:
            context.occlusions = occlusion_detector.detect_all_occlusions_for_truck(
                ego_dict, all_dicts, frame
            )
        
        return context
    
    def _agent_to_dict(self, agent: AgentState) -> Dict:
        """Convert AgentState to dict for occlusion detection."""
        return {
            'id': agent.id,
            'x': agent.x,
            'y': agent.y,
            'vx': agent.vx,
            'vy': agent.vy,
            'heading': agent.heading,
            'length': agent.length,
            'width': agent.width,
            'class': agent.vehicle_class
        }


# =============================================================================
# Merging Scenario Extractor
# =============================================================================

class MergingScenarioExtractor:
    """
    Extracts complete merging scenarios from exiD data.
    
    Identifies merging events, classifies patterns, and computes risk metrics.
    """
    
    def __init__(self, config: Config = None):
        self.config = config or Config()
        self.pattern_classifier = MergingPatternClassifier(config)
        self.risk_calculator = RiskMetricsCalculator(config)
        self.occlusion_detector = OcclusionDetector(config)
        self.context_extractor = SurroundingContextExtractor(config)
    
    def identify_lane_change_frames(self, track_data: pd.DataFrame) -> List[int]:
        """
        Identify frames where lane change occurs.
        
        Based on paper: laneChange=1 and latLaneCenterOffset transition from +2 to -2
        """
        lane_change_frames = []
        
        if 'laneChange' in track_data.columns:
            lc_frames = track_data[track_data['laneChange'] == 1]['frame'].tolist()
            lane_change_frames.extend(lc_frames)
        
        # Also detect by lateral position change
        if 'yCenter' in track_data.columns:
            y_vals = track_data['yCenter'].values
            frames = track_data['frame'].values
            
            for i in range(1, len(y_vals)):
                dy = y_vals[i] - y_vals[i-1]
                if abs(dy) > self.config.LANE_CHANGE_Y_DELTA / 10:  # Per frame threshold
                    lane_change_frames.append(int(frames[i]))
        
        return sorted(set(lane_change_frames))
    
    def is_merging_vehicle(self, track_data: pd.DataFrame, 
                          location_id: int) -> Tuple[bool, Optional[int]]:
        """
        Determine if vehicle is a merging vehicle based on lanelet sequence.
        
        Returns: (is_merging, merge_frame)
        """
        if 'laneletId' not in track_data.columns:
            # Fallback: use y-position change
            if 'yCenter' in track_data.columns:
                y_start = track_data['yCenter'].iloc[:10].mean()
                y_end = track_data['yCenter'].iloc[-10:].mean()
                
                # If moved significantly leftward (toward mainline)
                if y_start - y_end > self.config.LANE_CHANGE_Y_DELTA:
                    # Find the merge frame
                    lane_change_frames = self.identify_lane_change_frames(track_data)
                    merge_frame = lane_change_frames[0] if lane_change_frames else None
                    return True, merge_frame
            return False, None
        
        # Use lanelet information if available
        section_info = self.config.SECTION_BOUNDARIES.get(location_id, {})
        merge_lanelets = section_info.get('II_lanelets', [])
        main_lanelets = section_info.get('III_lanelets', [])
        
        lanelets = track_data['laneletId'].dropna().unique()
        
        # Check if vehicle passes through merge section to mainline
        was_in_merge = any(ll in merge_lanelets for ll in lanelets)
        ended_in_main = any(ll in main_lanelets for ll in lanelets)
        
        if was_in_merge and ended_in_main:
            # Find merge frame
            for idx, row in track_data.iterrows():
                ll = row.get('laneletId')
                if ll in main_lanelets:
                    return True, int(row['frame'])
        
        return False, None
    
    def extract_merging_event(self, track_id: int,
                              track_data: pd.DataFrame,
                              all_tracks: pd.DataFrame,
                              recording_id: int,
                              location_id: int) -> Optional[MergingEvent]:
        """Extract complete merging event for a vehicle."""
        
        is_merging, merge_frame = self.is_merging_vehicle(track_data, location_id)
        
        if not is_merging or merge_frame is None:
            return None
        
        # Get merge moment data
        merge_row = track_data[track_data['frame'] == merge_frame]
        if merge_row.empty:
            return None
        merge_row = merge_row.iloc[0]
        
        # Determine merging section
        section = self._determine_merging_section(merge_row, location_id)
        
        # Get surrounding vehicle history
        surrounding_history = self._get_surrounding_history(
            track_id, track_data, all_tracks, merge_frame
        )
        
        # Classify pattern
        pattern = self.pattern_classifier.classify_pattern(
            track_data, surrounding_history, merge_frame
        )
        
        # Calculate merging metrics
        start_frame = int(track_data['frame'].min())
        end_frame = int(track_data['frame'].max())
        
        # Merging distance and ratio
        section_info = self.config.SECTION_BOUNDARIES.get(location_id, {})
        total_accel_length = section_info.get('II_length', 150.0)
        
        merge_x = float(merge_row.get('xCenter', 0))
        # Estimate distance along acceleration lane
        merging_distance = merge_x  # Simplified; should use lanelet position
        merging_distance_ratio = min(1.0, merging_distance / total_accel_length)
        
        # Create event
        event = MergingEvent(
            vehicle_id=track_id,
            recording_id=recording_id,
            start_frame=start_frame,
            merge_frame=merge_frame,
            end_frame=end_frame,
            pattern=pattern,
            section=section,
            vehicle_class=str(merge_row.get('class', 'car')),
            merge_x=merge_x,
            merge_y=float(merge_row.get('yCenter', 0)),
            merging_distance=merging_distance,
            merging_distance_ratio=merging_distance_ratio,
            merging_duration=(merge_frame - start_frame) * self.config.TIMESTEP,
            merge_speed=float(np.sqrt(merge_row.get('xVelocity', 0)**2 + 
                                     merge_row.get('yVelocity', 0)**2)),
            merge_vx=float(merge_row.get('xVelocity', 0)),
            merge_vy=float(merge_row.get('yVelocity', 0)),
            merge_ax=float(merge_row.get('xAcceleration', 0)),
            merge_ay=float(merge_row.get('yAcceleration', 0)),
            lead_vehicle_id=surrounding_history.get('lv_history', [None])[-1],
            rear_vehicle_id=surrounding_history.get('rv_history', [None])[-1],
            alongside_vehicle_id=surrounding_history.get('alongside_history', [None])[-1],
            lv_was_alongside=pattern in [MergingPattern.C, MergingPattern.F],
            rv_was_alongside=pattern in [MergingPattern.G, MergingPattern.H]
        )
        
        # Calculate risk metrics
        event = self._compute_risk_metrics(event, track_data, all_tracks, merge_frame)
        
        return event
    
    def _determine_merging_section(self, merge_row: pd.Series, 
                                   location_id: int) -> MergingSection:
        """Determine which merging section the merge occurred in."""
        section_info = self.config.SECTION_BOUNDARIES.get(location_id, {})
        
        if 'laneletId' in merge_row:
            ll = merge_row['laneletId']
            if ll in section_info.get('I_lanelets', []):
                return MergingSection.SECTION_I
            elif ll in section_info.get('II_lanelets', []):
                return MergingSection.SECTION_II
            elif ll in section_info.get('III_lanelets', []):
                return MergingSection.SECTION_III
        
        # Fallback: use x-position
        x = merge_row.get('xCenter', 0)
        section_I_len = section_info.get('I_length', 50)
        section_II_len = section_info.get('II_length', 150)
        
        if x < section_I_len:
            return MergingSection.SECTION_I
        elif x < section_I_len + section_II_len:
            return MergingSection.SECTION_II
        else:
            return MergingSection.SECTION_III
    
    def _get_surrounding_history(self, track_id: int,
                                 track_data: pd.DataFrame,
                                 all_tracks: pd.DataFrame,
                                 merge_frame: int) -> Dict[str, List[Optional[int]]]:
        """Get history of surrounding vehicles over lookback period."""
        lookback = self.config.LOOKBACK_FRAMES
        frames = list(range(merge_frame - lookback, merge_frame + 1))
        
        history = {
            'lv_history': [],
            'rv_history': [],
            'alongside_history': []
        }
        
        for frame in frames:
            frame_data = all_tracks[all_tracks['frame'] == frame]
            ego_row = track_data[track_data['frame'] == frame]
            
            if ego_row.empty:
                history['lv_history'].append(None)
                history['rv_history'].append(None)
                history['alongside_history'].append(None)
                continue
            
            ego = ego_row.iloc[0]
            ego_x = ego['xCenter']
            ego_y = ego['yCenter']
            
            lv_id = None
            rv_id = None
            alongside_id = None
            lv_dist = float('inf')
            rv_dist = float('inf')
            alongside_dist = float('inf')
            
            for _, row in frame_data.iterrows():
                if row['trackId'] == track_id:
                    continue
                
                dx = row['xCenter'] - ego_x
                dy = row['yCenter'] - ego_y
                
                # Check if in target lane (left of ego typically)
                if dy > -self.config.OBS_RANGE_LEFT and dy < 0:
                    if abs(dx) < 5:  # Alongside
                        dist = abs(dy)
                        if dist < alongside_dist:
                            alongside_dist = dist
                            alongside_id = int(row['trackId'])
                    elif dx > 0 and dx < lv_dist:  # Lead
                        lv_dist = dx
                        lv_id = int(row['trackId'])
                    elif dx < 0 and abs(dx) < rv_dist:  # Rear
                        rv_dist = abs(dx)
                        rv_id = int(row['trackId'])
            
            history['lv_history'].append(lv_id)
            history['rv_history'].append(rv_id)
            history['alongside_history'].append(alongside_id)
        
        return history
    
    def _compute_risk_metrics(self, event: MergingEvent,
                              track_data: pd.DataFrame,
                              all_tracks: pd.DataFrame,
                              merge_frame: int) -> MergingEvent:
        """Compute risk metrics for merging event."""
        frame_data = all_tracks[all_tracks['frame'] == merge_frame]
        ego_row = track_data[track_data['frame'] == merge_frame].iloc[0]
        
        ego_x = ego_row['xCenter']
        ego_y = ego_row['yCenter']
        ego_vx = ego_row.get('xVelocity', 0)
        ego_vy = ego_row.get('yVelocity', 0)
        ego_length = ego_row.get('length', 5.0)
        
        # Find lead and rear vehicles and compute metrics
        if event.lead_vehicle_id is not None:
            lv_row = frame_data[frame_data['trackId'] == event.lead_vehicle_id]
            if not lv_row.empty:
                lv = lv_row.iloc[0]
                gap = lv['xCenter'] - ego_x - (lv.get('length', 5)/2 + ego_length/2)
                event.min_gap_lead = max(0, gap)
                
                # TTC
                relative_v = ego_vx - lv.get('xVelocity', 0)
                if relative_v > 0 and gap > 0:
                    event.ttc_lead = gap / relative_v
                
                # THW
                if ego_vx > 0:
                    event.thw_lead = gap / ego_vx
        
        if event.rear_vehicle_id is not None:
            rv_row = frame_data[frame_data['trackId'] == event.rear_vehicle_id]
            if not rv_row.empty:
                rv = rv_row.iloc[0]
                gap = ego_x - rv['xCenter'] - (rv.get('length', 5)/2 + ego_length/2)
                event.min_gap_rear = max(0, gap)
                
                # TTC from rear perspective
                relative_v = rv.get('xVelocity', 0) - ego_vx
                if relative_v > 0 and gap > 0:
                    event.ttc_rear = gap / relative_v
                
                # THW
                rv_vx = rv.get('xVelocity', 0)
                if rv_vx > 0:
                    event.thw_rear = gap / rv_vx
        
        return event


# =============================================================================
# Field Model Comparison Framework
# =============================================================================

class FieldModelComparator:
    """
    Framework for comparing PINN field model with benchmark models.
    
    Benchmark models:
    - APF (Artificial Potential Field)
    - GVF (Generalized Velocity Field)
    - Mechanical Wave model for aggressiveness
    """
    
    def __init__(self, config: Config = None):
        self.config = config or Config()
        self.risk_calculator = RiskMetricsCalculator(config)
    
    def compute_benchmark_fields(self, ego: AgentState,
                                surrounding: SurroundingContext,
                                goal: Tuple[float, float] = None) -> Dict[str, float]:
        """Compute all benchmark field values for comparison."""
        
        # Collect obstacles from surrounding context
        obstacles = []
        for attr in ['front', 'rear', 'left_lead', 'left_alongside', 'left_rear',
                    'right_lead', 'right_alongside', 'right_rear']:
            v = getattr(surrounding, attr, None)
            if v is not None:
                obstacles.append(v)
        
        # APF
        apf_value = self.risk_calculator.compute_apf(ego, obstacles, goal)
        
        # GVF
        lane_center = ego.y  # Simplification: current y as target
        gvf_vx, gvf_vy = self.risk_calculator.compute_gvf(ego, obstacles, lane_center)
        gvf_magnitude = np.sqrt(gvf_vx**2 + gvf_vy**2)
        
        # Mechanical wave / aggressiveness metric
        # Based on acceleration and speed deviation
        target_speed = 25.0  # m/s typical highway speed
        speed_deviation = ego.speed - target_speed
        aggressiveness = np.sqrt(ego.ax**2 + ego.ay**2) + 0.1 * abs(speed_deviation)
        
        return {
            'apf_value': apf_value,
            'gvf_vx': gvf_vx,
            'gvf_vy': gvf_vy,
            'gvf_magnitude': gvf_magnitude,
            'aggressiveness': aggressiveness,
            'ego_x': ego.x,
            'ego_y': ego.y,
            'ego_vx': ego.vx,
            'ego_vy': ego.vy,
            'ego_ax': ego.ax,
            'ego_ay': ego.ay
        }
    
    def evaluate_prediction_accuracy(self, predicted_field: np.ndarray,
                                    actual_trajectory: np.ndarray,
                                    timestep: float = 0.04) -> Dict[str, float]:
        """
        Evaluate field model prediction accuracy.
        
        For PINN comparison: how well does the field predict actual behavior?
        """
        # Convert field to predicted trajectory
        # This is simplified; real implementation would integrate field
        
        # Metrics
        position_error = np.mean(np.linalg.norm(
            predicted_field[:, :2] - actual_trajectory[:, :2], axis=1
        ))
        
        velocity_error = np.mean(np.linalg.norm(
            predicted_field[:, 2:4] - actual_trajectory[:, 2:4], axis=1
        ))
        
        # Final displacement error
        fde = np.linalg.norm(predicted_field[-1, :2] - actual_trajectory[-1, :2])
        
        return {
            'mean_position_error': position_error,
            'mean_velocity_error': velocity_error,
            'final_displacement_error': fde
        }
    
    def create_comparison_dataset(self, merging_events: List[MergingEvent],
                                  occlusion_events: List[OcclusionEvent]) -> pd.DataFrame:
        """
        Create dataset for model comparison.
        
        Each row contains:
        - Scenario features (pattern, section, vehicle types)
        - Ground truth behavior (trajectory)
        - Benchmark field values
        - Risk metrics
        - Occlusion indicators
        """
        records = []
        
        for event in merging_events:
            record = {
                # Event identification
                'vehicle_id': event.vehicle_id,
                'recording_id': event.recording_id,
                'merge_frame': event.merge_frame,
                
                # Pattern classification
                'pattern': event.pattern.value,
                'section': event.section.value,
                'vehicle_class': event.vehicle_class,
                
                # Spatial metrics
                'merge_x': event.merge_x,
                'merge_y': event.merge_y,
                'merging_distance': event.merging_distance,
                'merging_distance_ratio': event.merging_distance_ratio,
                
                # Kinematic state
                'merge_speed': event.merge_speed,
                'merge_vx': event.merge_vx,
                'merge_vy': event.merge_vy,
                'merge_ax': event.merge_ax,
                'merge_ay': event.merge_ay,
                
                # Temporal
                'merging_duration': event.merging_duration,
                
                # Risk metrics
                'ttc_lead': event.ttc_lead if event.ttc_lead != float('inf') else -1,
                'ttc_rear': event.ttc_rear if event.ttc_rear != float('inf') else -1,
                'thw_lead': event.thw_lead if event.thw_lead != float('inf') else -1,
                'thw_rear': event.thw_rear if event.thw_rear != float('inf') else -1,
                'min_gap_lead': event.min_gap_lead if event.min_gap_lead != float('inf') else -1,
                'min_gap_rear': event.min_gap_rear if event.min_gap_rear != float('inf') else -1,
                
                # Pattern features
                'lv_was_alongside': int(event.lv_was_alongside),
                'rv_was_alongside': int(event.rv_was_alongside),
                'has_lead_vehicle': int(event.lead_vehicle_id is not None),
                'has_rear_vehicle': int(event.rear_vehicle_id is not None),
                
                # Risk classification
                'is_high_risk': int(event.ttc_rear < self.config.TTC_HIGH_RISK),
            }
            records.append(record)
        
        # Add occlusion information
        df = pd.DataFrame(records)
        
        # Count occlusions per merge frame
        occ_counts = defaultdict(int)
        for occ in occlusion_events:
            key = (occ.frame,)
            occ_counts[key] += 1
            if occ.scenario == OcclusionScenario.MERGE_CONFLICT:
                occ_counts[key] += 10  # Weight conflicts higher
        
        df['occlusion_uncertainty'] = df['merge_frame'].apply(
            lambda f: occ_counts.get((f,), 0)
        )
        
        return df


# =============================================================================
# Main Analysis Pipeline
# =============================================================================

class MergingAnalysisPipeline:
    """Main pipeline for merging scenario analysis."""
    
    def __init__(self, data_dir: str, config: Config = None):
        self.data_dir = Path(data_dir)
        self.config = config or Config()
        
        self.scenario_extractor = MergingScenarioExtractor(config)
        self.comparator = FieldModelComparator(config)
        
        self.tracks_df = None
        self.tracks_meta = None
        self.recording_meta = None
    
    def load_recording(self, recording_id: int) -> bool:
        """Load recording data."""
        prefix = f"{recording_id:02d}_"
        
        try:
            tracks_path = self.data_dir / f"{prefix}tracks.csv"
            meta_path = self.data_dir / f"{prefix}tracksMeta.csv"
            rec_path = self.data_dir / f"{prefix}recordingMeta.csv"
            
            self.tracks_df = pd.read_csv(tracks_path)
            self.tracks_meta = pd.read_csv(meta_path)
            
            if rec_path.exists():
                self.recording_meta = pd.read_csv(rec_path).iloc[0]
            
            # Merge metadata
            self.tracks_df = self.tracks_df.merge(
                self.tracks_meta[['trackId', 'class', 'width', 'length']],
                on='trackId', how='left', suffixes=('', '_meta')
            )
            
            logger.info(f"Loaded recording {recording_id}: "
                       f"{len(self.tracks_meta)} tracks, "
                       f"{len(self.tracks_df)} frames")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load recording: {e}")
            return False
    
    def extract_all_merging_events(self, recording_id: int,
                                   location_id: int) -> List[MergingEvent]:
        """Extract all merging events from recording."""
        events = []
        
        for track_id in self.tracks_meta['trackId'].unique():
            track_data = self.tracks_df[self.tracks_df['trackId'] == track_id]
            
            event = self.scenario_extractor.extract_merging_event(
                track_id, track_data, self.tracks_df, recording_id, location_id
            )
            
            if event is not None:
                events.append(event)
        
        logger.info(f"Extracted {len(events)} merging events")
        return events
    
    def extract_truck_occlusion_scenarios(self, frame: int) -> List[OcclusionEvent]:
        """Extract all truck-caused occlusion scenarios for a frame."""
        frame_data = self.tracks_df[self.tracks_df['frame'] == frame]
        
        all_vehicles = []
        trucks = []
        
        for _, row in frame_data.iterrows():
            v_dict = {
                'id': int(row['trackId']),
                'x': float(row['xCenter']),
                'y': float(row['yCenter']),
                'vx': float(row.get('xVelocity', 0)),
                'vy': float(row.get('yVelocity', 0)),
                'heading': np.radians(float(row.get('heading', 0))),
                'length': float(row.get('length', 5.0)),
                'width': float(row.get('width', 2.0)),
                'class': str(row.get('class', 'car')).lower()
            }
            all_vehicles.append(v_dict)
            
            if v_dict['class'] in self.config.HEAVY_VEHICLE_CLASSES:
                trucks.append(v_dict)
        
        all_occlusions = []
        for truck in trucks:
            occlusions = self.scenario_extractor.occlusion_detector.detect_all_occlusions_for_truck(
                truck, all_vehicles, frame
            )
            all_occlusions.extend(occlusions)
        
        return all_occlusions
    
    def analyze_and_export(self, recording_id: int, 
                          location_id: int,
                          output_dir: str = './output') -> Dict:
        """Run complete analysis and export results."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        if not self.load_recording(recording_id):
            return {'error': 'Failed to load recording'}
        
        # Extract merging events
        merging_events = self.extract_all_merging_events(recording_id, location_id)
        
        # Extract occlusions (sample frames)
        frames = sorted(self.tracks_df['frame'].unique())
        sample_frames = frames[::25]  # Every 1 second
        
        all_occlusions = []
        for frame in sample_frames:
            occlusions = self.extract_truck_occlusion_scenarios(frame)
            all_occlusions.extend(occlusions)
        
        logger.info(f"Detected {len(all_occlusions)} occlusion events")
        
        # Create comparison dataset
        comparison_df = self.comparator.create_comparison_dataset(
            merging_events, all_occlusions
        )
        
        # Pattern statistics
        pattern_counts = comparison_df['pattern'].value_counts().to_dict()
        
        # Risk statistics
        high_risk_count = comparison_df['is_high_risk'].sum()
        
        # Export results
        comparison_df.to_csv(output_path / f'merging_events_rec{recording_id}.csv', index=False)
        
        # Export occlusion events
        occ_records = [{
            'frame': o.frame,
            'occluder_id': o.occluder_id,
            'occluded_id': o.occluded_id,
            'blocked_id': o.blocked_id,
            'scenario': o.scenario.value,
            'occlusion_ratio': o.occlusion_ratio,
            'conflict_type': o.conflict_type.value,
            'ttc_if_conflict': o.ttc_if_conflict if o.ttc_if_conflict != float('inf') else -1
        } for o in all_occlusions]
        
        occ_df = pd.DataFrame(occ_records)
        occ_df.to_csv(output_path / f'occlusion_events_rec{recording_id}.csv', index=False)
        
        # Summary
        summary = {
            'recording_id': recording_id,
            'location_id': location_id,
            'total_tracks': len(self.tracks_meta),
            'merging_events': len(merging_events),
            'occlusion_events': len(all_occlusions),
            'pattern_distribution': pattern_counts,
            'high_risk_events': int(high_risk_count),
            'high_risk_ratio': float(high_risk_count / len(merging_events)) if merging_events else 0
        }
        
        with open(output_path / f'summary_rec{recording_id}.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Results saved to {output_path}")
        return summary


# =============================================================================
# CLI
# =============================================================================

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Merging Scenario & Risk Analysis for PINN Field Modeling'
    )
    parser.add_argument('--data_dir', type=str, default='./data',
                       help='Path to exiD data directory')
    parser.add_argument('--recording', type=int, default=40,
                       help='Recording ID to analyze')
    parser.add_argument('--location', type=int, default=2,
                       help='Location ID (2, 3, 5, or 6)')
    parser.add_argument('--output_dir', type=str, default='./output_merging',
                       help='Output directory')
    
    args = parser.parse_args()
    
    pipeline = MergingAnalysisPipeline(args.data_dir)
    summary = pipeline.analyze_and_export(args.recording, args.location, args.output_dir)
    
    print("\n" + "="*60)
    print("Analysis Complete")
    print("="*60)
    for key, value in summary.items():
        print(f"  {key}: {value}")
