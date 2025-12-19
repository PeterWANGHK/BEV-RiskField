"""
exiD Dataset: Agent Role Classification and Occlusion Detection (v3)
=====================================================================
Improvements over v2:
1. Occlusion filtered by same traffic direction only
2. Occlusion logging to CSV for training/analysis
3. Ego truck perspective shows occlusions CAUSED BY ego (ego as occluder)
4. Consistent visualization style between plots with proper label placement

For PINN-based interaction field learning.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import matplotlib.patches as mpatches
from matplotlib.patches import Polygon, Circle
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Set
from collections import defaultdict
from enum import Enum
import warnings
import logging
import json

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


# =============================================================================
# Agent Role Definitions
# =============================================================================

class AgentRole(Enum):
    """Agent behavioral roles in highway merging scenarios."""
    NORMAL_MAIN = "normal_main"
    MERGING = "merging"
    EXITING = "exiting"
    YIELDING = "yielding"
    UNKNOWN = "unknown"


class OcclusionType(Enum):
    """Types of occlusion relationships."""
    FULL = "full"
    PARTIAL = "partial"
    NONE = "none"


@dataclass
class AgentState:
    """Complete agent state with role information."""
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


@dataclass
class OcclusionEvent:
    """Describes an occlusion relationship with detailed geometry."""
    frame: int
    occluder_id: int
    occluded_id: int
    blocked_id: int  # Observer who is blocked
    occlusion_type: OcclusionType
    occlusion_ratio: float
    # Positions for reference
    occluder_x: float = 0.0
    occluder_y: float = 0.0
    occluded_x: float = 0.0
    occluded_y: float = 0.0
    blocked_x: float = 0.0
    blocked_y: float = 0.0
    # Geometry for visualization
    tangent_left: Optional[Tuple[float, float]] = None
    tangent_right: Optional[Tuple[float, float]] = None
    shadow_polygon: Optional[np.ndarray] = None


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class Config:
    """Configuration for visualization and analysis."""
    
    HEAVY_VEHICLE_CLASSES: Set[str] = field(default_factory=lambda: {'truck', 'bus', 'van', 'trailer'})
    CAR_CLASSES: Set[str] = field(default_factory=lambda: {'car'})
    
    # Observation range
    OBS_RANGE_AHEAD: float = 60.0
    OBS_RANGE_BEHIND: float = 30.0
    OBS_RANGE_LEFT: float = 15.0
    OBS_RANGE_RIGHT: float = 15.0
    
    # Lane detection thresholds
    LATERAL_VEL_THRESHOLD: float = 0.3
    LANE_CHANGE_Y_DELTA: float = 2.0
    
    # Role classification
    MERGE_URGENCY_DIST: float = 100.0
    EXIT_URGENCY_DIST: float = 100.0
    
    # Occlusion
    OCCLUSION_RANGE: float = 80.0
    FOV_RANGE: float = 150.0
    MIN_OCCLUSION_ANGLE: float = 5.0
    
    # Direction filtering: max heading difference (radians) to be "same direction"
    SAME_DIRECTION_THRESHOLD: float = np.pi / 2  # 90 degrees
    
    # Physical
    MASS_HV: float = 15000.0
    MASS_PC: float = 3000.0
    FPS: int = 25
    
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
        'unknown': '#95A5A6',
    })
    
    # Theme colors
    BG_DARK: str = '#0D1117'
    BG_PANEL: str = '#1A1A2E'
    SPINE_COLOR: str = '#4A4A6A'
    
    # Occlusion visualization colors
    OCCLUSION_FILL: str = 'gray'
    OCCLUSION_EDGE: str = 'darkgray'
    TANGENT_COLOR: str = '#E74C3C'
    FOV_COLOR: str = '#3498DB'
    HIGHLIGHT_COLOR: str = 'yellow'


# =============================================================================
# Role Classification
# =============================================================================

class RoleClassifier:
    """Classifies agent roles based on position, trajectory, and context."""
    
    def __init__(self, config: Config = None):
        self.config = config or Config()
        
    def classify_agent(self, agent: Dict, lane_info: Dict,
                       trajectory_history: Optional[pd.DataFrame] = None,
                       merge_point: Optional[float] = None,
                       exit_point: Optional[float] = None) -> Tuple[AgentRole, float, float]:
        """Classify agent role based on current state and context."""
        x, y = agent['x'], agent['y']
        vy = agent.get('vy', 0)
        
        lane_type = lane_info.get('lane_type', 'main')
        
        role = AgentRole.NORMAL_MAIN
        confidence = 0.5
        urgency = 0.0
        
        if lane_type == 'accel' or self._is_in_accel_lane(y, lane_info):
            role = AgentRole.MERGING
            confidence = 0.8
            
            if merge_point is not None:
                dist_to_merge_end = merge_point - x
                if dist_to_merge_end > 0:
                    urgency = np.clip(1.0 - dist_to_merge_end / self.config.MERGE_URGENCY_DIST, 0, 1)
                else:
                    urgency = 1.0
                    
            if vy < -self.config.LATERAL_VEL_THRESHOLD:
                confidence = 0.95
                
        elif self._is_exiting_behavior(agent, trajectory_history, exit_point):
            role = AgentRole.EXITING
            confidence = 0.75
            
            if exit_point is not None:
                dist_to_exit = exit_point - x
                if dist_to_exit > 0:
                    urgency = np.clip(1.0 - dist_to_exit / self.config.EXIT_URGENCY_DIST, 0, 1)
                else:
                    urgency = 1.0
                    
            if vy > self.config.LATERAL_VEL_THRESHOLD:
                confidence = 0.9
                
        elif self._is_yielding_behavior(agent, lane_info):
            role = AgentRole.YIELDING
            confidence = 0.7
            urgency = 0.0
            
        else:
            role = AgentRole.NORMAL_MAIN
            confidence = 0.85 if lane_type == 'main' else 0.6
            urgency = 0.0
            
        return role, confidence, urgency
    
    def _is_in_accel_lane(self, y: float, lane_info: Dict) -> bool:
        if 'accel_lane_y_bounds' in lane_info:
            y_min, y_max = lane_info['accel_lane_y_bounds']
            return y_min <= y <= y_max
        return False
    
    def _is_exiting_behavior(self, agent: Dict, history: Optional[pd.DataFrame],
                             exit_point: Optional[float]) -> bool:
        if history is None or len(history) < 10:
            return False
            
        if 'yVelocity' in history.columns:
            recent_lat_vel = history['yVelocity'].tail(10).mean()
            if recent_lat_vel > self.config.LATERAL_VEL_THRESHOLD:
                return True
                
        if 'yCenter' in history.columns:
            y_change = history['yCenter'].iloc[-1] - history['yCenter'].iloc[0]
            if y_change > self.config.LANE_CHANGE_Y_DELTA:
                return True
                
        return False
    
    def _is_yielding_behavior(self, agent: Dict, lane_info: Dict) -> bool:
        ax = agent.get('ax', 0)
        if ax < -1.0 and lane_info.get('is_merge_adjacent', False):
            return True
        return False


# =============================================================================
# Enhanced Occlusion Detection with Direction Filtering
# =============================================================================

class OcclusionDetector:
    """Detects occlusion relationships filtered by same traffic direction."""
    
    def __init__(self, config: Config = None):
        self.config = config or Config()
        
    def is_same_direction(self, agent1: Dict, agent2: Dict) -> bool:
        """Check if two agents are traveling in the same direction."""
        # Use velocity direction if available, otherwise heading
        if agent1.get('vx', 0) != 0 or agent1.get('vy', 0) != 0:
            dir1 = np.arctan2(agent1.get('vy', 0), agent1.get('vx', 0))
        else:
            dir1 = agent1.get('heading', 0)
            
        if agent2.get('vx', 0) != 0 or agent2.get('vy', 0) != 0:
            dir2 = np.arctan2(agent2.get('vy', 0), agent2.get('vx', 0))
        else:
            dir2 = agent2.get('heading', 0)
        
        # Normalize angle difference to [-pi, pi]
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
    
    def compute_tangent_points(self, observer: Dict, occluder: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """Compute tangent points from observer to occluder vehicle."""
        obs_x, obs_y = observer['x'], observer['y']
        corners = self.compute_vehicle_corners(occluder)
        
        angles = []
        for corner in corners:
            dx = corner[0] - obs_x
            dy = corner[1] - obs_y
            angle = np.arctan2(dy, dx)
            angles.append((angle, corner))
        
        angles.sort(key=lambda x: x[0])
        
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
        
        return left_pt, right_pt
    
    def compute_shadow_polygon(self, observer: Dict, occluder: Dict, 
                               fov_range: float = None) -> np.ndarray:
        """Compute the shadow polygon cast by occluder from observer's perspective."""
        if fov_range is None:
            fov_range = self.config.FOV_RANGE
            
        obs_x, obs_y = observer['x'], observer['y']
        left_pt, right_pt = self.compute_tangent_points(observer, occluder)
        
        left_dir = left_pt - np.array([obs_x, obs_y])
        right_dir = right_pt - np.array([obs_x, obs_y])
        
        left_norm = left_dir / (np.linalg.norm(left_dir) + 1e-6)
        right_norm = right_dir / (np.linalg.norm(right_dir) + 1e-6)
        
        left_far = np.array([obs_x, obs_y]) + left_norm * fov_range
        right_far = np.array([obs_x, obs_y]) + right_norm * fov_range
        
        polygon = np.array([
            right_pt,
            left_pt,
            left_far,
            right_far
        ])
        
        return polygon
    
    def compute_vehicle_shadow(self, observer: Dict, occluder: Dict) -> Tuple[float, float]:
        """Compute angular shadow cast by occluder from observer's perspective."""
        dx = occluder['x'] - observer['x']
        dy = occluder['y'] - observer['y']
        dist = np.sqrt(dx**2 + dy**2)
        
        if dist < 1.0:
            return (0, 0)
            
        center_angle = np.arctan2(dy, dx)
        occluder_heading = occluder.get('heading', 0)
        angle_to_occluder = center_angle - occluder_heading
        
        eff_width = abs(occluder['length'] * np.sin(angle_to_occluder)) + \
                   abs(occluder['width'] * np.cos(angle_to_occluder))
        
        angular_half_width = np.arctan2(eff_width / 2, dist)
        
        return (center_angle - angular_half_width, center_angle + angular_half_width)
    
    def check_occlusion(self, observer: Dict, target: Dict,
                        potential_occluders: List[Dict],
                        frame: int = 0) -> Tuple[OcclusionType, float, Optional[int], Optional[Dict]]:
        """Check if target is occluded from observer's view (same direction only)."""
        # Filter by same direction
        if not self.is_same_direction(observer, target):
            return (OcclusionType.NONE, 0.0, None, None)
        
        dx = target['x'] - observer['x']
        dy = target['y'] - observer['y']
        dist_to_target = np.sqrt(dx**2 + dy**2)
        
        if dist_to_target < 1.0 or dist_to_target > self.config.OCCLUSION_RANGE:
            return (OcclusionType.NONE, 0.0, None, None)
            
        target_angle = np.arctan2(dy, dx)
        target_heading = target.get('heading', 0)
        angle_diff = target_angle - target_heading
        target_eff_width = abs(target['length'] * np.sin(angle_diff)) + \
                          abs(target['width'] * np.cos(angle_diff))
        target_half_angle = np.arctan2(target_eff_width / 2, dist_to_target)
        target_angle_range = (target_angle - target_half_angle, target_angle + target_half_angle)
        
        max_occlusion = 0.0
        occluding_vehicle = None
        geometry_info = None
        
        for occluder in potential_occluders:
            if occluder['id'] == observer.get('id') or occluder['id'] == target.get('id'):
                continue
            
            # Occluder must also be same direction
            if not self.is_same_direction(observer, occluder):
                continue
                
            occ_dx = occluder['x'] - observer['x']
            occ_dy = occluder['y'] - observer['y']
            dist_to_occluder = np.sqrt(occ_dx**2 + occ_dy**2)
            
            # Occluder must be between observer and target
            if dist_to_occluder >= dist_to_target:
                continue
                
            shadow = self.compute_vehicle_shadow(observer, occluder)
            
            if shadow == (0, 0):
                continue
                
            overlap = self._angle_range_overlap(shadow, target_angle_range)
            target_span = target_angle_range[1] - target_angle_range[0]
            
            if target_span > 0:
                occlusion_ratio = overlap / target_span
            else:
                occlusion_ratio = 0.0
                
            if occlusion_ratio > max_occlusion:
                max_occlusion = occlusion_ratio
                occluding_vehicle = occluder['id']
                
                left_pt, right_pt = self.compute_tangent_points(observer, occluder)
                shadow_poly = self.compute_shadow_polygon(observer, occluder)
                geometry_info = {
                    'tangent_left': tuple(left_pt),
                    'tangent_right': tuple(right_pt),
                    'shadow_polygon': shadow_poly,
                    'occluder': occluder
                }
                
        if max_occlusion > 0.8:
            occ_type = OcclusionType.FULL
        elif max_occlusion > 0.2:
            occ_type = OcclusionType.PARTIAL
        else:
            occ_type = OcclusionType.NONE
            
        return (occ_type, max_occlusion, occluding_vehicle, geometry_info)
    
    def _angle_range_overlap(self, range1: Tuple[float, float],
                             range2: Tuple[float, float]) -> float:
        start = max(range1[0], range2[0])
        end = min(range1[1], range2[1])
        return max(0, end - start)
    
    def find_all_occlusions(self, agents: List[Dict], frame: int = 0) -> List[OcclusionEvent]:
        """Find all occlusion relationships (same direction only)."""
        occlusions = []
        
        trucks = [a for a in agents if a.get('class', '').lower() in self.config.HEAVY_VEHICLE_CLASSES]
        cars = [a for a in agents if a.get('class', '').lower() in self.config.CAR_CLASSES]
        all_vehicles = trucks + cars
        
        # Check all observer-target pairs where a truck could occlude
        for observer in all_vehicles:
            for target in all_vehicles:
                if observer['id'] == target['id']:
                    continue
                
                # Only trucks can be occluders
                occ_type, occ_ratio, occluder_id, geom = self.check_occlusion(
                    observer, target, trucks, frame
                )
                
                if occ_type != OcclusionType.NONE:
                    occluder = geom['occluder'] if geom else None
                    
                    event = OcclusionEvent(
                        frame=frame,
                        occluder_id=occluder_id,
                        occluded_id=target['id'],
                        blocked_id=observer['id'],
                        occlusion_type=occ_type,
                        occlusion_ratio=occ_ratio,
                        occluder_x=occluder['x'] if occluder else 0,
                        occluder_y=occluder['y'] if occluder else 0,
                        occluded_x=target['x'],
                        occluded_y=target['y'],
                        blocked_x=observer['x'],
                        blocked_y=observer['y'],
                    )
                    if geom:
                        event.tangent_left = geom['tangent_left']
                        event.tangent_right = geom['tangent_right']
                        event.shadow_polygon = geom['shadow_polygon']
                    occlusions.append(event)
                    
        return occlusions
    
    def find_occlusions_caused_by(self, ego_id: int, agents: List[Dict], 
                                   frame: int = 0) -> List[OcclusionEvent]:
        """Find occlusions where ego is the occluder (blocking others' view)."""
        occlusions = []
        
        ego = next((a for a in agents if a['id'] == ego_id), None)
        if ego is None:
            return occlusions
        
        # Get all other same-direction vehicles
        other_vehicles = [a for a in agents if a['id'] != ego_id and self.is_same_direction(ego, a)]
        
        # For each pair of other vehicles, check if ego blocks the view
        for observer in other_vehicles:
            for target in other_vehicles:
                if observer['id'] == target['id']:
                    continue
                
                # Check if ego occludes target from observer's view
                occ_type, occ_ratio, occluder_id, geom = self.check_occlusion(
                    observer, target, [ego], frame
                )
                
                if occ_type != OcclusionType.NONE and occluder_id == ego_id:
                    event = OcclusionEvent(
                        frame=frame,
                        occluder_id=ego_id,
                        occluded_id=target['id'],
                        blocked_id=observer['id'],
                        occlusion_type=occ_type,
                        occlusion_ratio=occ_ratio,
                        occluder_x=ego['x'],
                        occluder_y=ego['y'],
                        occluded_x=target['x'],
                        occluded_y=target['y'],
                        blocked_x=observer['x'],
                        blocked_y=observer['y'],
                    )
                    if geom:
                        event.tangent_left = geom['tangent_left']
                        event.tangent_right = geom['tangent_right']
                        event.shadow_polygon = geom['shadow_polygon']
                    occlusions.append(event)
                    
        return occlusions


# =============================================================================
# Data Loader
# =============================================================================

class ExiDRoleLoader:
    """Extended exiD loader with role classification and background support."""
    
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.config = Config()
        self.role_classifier = RoleClassifier(self.config)
        self.occlusion_detector = OcclusionDetector(self.config)
        
        self.tracks_df = None
        self.tracks_meta_df = None
        self.recording_meta = None
        self.recording_id = None
        
        self.background_image = None
        self.ortho_px_to_meter = 0.1
        
        self.lane_structure = {}
        
    def load_recording(self, recording_id: int) -> bool:
        """Load recording data including background."""
        prefix = f"{recording_id:02d}_"
        self.recording_id = recording_id
        
        try:
            self.tracks_df = pd.read_csv(self.data_dir / f"{prefix}tracks.csv")
            self.tracks_meta_df = pd.read_csv(self.data_dir / f"{prefix}tracksMeta.csv")
            
            rec_meta_path = self.data_dir / f"{prefix}recordingMeta.csv"
            if rec_meta_path.exists():
                rec_meta_df = pd.read_csv(rec_meta_path)
                if not rec_meta_df.empty:
                    self.recording_meta = rec_meta_df.iloc[0]
                    self.ortho_px_to_meter = float(
                        self.recording_meta.get('orthoPxToMeter', self.ortho_px_to_meter)
                    )
            
            bg_path = self.data_dir / f"{prefix}background.png"
            if bg_path.exists():
                self.background_image = plt.imread(str(bg_path))
                logger.info("Loaded lane layout background image.")
            
            self.tracks_df = self.tracks_df.merge(
                self.tracks_meta_df[['trackId', 'class', 'width', 'length']],
                on='trackId', how='left', suffixes=('', '_meta')
            )
            
            if 'width_meta' in self.tracks_df.columns:
                self.tracks_df['width'] = self.tracks_df['width'].fillna(self.tracks_df['width_meta'])
                self.tracks_df['length'] = self.tracks_df['length'].fillna(self.tracks_df['length_meta'])
            
            self._infer_lane_structure()
            
            logger.info(f"Loaded recording {recording_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading recording: {e}")
            return False
    
    def get_background_extent(self) -> Optional[List[float]]:
        """Get extent for plotting background image in meters."""
        if self.background_image is None:
            return None
        h, w = self.background_image.shape[:2]
        return [0, w * self.ortho_px_to_meter, -h * self.ortho_px_to_meter, 0]
    
    def get_full_data_extent(self) -> Tuple[float, float, float, float]:
        """Get the full extent of trajectory data."""
        if self.tracks_df is None:
            return (0, 100, -20, 20)
        
        x_min = self.tracks_df['xCenter'].min()
        x_max = self.tracks_df['xCenter'].max()
        y_min = self.tracks_df['yCenter'].min()
        y_max = self.tracks_df['yCenter'].max()
        
        pad_x = (x_max - x_min) * 0.05
        pad_y = (y_max - y_min) * 0.1
        
        return (x_min - pad_x, x_max + pad_x, y_min - pad_y, y_max + pad_y)
    
    def _infer_lane_structure(self):
        """Infer lane structure from trajectory data."""
        if self.tracks_df is None:
            return
            
        y_vals = self.tracks_df['yCenter'].dropna()
        
        if len(y_vals) == 0:
            return
            
        hist, bin_edges = np.histogram(y_vals, bins=50)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        try:
            from scipy.signal import find_peaks
            peaks, _ = find_peaks(hist, height=len(y_vals) * 0.01, distance=5)
            lane_centers = bin_centers[peaks]
        except ImportError:
            lane_centers = []
            for i in range(1, len(hist) - 1):
                if hist[i] > hist[i-1] and hist[i] > hist[i+1] and hist[i] > len(y_vals) * 0.01:
                    lane_centers.append(bin_centers[i])
            lane_centers = np.array(lane_centers)
        
        if len(lane_centers) >= 2:
            lane_width = np.median(np.diff(sorted(lane_centers)))
        else:
            lane_width = 3.5
            
        self.lane_structure = {
            'lane_centers': sorted(lane_centers) if len(lane_centers) > 0 else [],
            'lane_width': lane_width,
            'y_min': y_vals.min(),
            'y_max': y_vals.max()
        }
        
        if len(lane_centers) > 0:
            rightmost_y = max(lane_centers)
            self.lane_structure['accel_lane_y_bounds'] = (
                rightmost_y - lane_width/2,
                rightmost_y + lane_width * 1.5
            )
            
        logger.info(f"Inferred {len(lane_centers)} lanes, width={lane_width:.1f}m")
    
    def get_lane_info(self, x: float, y: float) -> Dict:
        """Get lane information for a position."""
        lane_info = {
            'lane_id': None,
            'lane_type': 'main',
            'is_merge_adjacent': False
        }
        
        if not self.lane_structure:
            return lane_info
            
        lane_centers = self.lane_structure.get('lane_centers', [])
        lane_width = self.lane_structure.get('lane_width', 3.5)
        
        if not lane_centers:
            return lane_info
            
        distances = [abs(y - lc) for lc in lane_centers]
        closest_idx = np.argmin(distances)
        
        if distances[closest_idx] < lane_width:
            lane_info['lane_id'] = closest_idx
            
        accel_bounds = self.lane_structure.get('accel_lane_y_bounds')
        if accel_bounds and accel_bounds[0] <= y <= accel_bounds[1]:
            if y > max(lane_centers) + lane_width/2:
                lane_info['lane_type'] = 'accel'
                
        if closest_idx == len(lane_centers) - 1:
            lane_info['is_merge_adjacent'] = True
            
        return lane_info
    
    def get_merge_exit_points(self) -> Tuple[Optional[float], Optional[float]]:
        """Estimate merge and exit point locations."""
        if self.tracks_df is None:
            return None, None
            
        x_min = self.tracks_df['xCenter'].min()
        x_max = self.tracks_df['xCenter'].max()
        
        merge_point = x_min + 0.3 * (x_max - x_min)
        exit_point = x_min + 0.7 * (x_max - x_min)
        
        return merge_point, exit_point
    
    def classify_frame_agents(self, frame: int) -> List[AgentState]:
        """Classify all agents in a frame."""
        frame_data = self.tracks_df[self.tracks_df['frame'] == frame]
        
        if frame_data.empty:
            return []
            
        merge_point, exit_point = self.get_merge_exit_points()
        agents = []
        
        for _, row in frame_data.iterrows():
            vclass = str(row.get('class', 'car')).lower()
            
            agent_dict = {
                'id': int(row['trackId']),
                'x': float(row['xCenter']),
                'y': float(row['yCenter']),
                'vx': float(row.get('xVelocity', 0)),
                'vy': float(row.get('yVelocity', 0)),
                'ax': float(row.get('xAcceleration', 0)),
                'ay': float(row.get('yAcceleration', 0)),
                'heading': np.radians(float(row.get('heading', 0))),
                'speed': np.sqrt(row.get('xVelocity', 0)**2 + row.get('yVelocity', 0)**2),
                'width': float(row.get('width', 2.0)),
                'length': float(row.get('length', 5.0)),
                'class': vclass,
            }
            
            track_history = self.tracks_df[
                (self.tracks_df['trackId'] == agent_dict['id']) &
                (self.tracks_df['frame'] <= frame) &
                (self.tracks_df['frame'] >= frame - 50)
            ].sort_values('frame')
            
            lane_info = self.get_lane_info(agent_dict['x'], agent_dict['y'])
            
            role, confidence, urgency = self.role_classifier.classify_agent(
                agent_dict, lane_info, track_history, merge_point, exit_point
            )
            
            mass = self.config.MASS_HV if vclass in self.config.HEAVY_VEHICLE_CLASSES else self.config.MASS_PC
            
            agent_state = AgentState(
                id=agent_dict['id'],
                x=agent_dict['x'],
                y=agent_dict['y'],
                vx=agent_dict['vx'],
                vy=agent_dict['vy'],
                ax=agent_dict['ax'],
                ay=agent_dict['ay'],
                heading=agent_dict['heading'],
                speed=agent_dict['speed'],
                length=agent_dict['length'],
                width=agent_dict['width'],
                vehicle_class=vclass,
                mass=mass,
                lane_id=lane_info.get('lane_id'),
                role=role,
                role_confidence=confidence,
                urgency=urgency
            )
            
            agents.append(agent_state)
            
        return agents
    
    def get_heavy_vehicles(self) -> List[int]:
        """Get list of heavy vehicle IDs."""
        mask = self.tracks_meta_df['class'].str.lower().isin(self.config.HEAVY_VEHICLE_CLASSES)
        return self.tracks_meta_df[mask]['trackId'].tolist()
    
    def find_best_interaction_frame(self, ego_id: int) -> Optional[int]:
        """Find frame with most surrounding vehicles for ego."""
        ego_data = self.tracks_df[self.tracks_df['trackId'] == ego_id]
        if ego_data.empty:
            return None
            
        frames = ego_data['frame'].values
        best_frame = None
        best_count = -1
        
        for frame in frames[::10]:
            frame_data = self.tracks_df[self.tracks_df['frame'] == frame]
            ego_row = frame_data[frame_data['trackId'] == ego_id]
            
            if ego_row.empty:
                continue
                
            ego_x = ego_row.iloc[0]['xCenter']
            ego_y = ego_row.iloc[0]['yCenter']
            
            count = 0
            for _, row in frame_data.iterrows():
                if row['trackId'] == ego_id:
                    continue
                dx = row['xCenter'] - ego_x
                dy = row['yCenter'] - ego_y
                if (-self.config.OBS_RANGE_BEHIND <= dx <= self.config.OBS_RANGE_AHEAD and
                    -self.config.OBS_RANGE_RIGHT <= dy <= self.config.OBS_RANGE_LEFT):
                    count += 1
                    
            if count > best_count:
                best_count = count
                best_frame = frame
                
        if best_frame is None and len(frames) > 0:
            best_frame = int(np.median(frames))
            
        return best_frame


# =============================================================================
# Occlusion Logger (CSV Export)
# =============================================================================

class OcclusionLogger:
    """Logs occlusion events to CSV for training and analysis."""
    
    def __init__(self, recording_id: int):
        self.recording_id = recording_id
        self.events: List[Dict] = []
        
    def add_event(self, event: OcclusionEvent):
        """Add an occlusion event to the log."""
        self.events.append({
            'recording_id': self.recording_id,
            'frame': event.frame,
            'occluder_id': event.occluder_id,
            'occluded_id': event.occluded_id,
            'blocked_id': event.blocked_id,
            'occlusion_type': event.occlusion_type.value,
            'occlusion_ratio': round(event.occlusion_ratio, 4),
            'occluder_x': round(event.occluder_x, 2),
            'occluder_y': round(event.occluder_y, 2),
            'occluded_x': round(event.occluded_x, 2),
            'occluded_y': round(event.occluded_y, 2),
            'blocked_x': round(event.blocked_x, 2),
            'blocked_y': round(event.blocked_y, 2),
        })
    
    def add_events(self, events: List[OcclusionEvent]):
        """Add multiple occlusion events."""
        for event in events:
            self.add_event(event)
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert to DataFrame."""
        return pd.DataFrame(self.events)
    
    def save_csv(self, output_path: Path):
        """Save to CSV file."""
        df = self.to_dataframe()
        df.to_csv(output_path, index=False)
        logger.info(f"Saved occlusion log: {output_path} ({len(df)} events)")
        return df


# =============================================================================
# Visualization (Unified Style)
# =============================================================================

class RoleOcclusionVisualizer:
    """Visualize agent roles and occlusions with consistent style."""
    
    def __init__(self, config: Config = None, loader: ExiDRoleLoader = None):
        self.config = config or Config()
        self.loader = loader
        
    def _get_rotated_rect(self, cx, cy, length, width, heading):
        """Get corners of rotated rectangle."""
        half_l, half_w = length/2, width/2
        
        corners = np.array([
            [-half_l, -half_w],
            [half_l, -half_w],
            [half_l, half_w],
            [-half_l, half_w]
        ])
        
        cos_h = np.cos(heading)
        sin_h = np.sin(heading)
        R = np.array([[cos_h, -sin_h], [sin_h, cos_h]])
        
        return corners @ R.T + np.array([cx, cy])
    
    def _draw_vehicle(self, ax, agent: AgentState, is_ego: bool = False, 
                      show_role: bool = True, xlim: Tuple[float, float] = None,
                      ylim: Tuple[float, float] = None):
        """Draw a vehicle with role-based coloring and proper label placement."""
        is_truck = agent.vehicle_class in self.config.HEAVY_VEHICLE_CLASSES
        
        if is_ego:
            color = self.config.COLORS.get(agent.vehicle_class, '#E74C3C')
            edgecolor = 'yellow'
            linewidth = 3
            alpha = 1.0
        elif is_truck:
            color = self.config.COLORS.get(agent.vehicle_class, '#E74C3C')
            edgecolor = 'white'
            linewidth = 2
            alpha = 0.9
        else:
            color = self.config.ROLE_COLORS.get(agent.role.value, '#3498DB')
            edgecolor = 'white'
            linewidth = 1.5
            alpha = 0.85
        
        corners = self._get_rotated_rect(
            agent.x, agent.y, agent.length, agent.width, agent.heading
        )
        
        rect = plt.Polygon(corners, closed=True, facecolor=color,
                          edgecolor=edgecolor, linewidth=linewidth, 
                          alpha=alpha, zorder=4)
        ax.add_patch(rect)
        
        # Build label
        label_parts = []
        if is_ego:
            label_parts.append("EGO")
        else:
            label_parts.append(f"{agent.id}")
            
        if show_role and not is_truck and agent.role != AgentRole.UNKNOWN:
            role_short = {
                AgentRole.NORMAL_MAIN: "N",
                AgentRole.MERGING: "M",
                AgentRole.EXITING: "E",
                AgentRole.YIELDING: "Y"
            }.get(agent.role, "?")
            if agent.urgency > 0.3:
                label_parts.append(f"{role_short}:{agent.urgency:.1f}")
            else:
                label_parts.append(role_short)
        
        label = " ".join(label_parts)
        text_color = 'yellow' if is_ego else 'white'
        fontsize = 8 if is_ego else 7
        
        # Compute label position - ensure it stays within plot limits
        label_x = agent.x
        label_y = agent.y + agent.width/2 + 1.2
        
        # Adjust if outside limits
        if xlim and ylim:
            margin_x = (xlim[1] - xlim[0]) * 0.02
            margin_y = (ylim[1] - ylim[0]) * 0.02
            
            # Keep label inside bounds
            label_x = np.clip(label_x, xlim[0] + margin_x, xlim[1] - margin_x)
            label_y = np.clip(label_y, ylim[0] + margin_y, ylim[1] - margin_y)
            
            # If vehicle is near top edge, put label below
            if agent.y > ylim[1] - (ylim[1] - ylim[0]) * 0.15:
                label_y = agent.y - agent.width/2 - 1.2
        
        ax.text(label_x, label_y, label,
               ha='center', va='bottom', fontsize=fontsize,
               color=text_color, fontweight='bold' if is_ego else 'normal',
               bbox=dict(boxstyle='round,pad=0.15', facecolor='black', alpha=0.5),
               zorder=6)
    
    def _compute_plot_limits(self, agents: List[AgentState], 
                             use_full_extent: bool = True,
                             ego_id: Optional[int] = None) -> Tuple[float, float, float, float]:
        """Compute consistent plot limits."""
        bg_extent = self.loader.get_background_extent() if self.loader else None
        
        if use_full_extent and bg_extent:
            margin = 5
            x_min, x_max = bg_extent[0] - margin, bg_extent[1] + margin
            y_min, y_max = bg_extent[2] - margin, bg_extent[3] + margin
        elif use_full_extent and self.loader:
            x_min, x_max, y_min, y_max = self.loader.get_full_data_extent()
        else:
            all_x = [a.x for a in agents]
            all_y = [a.y for a in agents]
            padding_x = 30
            padding_y = 15
            x_min = min(all_x) - padding_x
            x_max = max(all_x) + padding_x
            y_min = min(all_y) - padding_y
            y_max = max(all_y) + padding_y
        
        return (x_min, x_max, y_min, y_max)
    
    def _draw_background_and_lanes(self, ax, xlim, ylim):
        """Draw background image and lane markings."""
        bg_extent = self.loader.get_background_extent() if self.loader else None
        
        if bg_extent and self.loader.background_image is not None:
            ax.imshow(self.loader.background_image, extent=bg_extent, 
                     alpha=0.6, aspect='equal', zorder=0)
        else:
            # Draw lane markings
            lane_centers = self.loader.lane_structure.get('lane_centers', []) if self.loader else []
            for lc in lane_centers:
                ax.axhline(lc, color='white', linestyle='--', alpha=0.4, linewidth=1, zorder=1)
            
            # Road boundaries
            if self.loader and self.loader.lane_structure:
                y_road_min = self.loader.lane_structure.get('y_min', ylim[0])
                y_road_max = self.loader.lane_structure.get('y_max', ylim[1])
                ax.axhline(y_road_min - 2, color='gray', linestyle='-', linewidth=3, zorder=1)
                ax.axhline(y_road_max + 2, color='gray', linestyle='-', linewidth=3, zorder=1)
    
    def _draw_occlusion_shadows(self, ax, occlusions: List[OcclusionEvent], 
                                agents: List[AgentState], show_polygons: bool = True):
        """Draw occlusion shadows and tangent lines."""
        for occ in occlusions:
            occluder = next((a for a in agents if a.id == occ.occluder_id), None)
            blocked = next((a for a in agents if a.id == occ.blocked_id), None)
            occluded = next((a for a in agents if a.id == occ.occluded_id), None)
            
            if not all([occluder, blocked, occluded]):
                continue
            
            shadow_color = '#E74C3C' if occ.occlusion_type == OcclusionType.FULL else '#F39C12'
            
            # Draw shadow polygon if available
            if show_polygons and occ.shadow_polygon is not None:
                shadow_patch = plt.Polygon(
                    occ.shadow_polygon, closed=True,
                    facecolor=self.config.OCCLUSION_FILL, alpha=0.2,
                    edgecolor=self.config.OCCLUSION_EDGE, linewidth=1.5, linestyle='--',
                    zorder=2
                )
                ax.add_patch(shadow_patch)
            
            # Draw tangent lines
            if occ.tangent_left and occ.tangent_right:
                ax.plot([blocked.x, occ.tangent_left[0]], 
                       [blocked.y, occ.tangent_left[1]],
                       color=self.config.TANGENT_COLOR, linestyle='--', 
                       linewidth=1.5, alpha=0.6, zorder=3)
                ax.plot([blocked.x, occ.tangent_right[0]], 
                       [blocked.y, occ.tangent_right[1]],
                       color=self.config.TANGENT_COLOR, linestyle='--', 
                       linewidth=1.5, alpha=0.6, zorder=3)
            else:
                # Fallback: simple connecting lines
                ax.plot([blocked.x, occluder.x], [blocked.y, occluder.y],
                       color=shadow_color, linestyle=':', alpha=0.5, linewidth=1.5, zorder=2)
                ax.plot([occluder.x, occluded.x], [occluder.y, occluded.y],
                       color=shadow_color, linestyle=':', alpha=0.5, linewidth=1.5, zorder=2)
            
            # Highlight occluded vehicle
            radius = np.hypot(occluded.length/2, occluded.width/2) + 1.0
            highlight = Circle(
                (occluded.x, occluded.y),
                radius=radius,
                edgecolor=self.config.HIGHLIGHT_COLOR,
                linewidth=2,
                linestyle='--',
                fill=False,
                alpha=0.8,
                zorder=7
            )
            ax.add_patch(highlight)
    
    def _add_legend(self, ax, include_occlusion: bool = False):
        """Add legend to plot."""
        legend_elements = [
            mpatches.Patch(facecolor=self.config.ROLE_COLORS['normal_main'], 
                          edgecolor='white', label='Normal'),
            mpatches.Patch(facecolor=self.config.ROLE_COLORS['merging'], 
                          edgecolor='white', label='Merging'),
            mpatches.Patch(facecolor=self.config.ROLE_COLORS['exiting'], 
                          edgecolor='white', label='Exiting'),
            mpatches.Patch(facecolor=self.config.ROLE_COLORS['yielding'], 
                          edgecolor='white', label='Yielding'),
            mpatches.Patch(facecolor='#E74C3C', edgecolor='yellow', 
                          linewidth=2, label='Truck/Ego'),
        ]
        
        if include_occlusion:
            legend_elements.extend([
                plt.Line2D([0], [0], color=self.config.TANGENT_COLOR, linestyle='--', 
                          linewidth=1.5, label='Tangent Lines'),
                mpatches.Patch(facecolor=self.config.OCCLUSION_FILL, alpha=0.25,
                              edgecolor=self.config.OCCLUSION_EDGE, linestyle='--',
                              label='Occluded Region'),
                Circle((0, 0), radius=1, edgecolor=self.config.HIGHLIGHT_COLOR,
                      linewidth=2, linestyle='--', fill=False, label='Occluded Vehicle'),
            ])
        
        ax.legend(handles=legend_elements, loc='upper left', fontsize=7,
                 facecolor=self.config.BG_PANEL, edgecolor='white', 
                 labelcolor='white')
    
    def _finalize_plot(self, ax, xlim, ylim, title: str):
        """Finalize plot with consistent styling."""
        ax.set_xlim(xlim[0], xlim[1])
        ax.set_ylim(ylim[0], ylim[1])
        ax.set_xlabel('X (m)', color='white')
        ax.set_ylabel('Y (m)', color='white')
        ax.set_title(title, fontsize=11, fontweight='bold', color='white')
        ax.tick_params(colors='white')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.2)
        
        for spine in ax.spines.values():
            spine.set_color(self.config.SPINE_COLOR)
    
    def plot_traffic_snapshot(self, ax, agents: List[AgentState],
                              occlusions: List[OcclusionEvent],
                              ego_id: Optional[int] = None,
                              use_full_extent: bool = True,
                              title: str = None,
                              show_occlusion_shadows: bool = True):
        """
        Plot traffic snapshot with full road layout visible.
        Same style used for both traffic and ego-occlusion views.
        """
        ax.set_facecolor(self.config.BG_PANEL)
        
        # Compute limits
        xlim_ylim = self._compute_plot_limits(agents, use_full_extent, ego_id)
        xlim = (xlim_ylim[0], xlim_ylim[1])
        ylim = (xlim_ylim[2], xlim_ylim[3])
        
        # Draw background
        self._draw_background_and_lanes(ax, xlim, ylim)
        
        # Draw occlusion shadows
        if show_occlusion_shadows and occlusions:
            self._draw_occlusion_shadows(ax, occlusions, agents, show_polygons=True)
        
        # Draw vehicles
        for agent in agents:
            is_ego = (agent.id == ego_id)
            self._draw_vehicle(ax, agent, is_ego=is_ego, show_role=True, 
                             xlim=xlim, ylim=ylim)
            
            # Velocity arrow
            arrow_scale = 0.5
            arrow_color = 'yellow' if is_ego else 'cyan'
            head_width = 1 if is_ego else 0.8
            head_length = 0.5 if is_ego else 0.4
            alpha_val = 1.0 if is_ego else 0.7
            
            ax.arrow(agent.x, agent.y, 
                    agent.vx * arrow_scale, agent.vy * arrow_scale,
                    head_width=head_width, head_length=head_length,
                    fc=arrow_color, ec=arrow_color, alpha=alpha_val, zorder=5)
        
        # Add legend
        self._add_legend(ax, include_occlusion=show_occlusion_shadows and len(occlusions) > 0)
        
        # Finalize
        self._finalize_plot(ax, xlim, ylim, title or 'Traffic Snapshot with Roles & Occlusions')
    
    def plot_ego_occlusion_view(self, ax, agents: List[AgentState],
                                ego_occlusions: List[OcclusionEvent],
                                ego_id: int,
                                use_full_extent: bool = True,
                                title: str = None):
        """
        Plot occlusions CAUSED BY the ego truck.
        Uses the same style as traffic snapshot for consistency.
        """
        ax.set_facecolor(self.config.BG_PANEL)
        
        # Compute limits - same as traffic snapshot
        xlim_ylim = self._compute_plot_limits(agents, use_full_extent, ego_id)
        xlim = (xlim_ylim[0], xlim_ylim[1])
        ylim = (xlim_ylim[2], xlim_ylim[3])
        
        # Draw background
        self._draw_background_and_lanes(ax, xlim, ylim)
        
        # Draw occlusion shadows caused by ego
        if ego_occlusions:
            self._draw_occlusion_shadows(ax, ego_occlusions, agents, show_polygons=True)
        
        # Draw vehicles
        for agent in agents:
            is_ego = (agent.id == ego_id)
            self._draw_vehicle(ax, agent, is_ego=is_ego, show_role=True,
                             xlim=xlim, ylim=ylim)
            
            # Velocity arrow
            arrow_scale = 0.5
            arrow_color = 'yellow' if is_ego else 'cyan'
            head_width = 1 if is_ego else 0.8
            head_length = 0.5 if is_ego else 0.4
            alpha_val = 1.0 if is_ego else 0.7
            
            ax.arrow(agent.x, agent.y, 
                    agent.vx * arrow_scale, agent.vy * arrow_scale,
                    head_width=head_width, head_length=head_length,
                    fc=arrow_color, ec=arrow_color, alpha=alpha_val, zorder=5)
        
        # Mark ego as the occluder with special indicator
        ego_agent = next((a for a in agents if a.id == ego_id), None)
        if ego_agent:
            # Draw "OCCLUDER" label
            ax.text(ego_agent.x, ego_agent.y - ego_agent.width/2 - 2.5, 
                   "OCCLUDER",
                   ha='center', va='top', fontsize=8, fontweight='bold',
                   color='red', 
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8),
                   zorder=10)
        
        # Add legend
        self._add_legend(ax, include_occlusion=len(ego_occlusions) > 0)
        
        # Add occlusion count info
        n_occ = len(ego_occlusions)
        n_full = sum(1 for o in ego_occlusions if o.occlusion_type == OcclusionType.FULL)
        n_partial = n_occ - n_full
        info_text = f"Ego causes {n_occ} occlusions ({n_full} full, {n_partial} partial)"
        ax.text(0.98, 0.02, info_text, transform=ax.transAxes,
               ha='right', va='bottom', fontsize=9, color='white',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.6))
        
        # Finalize
        title = title or f'Occlusions Caused by Ego Truck (ID: {ego_id})'
        self._finalize_plot(ax, xlim, ylim, title)
    
    def plot_role_distribution(self, ax, agents: List[AgentState]):
        """Plot role distribution bar chart."""
        ax.set_facecolor(self.config.BG_PANEL)
        
        role_counts = defaultdict(int)
        for agent in agents:
            role_counts[agent.role.value] += 1
        
        roles = list(role_counts.keys())
        counts = [role_counts[r] for r in roles]
        colors = [self.config.ROLE_COLORS.get(r, '#95A5A6') for r in roles]
        
        bars = ax.bar(roles, counts, color=colors, edgecolor='white', alpha=0.8)
        
        for bar, count in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                   str(count), ha='center', va='bottom', color='white', fontsize=10)
        
        ax.set_ylabel('Count', color='white')
        ax.set_title('Role Distribution', fontsize=11, fontweight='bold', color='white')
        ax.tick_params(colors='white')
        ax.set_xticklabels([r.replace('_', '\n') for r in roles], rotation=0, fontsize=8)
        ax.grid(True, alpha=0.2, axis='y')
        
        for spine in ax.spines.values():
            spine.set_color(self.config.SPINE_COLOR)
    
    def plot_occlusion_table(self, ax, occlusions: List[OcclusionEvent], title: str = None):
        """Plot occlusion details as table."""
        ax.set_facecolor(self.config.BG_PANEL)
        ax.axis('off')
        
        if not occlusions:
            ax.text(0.5, 0.5, 'No Occlusions Detected', ha='center', va='center',
                   color='white', fontsize=12)
            ax.set_title(title or 'Occlusion Details', fontsize=11, fontweight='bold', 
                        color='white', pad=20)
            return
        
        headers = ['Frame', 'Occluder', 'Blocks', 'From', 'Type', 'Ratio']
        rows = []
        
        for occ in occlusions[:12]:  # Limit rows
            type_str = 'FULL' if occ.occlusion_type == OcclusionType.FULL else 'PARTIAL'
            rows.append([
                str(occ.frame),
                str(occ.occluder_id),
                str(occ.occluded_id),
                str(occ.blocked_id),
                type_str,
                f'{occ.occlusion_ratio:.0%}'
            ])
        
        if rows:
            table = ax.table(
                cellText=rows,
                colLabels=headers,
                loc='center',
                cellLoc='center',
                colColours=['#2C3E50'] * len(headers),
                cellColours=[['#34495E'] * len(headers)] * len(rows)
            )
            
            table.auto_set_font_size(False)
            table.set_fontsize(8)
            table.scale(1.0, 1.3)
            
            for key, cell in table.get_celld().items():
                cell.set_text_props(color='white')
                cell.set_edgecolor(self.config.SPINE_COLOR)
        
        ax.set_title(title or 'Occlusion Details', fontsize=11, fontweight='bold', 
                    color='white', pad=20)
    
    def plot_summary(self, ax, agents: List[AgentState], 
                     occlusions: List[OcclusionEvent],
                     ego_id: Optional[int], frame: int):
        """Plot summary statistics."""
        ax.set_facecolor(self.config.BG_PANEL)
        ax.axis('off')
        
        n_trucks = sum(1 for a in agents if a.vehicle_class in self.config.HEAVY_VEHICLE_CLASSES)
        n_cars = sum(1 for a in agents if a.vehicle_class in self.config.CAR_CLASSES)
        n_merging = sum(1 for a in agents if a.role == AgentRole.MERGING)
        n_exiting = sum(1 for a in agents if a.role == AgentRole.EXITING)
        n_yielding = sum(1 for a in agents if a.role == AgentRole.YIELDING)
        
        n_full_occ = sum(1 for o in occlusions if o.occlusion_type == OcclusionType.FULL)
        n_partial_occ = sum(1 for o in occlusions if o.occlusion_type == OcclusionType.PARTIAL)
        
        max_urgency = max((a.urgency for a in agents), default=0)
        avg_speed = np.mean([a.speed for a in agents]) * 3.6 if agents else 0
        
        ego_info = "Not selected"
        if ego_id is not None:
            ego_agent = next((a for a in agents if a.id == ego_id), None)
            if ego_agent:
                ego_info = f"{ego_agent.vehicle_class.title()} | {ego_agent.speed*3.6:.1f} km/h"
        
        summary_lines = [
            "ANALYSIS SUMMARY",
            f"Frame: {frame}",
            f"Ego: {ego_info}",
            "",
            "Vehicle Counts",
            f"  Trucks:     {n_trucks}",
            f"  Cars:       {n_cars}",
            f"  Total:      {len(agents)}",
            "",
            "Role Breakdown",
            f"  Merging:    {n_merging}",
            f"  Exiting:    {n_exiting}",
            f"  Yielding:   {n_yielding}",
            "",
            "Occlusions (same dir)",
            f"  Full:       {n_full_occ}",
            f"  Partial:    {n_partial_occ}",
            f"  Total:      {len(occlusions)}",
            "",
            "Dynamics",
            f"  Avg Speed:  {avg_speed:.1f} km/h",
            f"  Max Urgency: {max_urgency:.2f}",
        ]
        
        summary = "\n".join(summary_lines)
        
        ax.text(0.05, 0.95, summary, transform=ax.transAxes,
               fontsize=9, color='white', family='monospace',
               verticalalignment='top')
        
        for spine in ax.spines.values():
            spine.set_color(self.config.SPINE_COLOR)


# =============================================================================
# Output Manager
# =============================================================================

class OutputManager:
    """Manages organized output folder structure."""
    
    def __init__(self, base_dir: str):
        self.base_dir = Path(base_dir)
        
    def create_analysis_folder(self, recording_id: int, ego_id: Optional[int], 
                               frame: int) -> Path:
        """Create organized folder for analysis outputs."""
        folder_name = f"rec{recording_id:02d}_ego{ego_id or 'none'}_frame{frame}"
        folder_path = self.base_dir / folder_name
        folder_path.mkdir(parents=True, exist_ok=True)
        return folder_path
    
    def get_plot_paths(self, folder: Path) -> Dict[str, Path]:
        """Get paths for all output plots."""
        return {
            'traffic_snapshot': folder / '01_traffic_snapshot.png',
            'ego_occlusion': folder / '02_ego_occlusion.png',
            'role_distribution': folder / '03_role_distribution.png',
            'occlusion_table': folder / '04_occlusion_table.png',
            'summary': folder / '05_summary.png',
            'combined': folder / '06_combined.png',
            'animation': folder / 'animation.gif',
            'occlusion_csv': folder / 'occlusion_log.csv',
        }
    
    def save_metadata(self, folder: Path, metadata: Dict):
        """Save analysis metadata as JSON."""
        with open(folder / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2, default=str)


# =============================================================================
# Main Analysis Pipeline
# =============================================================================

def analyze_recording(data_dir: str, recording_id: int,
                      ego_id: Optional[int] = None,
                      frame: Optional[int] = None,
                      output_dir: str = './output_roles',
                      create_animation: bool = True,
                      animation_frames: int = 60,
                      animation_step: int = 2,
                      log_all_frames: bool = False) -> Dict:
    """
    Analyze a recording for roles and occlusions with organized outputs.
    
    Args:
        log_all_frames: If True, log occlusions for entire recording to CSV.
    """
    output_mgr = OutputManager(output_dir)
    
    logger.info("=" * 60)
    logger.info("Role & Occlusion Analysis (v3)")
    logger.info("  - Same direction filtering")
    logger.info("  - CSV occlusion logging")
    logger.info("  - Ego-as-occluder view")
    logger.info("=" * 60)
    
    # Load data
    loader = ExiDRoleLoader(data_dir)
    if not loader.load_recording(recording_id):
        return {}
    
    # Find ego vehicle
    if ego_id is None:
        heavy_ids = loader.get_heavy_vehicles()
        if heavy_ids:
            ego_id = heavy_ids[0]
            logger.info(f"Auto-selected ego (truck): {ego_id}")
    
    # Find best frame
    if frame is None and ego_id is not None:
        frame = loader.find_best_interaction_frame(ego_id)
        if frame is None:
            frames = loader.tracks_df['frame'].unique()
            frame = int(np.median(frames))
        logger.info(f"Auto-selected frame: {frame}")
    elif frame is None:
        frames = loader.tracks_df['frame'].unique()
        frame = int(np.median(frames))
    
    # Create output folder
    output_folder = output_mgr.create_analysis_folder(recording_id, ego_id, frame)
    paths = output_mgr.get_plot_paths(output_folder)
    logger.info(f"Output folder: {output_folder}")
    
    # Classify agents
    agents = loader.classify_frame_agents(frame)
    logger.info(f"Classified {len(agents)} agents")
    
    # Find occlusions (same direction only)
    agent_dicts = [
        {'id': a.id, 'x': a.x, 'y': a.y, 'heading': a.heading,
         'vx': a.vx, 'vy': a.vy,
         'width': a.width, 'length': a.length, 'class': a.vehicle_class}
        for a in agents
    ]
    all_occlusions = loader.occlusion_detector.find_all_occlusions(agent_dicts, frame)
    logger.info(f"Found {len(all_occlusions)} occlusions (same direction)")
    
    # Find occlusions caused by ego
    ego_occlusions = []
    if ego_id is not None:
        ego_occlusions = loader.occlusion_detector.find_occlusions_caused_by(
            ego_id, agent_dicts, frame
        )
        logger.info(f"Ego truck causes {len(ego_occlusions)} occlusions")
    
    # Create visualizer
    viz = RoleOcclusionVisualizer(loader=loader)
    
    # 1. Traffic Snapshot
    fig1, ax1 = plt.subplots(figsize=(16, 9))
    fig1.patch.set_facecolor(viz.config.BG_DARK)
    viz.plot_traffic_snapshot(ax1, agents, all_occlusions, ego_id, use_full_extent=True)
    fig1.tight_layout()
    fig1.savefig(paths['traffic_snapshot'], dpi=150, bbox_inches='tight',
                facecolor=fig1.get_facecolor())
    plt.close(fig1)
    logger.info(f"Saved: {paths['traffic_snapshot'].name}")
    
    # 2. Ego Occlusion View (occlusions CAUSED BY ego)
    if ego_id is not None:
        fig2, ax2 = plt.subplots(figsize=(16, 9))
        fig2.patch.set_facecolor(viz.config.BG_DARK)
        viz.plot_ego_occlusion_view(ax2, agents, ego_occlusions, ego_id, use_full_extent=True)
        fig2.tight_layout()
        fig2.savefig(paths['ego_occlusion'], dpi=150, bbox_inches='tight',
                    facecolor=fig2.get_facecolor())
        plt.close(fig2)
        logger.info(f"Saved: {paths['ego_occlusion'].name}")
    
    # 3. Role Distribution
    fig3, ax3 = plt.subplots(figsize=(8, 6))
    fig3.patch.set_facecolor(viz.config.BG_DARK)
    viz.plot_role_distribution(ax3, agents)
    fig3.tight_layout()
    fig3.savefig(paths['role_distribution'], dpi=150, bbox_inches='tight',
                facecolor=fig3.get_facecolor())
    plt.close(fig3)
    logger.info(f"Saved: {paths['role_distribution'].name}")
    
    # 4. Occlusion Table
    fig4, ax4 = plt.subplots(figsize=(10, 6))
    fig4.patch.set_facecolor(viz.config.BG_DARK)
    viz.plot_occlusion_table(ax4, all_occlusions, title="All Occlusions (Same Direction)")
    fig4.tight_layout()
    fig4.savefig(paths['occlusion_table'], dpi=150, bbox_inches='tight',
                facecolor=fig4.get_facecolor())
    plt.close(fig4)
    logger.info(f"Saved: {paths['occlusion_table'].name}")
    
    # 5. Summary
    fig5, ax5 = plt.subplots(figsize=(6, 8))
    fig5.patch.set_facecolor(viz.config.BG_DARK)
    viz.plot_summary(ax5, agents, all_occlusions, ego_id, frame)
    fig5.tight_layout()
    fig5.savefig(paths['summary'], dpi=150, bbox_inches='tight',
                facecolor=fig5.get_facecolor())
    plt.close(fig5)
    logger.info(f"Saved: {paths['summary'].name}")
    
    # 6. Combined Figure
    fig_comb = plt.figure(figsize=(20, 14))
    fig_comb.patch.set_facecolor(viz.config.BG_DARK)
    gs = fig_comb.add_gridspec(2, 3, hspace=0.3, wspace=0.25)
    
    ax_c1 = fig_comb.add_subplot(gs[0, :2])
    ax_c2 = fig_comb.add_subplot(gs[0, 2])
    ax_c3 = fig_comb.add_subplot(gs[1, 0])
    ax_c4 = fig_comb.add_subplot(gs[1, 1])
    ax_c5 = fig_comb.add_subplot(gs[1, 2])
    
    viz.plot_traffic_snapshot(ax_c1, agents, all_occlusions, ego_id, use_full_extent=True)
    viz.plot_role_distribution(ax_c2, agents)
    if ego_id is not None:
        viz.plot_ego_occlusion_view(ax_c3, agents, ego_occlusions, ego_id, use_full_extent=True)
    else:
        ax_c3.set_facecolor(viz.config.BG_PANEL)
        ax_c3.text(0.5, 0.5, 'No Ego Selected', ha='center', va='center', color='white')
    viz.plot_occlusion_table(ax_c4, all_occlusions)
    viz.plot_summary(ax_c5, agents, all_occlusions, ego_id, frame)
    
    fig_comb.suptitle(
        f"Role & Occlusion Analysis | Recording: {recording_id} | Frame: {frame} | "
        f"Ego: {ego_id} | Agents: {len(agents)} | Occlusions: {len(all_occlusions)}",
        fontsize=14, fontweight='bold', color='white', y=0.98
    )
    fig_comb.tight_layout(rect=[0, 0, 1, 0.96])
    fig_comb.savefig(paths['combined'], dpi=150, bbox_inches='tight',
                    facecolor=fig_comb.get_facecolor())
    plt.close(fig_comb)
    logger.info(f"Saved: {paths['combined'].name}")
    
    # 7. Occlusion CSV Log
    occ_logger = OcclusionLogger(recording_id)
    
    if log_all_frames:
        # Log occlusions for all frames
        logger.info("Logging occlusions for all frames...")
        all_frames = sorted(loader.tracks_df['frame'].unique())
        for f in all_frames[::5]:  # Sample every 5 frames
            f_agents = loader.classify_frame_agents(f)
            f_agent_dicts = [
                {'id': a.id, 'x': a.x, 'y': a.y, 'heading': a.heading,
                 'vx': a.vx, 'vy': a.vy,
                 'width': a.width, 'length': a.length, 'class': a.vehicle_class}
                for a in f_agents
            ]
            f_occlusions = loader.occlusion_detector.find_all_occlusions(f_agent_dicts, f)
            occ_logger.add_events(f_occlusions)
    else:
        # Just log current frame
        occ_logger.add_events(all_occlusions)
    
    occ_df = occ_logger.save_csv(paths['occlusion_csv'])
    
    # 8. Animation
    if create_animation:
        logger.info("Creating animation...")
        _create_animation(loader, viz, ego_id, frame, animation_frames, 
                         animation_step, paths['animation'])
    
    # Save metadata
    role_counts = defaultdict(int)
    for agent in agents:
        role_counts[agent.role.value] += 1
    
    metadata = {
        'recording_id': recording_id,
        'frame': frame,
        'ego_id': ego_id,
        'total_agents': len(agents),
        'total_occlusions': len(all_occlusions),
        'ego_caused_occlusions': len(ego_occlusions),
        'role_distribution': dict(role_counts),
        'output_folder': str(output_folder),
        'files': {k: str(v) for k, v in paths.items()}
    }
    output_mgr.save_metadata(output_folder, metadata)
    
    logger.info("=" * 60)
    logger.info("Analysis Complete!")
    logger.info(f"Outputs saved to: {output_folder}")
    logger.info("=" * 60)
    
    return metadata


def _create_animation(loader: ExiDRoleLoader, viz: RoleOcclusionVisualizer,
                      ego_id: Optional[int], center_frame: int,
                      num_frames: int, step: int, output_path: Path):
    """Create traffic snapshot animation."""
    frames_available = sorted(loader.tracks_df['frame'].unique())
    
    half_window = num_frames // 2
    frame_min = center_frame - half_window
    frame_max = center_frame + half_window
    
    frames = [f for f in frames_available if frame_min <= f <= frame_max]
    frames = frames[::step]
    
    if not frames:
        logger.warning("No frames for animation")
        return
    
    # Precompute data
    precomputed = []
    for frame_id in frames:
        agents = loader.classify_frame_agents(frame_id)
        agent_dicts = [
            {'id': a.id, 'x': a.x, 'y': a.y, 'heading': a.heading,
             'vx': a.vx, 'vy': a.vy,
             'width': a.width, 'length': a.length, 'class': a.vehicle_class}
            for a in agents
        ]
        occlusions = loader.occlusion_detector.find_all_occlusions(agent_dicts, frame_id)
        precomputed.append((frame_id, agents, occlusions))
    
    fig, ax = plt.subplots(figsize=(16, 9))
    fig.patch.set_facecolor(viz.config.BG_DARK)
    
    def update(idx):
        ax.clear()
        frame_id, agents, occlusions = precomputed[idx]
        viz.plot_traffic_snapshot(ax, agents, occlusions, ego_id, use_full_extent=True,
                                 title=f"Traffic Snapshot | Frame {frame_id}")
        return []
    
    anim = FuncAnimation(fig, update, frames=len(precomputed), 
                        interval=100, blit=False, repeat=False)
    
    try:
        writer = PillowWriter(fps=10)
        anim.save(str(output_path), writer=writer, dpi=100,
                 savefig_kwargs={'facecolor': fig.get_facecolor()})
        logger.info(f"Saved: {output_path.name}")
    except Exception as e:
        logger.error(f"Failed to save animation: {e}")
    
    plt.close(fig)


# =============================================================================
# CLI
# =============================================================================

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Agent Role and Occlusion Analysis (v3)')
    parser.add_argument('--data_dir', type=str, default='./data',
                       help='Path to exiD data directory')
    parser.add_argument('--recording', type=int, default=25,
                       help='Recording ID to analyze')
    parser.add_argument('--ego_id', type=int, default=None,
                       help='Ego vehicle ID')
    parser.add_argument('--frame', type=int, default=None,
                       help='Frame to analyze')
    parser.add_argument('--output_dir', type=str, default='./output_roles',
                       help='Output directory')
    parser.add_argument('--no-animation', action='store_true',
                       help='Skip animation generation')
    parser.add_argument('--log-all-frames', action='store_true',
                       help='Log occlusions for all frames to CSV')
    
    args = parser.parse_args()
    
    summary = analyze_recording(
        args.data_dir, 
        args.recording, 
        args.ego_id, 
        args.frame,
        args.output_dir,
        create_animation=not args.no_animation,
        log_all_frames=args.log_all_frames
    )
    
    print(f"\n{'='*50}")
    print("Analysis Complete")
    print(f"{'='*50}")
    for key, value in summary.items():
        if key != 'files':
            print(f"  {key}: {value}")