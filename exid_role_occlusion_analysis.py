"""
exiD Dataset: Agent Role Classification and Occlusion Detection
================================================================
Extends GVF/SVO analysis with:
1. Agent role classification: Normal, Merging, Exiting, Yielding
2. Truck occlusion detection for surrounding vehicles
3. Scenario extraction for PINN field learning
4. Road topology visualization with background

Visualization style follows the GVF/SVO reference implementation.

For PINN-based interaction field learning.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Polygon, FancyBboxPatch, FancyArrowPatch, Wedge
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.collections import LineCollection
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Set, NamedTuple
from collections import defaultdict
from enum import Enum
import warnings
import logging

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


# =============================================================================
# Agent Role Definitions
# =============================================================================

class AgentRole(Enum):
    """Agent behavioral roles in highway merging scenarios."""
    NORMAL_MAIN = "normal_main"           # Normal driving in main lane
    MERGING = "merging"                    # On acceleration lane, intending to merge
    EXITING = "exiting"                    # Intending to exit via off-ramp
    YIELDING = "yielding"                  # Yielding to merging traffic
    UNKNOWN = "unknown"


class OcclusionType(Enum):
    """Types of occlusion relationships."""
    FULL = "full"               # Completely blocked
    PARTIAL = "partial"         # Partially blocked
    NONE = "none"               # Clear line of sight


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
    """Describes an occlusion relationship."""
    occluder_id: int
    occluded_id: int
    blocked_id: int
    occlusion_type: OcclusionType
    occlusion_ratio: float
    angular_span: Tuple[float, float]


@dataclass
class InteractionScenario:
    """A complete interaction scenario for PINN training."""
    recording_id: int
    frame_start: int
    frame_end: int
    ego_id: int
    ego_role: AgentRole
    agents: List[AgentState]
    occlusions: List[OcclusionEvent]
    scenario_type: str


# =============================================================================
# Configuration (matching GVF/SVO style)
# =============================================================================

@dataclass
class Config:
    """Configuration matching GVF/SVO visualization."""
    
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
    MIN_OCCLUSION_ANGLE: float = 5.0
    
    # Physical
    MASS_HV: float = 15000.0
    MASS_PC: float = 3000.0
    
    # Visualization colors (matching GVF/SVO)
    FPS: int = 25
    COLORS: Dict[str, str] = field(default_factory=lambda: {
        'truck': '#E74C3C',
        'car': '#3498DB', 
        'bus': '#F39C12',
        'van': '#9B59B6',
    })
    
    # Role colors
    ROLE_COLORS: Dict[str, str] = field(default_factory=lambda: {
        'normal_main': '#3498DB',   # Blue
        'merging': '#E74C3C',       # Red  
        'exiting': '#F39C12',       # Orange
        'yielding': '#9B59B6',      # Purple
        'unknown': '#95A5A6',       # Gray
    })
    
    # Theme colors
    BG_DARK: str = '#0D1117'
    BG_PANEL: str = '#1A1A2E'
    SPINE_COLOR: str = '#4A4A6A'


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
        """
        Classify agent role based on current state and context.
        
        Returns:
            (role, confidence, urgency)
        """
        x, y = agent['x'], agent['y']
        vx, vy = agent.get('vx', 0), agent.get('vy', 0)
        
        lane_type = lane_info.get('lane_type', 'main')
        lat_vel = abs(vy)
        
        role = AgentRole.NORMAL_MAIN
        confidence = 0.5
        urgency = 0.0
        
        # Check acceleration lane (merging)
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
                
        # Check exiting behavior
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
                
        # Check yielding
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
        if lane_info.get('lane_type') == 'accel':
            return True
        
        accel_bounds = lane_info.get('accel_lane_y_bounds')
        if accel_bounds:
            y_min, y_max = accel_bounds
            if y_min <= y <= y_max:
                return True
        
        lane_centers = lane_info.get('lane_centers', [])
        lane_width = lane_info.get('lane_width', 3.5)
        if lane_centers:
            main_band_low = min(lane_centers) - lane_width * 0.6
            main_band_high = max(lane_centers) + lane_width * 0.6
            if y < main_band_low or y > main_band_high:
                return True
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
# Occlusion Detection
# =============================================================================

class OcclusionDetector:
    """Detects occlusion relationships between vehicles."""
    
    def __init__(self, config: Config = None):
        self.config = config or Config()
        
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
                        potential_occluders: List[Dict]) -> Tuple[OcclusionType, float, Optional[int]]:
        """Check if target is occluded from observer's view."""
        dx = target['x'] - observer['x']
        dy = target['y'] - observer['y']
        dist_to_target = np.sqrt(dx**2 + dy**2)
        
        if dist_to_target < 1.0 or dist_to_target > self.config.OCCLUSION_RANGE:
            return (OcclusionType.NONE, 0.0, None)
            
        target_angle = np.arctan2(dy, dx)
        target_heading = target.get('heading', 0)
        angle_diff = target_angle - target_heading
        target_eff_width = abs(target['length'] * np.sin(angle_diff)) + \
                          abs(target['width'] * np.cos(angle_diff))
        target_half_angle = np.arctan2(target_eff_width / 2, dist_to_target)
        target_angle_range = (target_angle - target_half_angle, target_angle + target_half_angle)
        
        max_occlusion = 0.0
        occluding_vehicle = None
        
        for occluder in potential_occluders:
            if occluder['id'] == observer.get('id') or occluder['id'] == target.get('id'):
                continue
                
            occ_dx = occluder['x'] - observer['x']
            occ_dy = occluder['y'] - observer['y']
            dist_to_occluder = np.sqrt(occ_dx**2 + occ_dy**2)
            
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
                
        if max_occlusion > 0.8:
            occ_type = OcclusionType.FULL
        elif max_occlusion > 0.2:
            occ_type = OcclusionType.PARTIAL
        else:
            occ_type = OcclusionType.NONE
            
        return (occ_type, max_occlusion, occluding_vehicle)
    
    def _angle_range_overlap(self, range1: Tuple[float, float],
                             range2: Tuple[float, float]) -> float:
        start = max(range1[0], range2[0])
        end = min(range1[1], range2[1])
        return max(0, end - start)
    
    def find_all_occlusions(self, agents: List[Dict]) -> List[OcclusionEvent]:
        """Find all occlusion relationships in a scene."""
        occlusions = []
        
        trucks = [a for a in agents if a.get('class', '').lower() in self.config.HEAVY_VEHICLE_CLASSES]
        cars = [a for a in agents if a.get('class', '').lower() in self.config.CAR_CLASSES]
        
        # Cars blocked by trucks
        for observer in cars:
            for target in cars:
                if observer['id'] == target['id']:
                    continue
                    
                occ_type, occ_ratio, occluder_id = self.check_occlusion(
                    observer, target, trucks
                )
                
                if occ_type != OcclusionType.NONE:
                    occlusions.append(OcclusionEvent(
                        occluder_id=occluder_id,
                        occluded_id=target['id'],
                        blocked_id=observer['id'],
                        occlusion_type=occ_type,
                        occlusion_ratio=occ_ratio,
                        angular_span=(0, 0)
                    ))
                    
        # Trucks blocking each other's view of cars
        for observer in trucks:
            for target in cars:
                occ_type, occ_ratio, occluder_id = self.check_occlusion(
                    observer, target, trucks
                )
                
                if occ_type != OcclusionType.NONE and occluder_id != observer['id']:
                    occlusions.append(OcclusionEvent(
                        occluder_id=occluder_id,
                        occluded_id=target['id'],
                        blocked_id=observer['id'],
                        occlusion_type=occ_type,
                        occlusion_ratio=occ_ratio,
                        angular_span=(0, 0)
                    ))
                    
        return occlusions


# =============================================================================
# Data Loader (with background support - matching GVF/SVO)
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
        
        # Background image support (matching GVF/SVO)
        self.background_image = None
        self.ortho_px_to_meter = 0.1
        
        # Lane structure
        self.lane_structure = {}
        self.merge_bounds: Dict[int, Tuple[float, float]] = {}
        
    def load_recording(self, recording_id: int) -> bool:
        """Load recording data including background."""
        prefix = f"{recording_id:02d}_"
        self.recording_id = recording_id
        
        try:
            self.tracks_df = pd.read_csv(self.data_dir / f"{prefix}tracks.csv")
            self.tracks_meta_df = pd.read_csv(self.data_dir / f"{prefix}tracksMeta.csv")
            
            # Recording metadata (for background scaling) - matching GVF/SVO
            rec_meta_path = self.data_dir / f"{prefix}recordingMeta.csv"
            if rec_meta_path.exists():
                rec_meta_df = pd.read_csv(rec_meta_path)
                if not rec_meta_df.empty:
                    self.recording_meta = rec_meta_df.iloc[0]
                    self.ortho_px_to_meter = float(
                        self.recording_meta.get('orthoPxToMeter', self.ortho_px_to_meter)
                    )
            
            # Load background image - matching GVF/SVO
            bg_path = self.data_dir / f"{prefix}background.png"
            if bg_path.exists():
                self.background_image = plt.imread(str(bg_path))
                logger.info("Loaded lane layout background image.")
            else:
                logger.warning(f"Background image not found: {bg_path}")
            
            # Merge metadata
            self.tracks_df = self.tracks_df.merge(
                self.tracks_meta_df[['trackId', 'class', 'width', 'length']],
                on='trackId', how='left', suffixes=('', '_meta')
            )
            
            if 'width_meta' in self.tracks_df.columns:
                self.tracks_df['width'] = self.tracks_df['width'].fillna(self.tracks_df['width_meta'])
                self.tracks_df['length'] = self.tracks_df['length'].fillna(self.tracks_df['length_meta'])
            
            # Infer lane structure
            self._infer_lane_structure()
            
            logger.info(f"Loaded recording {recording_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading recording: {e}")
            return False
    
    def get_background_extent(self) -> List[float]:
        """Get extent for plotting background image in meters - matching GVF/SVO."""
        if self.background_image is None:
            return [0, 0, 0, 0]
        h, w = self.background_image.shape[:2]
        return [0, w * self.ortho_px_to_meter, -h * self.ortho_px_to_meter, 0]
    
    def _infer_lane_structure(self):
        """Infer lane structure from trajectory data."""
        if self.tracks_df is None:
            return
            
        y_vals = self.tracks_df['yCenter'].dropna()
        
        if len(y_vals) == 0:
            return
            
        # Histogram to find lane centers
        hist, bin_edges = np.histogram(y_vals, bins=50)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Find peaks
        try:
            from scipy.signal import find_peaks
            peaks, _ = find_peaks(hist, height=len(y_vals) * 0.01, distance=5)
            lane_centers = bin_centers[peaks]
        except ImportError:
            # Fallback without scipy
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
        
        # Rightmost lane as potential acceleration lane
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
            'is_merge_adjacent': False,
            'accel_lane_y_bounds': None,
            'lane_centers': [],
            'lane_width': 3.5
        }
        
        if not self.lane_structure:
            return lane_info
            
        lane_centers = self.lane_structure.get('lane_centers', [])
        lane_width = self.lane_structure.get('lane_width', 3.5)
        lane_info['lane_centers'] = lane_centers
        lane_info['lane_width'] = lane_width
        lane_info['accel_lane_y_bounds'] = self.lane_structure.get('accel_lane_y_bounds')
        
        if not lane_centers:
            return lane_info
            
        # Find closest lane
        distances = [abs(y - lc) for lc in lane_centers]
        closest_idx = np.argmin(distances)
        
        if distances[closest_idx] < lane_width:
            lane_info['lane_id'] = closest_idx
            
        # Check acceleration lane
        accel_bounds = self.lane_structure.get('accel_lane_y_bounds')
        accel_margin = lane_width * 0.6
        band_low = min(lane_centers) - accel_margin
        band_high = max(lane_centers) + accel_margin
        
        if accel_bounds and (accel_bounds[0] - accel_margin) <= y <= (accel_bounds[1] + accel_margin):
            lane_info['lane_type'] = 'accel'
            lane_info['lane_id'] = None
        elif y <= band_low or y >= band_high:
            lane_info['lane_type'] = 'accel'
            lane_info['lane_id'] = None
                
        # Merge adjacent (outermost main lanes)
        if closest_idx == len(lane_centers) - 1 or closest_idx == 0:
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
            
            # Trajectory history
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
    
    def find_occlusion_scenarios(self, min_frames: int = 25) -> List[Dict]:
        """Find scenarios where trucks block views between cars."""
        scenarios = []
        
        if self.tracks_df is None:
            return scenarios
            
        frames = sorted(self.tracks_df['frame'].unique())
        active_occlusions = defaultdict(list)
        
        for frame in frames:
            frame_data = self.tracks_df[self.tracks_df['frame'] == frame]
            
            agents = []
            for _, row in frame_data.iterrows():
                vclass = str(row.get('class', 'car')).lower()
                agents.append({
                    'id': int(row['trackId']),
                    'x': float(row['xCenter']),
                    'y': float(row['yCenter']),
                    'heading': np.radians(float(row.get('heading', 0))),
                    'width': float(row.get('width', 2.0)),
                    'length': float(row.get('length', 5.0)),
                    'class': vclass,
                })
            
            occlusions = self.occlusion_detector.find_all_occlusions(agents)
            
            current_keys = set()
            for occ in occlusions:
                key = (occ.occluder_id, occ.blocked_id, occ.occluded_id)
                current_keys.add(key)
                active_occlusions[key].append(frame)
                
            ended_keys = set(active_occlusions.keys()) - current_keys
            for key in ended_keys:
                frames_list = active_occlusions.pop(key)
                if len(frames_list) >= min_frames:
                    scenarios.append({
                        'truck_id': key[0],
                        'blocked_car_id': key[1],
                        'occluded_car_id': key[2],
                        'frame_start': min(frames_list),
                        'frame_end': max(frames_list),
                        'duration_frames': len(frames_list),
                        'duration_seconds': len(frames_list) / self.config.FPS
                    })
        
        # Remaining active
        for key, frames_list in active_occlusions.items():
            if len(frames_list) >= min_frames:
                scenarios.append({
                    'truck_id': key[0],
                    'blocked_car_id': key[1],
                    'occluded_car_id': key[2],
                    'frame_start': min(frames_list),
                    'frame_end': max(frames_list),
                    'duration_frames': len(frames_list),
                    'duration_seconds': len(frames_list) / self.config.FPS
                })
                
        return scenarios
    
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
            
            # Count nearby vehicles
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
# Visualization (matching GVF/SVO style exactly)
# =============================================================================

class RoleOcclusionVisualizer:
    """Visualize agent roles and occlusions with road topology - matching GVF/SVO style."""
    
    def __init__(self, config: Config = None, loader: ExiDRoleLoader = None):
        self.config = config or Config()
        self.loader = loader
        
    def create_combined_figure(self, agents: List[AgentState], 
                               occlusions: List[OcclusionEvent],
                               ego_id: Optional[int] = None,
                               frame: int = 0,
                               output_path: str = None):
        """
        Create combined figure with:
        1. Traffic snapshot with background and roles
        2. Occlusion diagram (ego-centric)
        3. Role distribution summary
        4. Occlusion details
        
        Style matches GVF/SVO visualization.
        """
        fig = plt.figure(figsize=(20, 12))
        fig.patch.set_facecolor(self.config.BG_DARK)
        
        gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.25)
        
        ax1 = fig.add_subplot(gs[0, :2])  # Main traffic view (wider)
        ax2 = fig.add_subplot(gs[0, 2])   # Role distribution
        ax3 = fig.add_subplot(gs[1, 0])   # Occlusion diagram (ego-centric)
        ax4 = fig.add_subplot(gs[1, 1])   # Occlusion details
        ax5 = fig.add_subplot(gs[1, 2])   # Summary panel
        
        # 1. Traffic snapshot with background
        self._plot_traffic_snapshot(ax1, agents, occlusions, ego_id)
        
        # 2. Role distribution
        self._plot_role_distribution(ax2, agents)
        
        # 3. Occlusion diagram
        if ego_id is not None:
            ego_agent = next((a for a in agents if a.id == ego_id), None)
            if ego_agent:
                self._plot_occlusion_diagram(ax3, agents, occlusions, ego_agent)
            else:
                self._plot_placeholder(ax3, "Occlusion Diagram\n(No ego selected)")
        else:
            self._plot_placeholder(ax3, "Occlusion Diagram\n(No ego selected)")
        
        # 4. Occlusion details
        self._plot_occlusion_details(ax4, agents, occlusions)
        
        # 5. Summary panel
        self._plot_summary(ax5, agents, occlusions, ego_id, frame)
        
        # Title - matching GVF/SVO style
        ego_info = ""
        if ego_id is not None:
            ego_agent = next((a for a in agents if a.id == ego_id), None)
            if ego_agent:
                ego_info = f" | Ego: {ego_agent.vehicle_class.title()} (ID: {ego_id})"
        
        fig.suptitle(
            f"Role & Occlusion Analysis | Recording: {self.loader.recording_id if self.loader else '?'} | "
            f"Frame: {frame}{ego_info} | Agents: {len(agents)}",
            fontsize=14, fontweight='bold', color='white', y=0.98
        )
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        if output_path:
            fig.savefig(output_path, dpi=150, bbox_inches='tight', 
                       facecolor=fig.get_facecolor())
            logger.info(f"Saved: {output_path}")
            plt.close(fig)
        else:
            plt.show()
            
        return fig
    
    def _plot_traffic_snapshot(self, ax, agents: List[AgentState],
                               occlusions: List[OcclusionEvent],
                               ego_id: Optional[int] = None):
        """Plot traffic snapshot with background and roles - matching GVF/SVO style."""
        ax.set_facecolor(self.config.BG_PANEL)
        arrow_scale = 0.5  # keep velocity arrow length consistent and usable for padding
        
        # Background image - exactly matching GVF/SVO
        bg_extent = None
        if self.loader and self.loader.background_image is not None:
            bg_extent = self.loader.get_background_extent()
            ax.imshow(self.loader.background_image, extent=bg_extent, 
                     alpha=0.6, aspect='equal', zorder=0)
        
        # Compute bounds - matching GVF/SVO style
        all_x = [a.x for a in agents]
        all_y = [a.y for a in agents]
        
        if ego_id is not None:
            ego_agent = next((a for a in agents if a.id == ego_id), None)
            if ego_agent:
                x_center = ego_agent.x
                y_center = ego_agent.y
            else:
                x_center = np.mean(all_x)
                y_center = np.mean(all_y)
        else:
            x_center = np.mean(all_x)
            y_center = np.mean(all_y)
        
        # Add padding so vehicle arrows/labels never clip outside the axes
        max_length = max((a.length for a in agents), default=0.0)
        max_width = max((a.width for a in agents), default=0.0)
        max_vx = max((abs(a.vx) for a in agents), default=0.0)
        max_vy = max((abs(a.vy) for a in agents), default=0.0)
        max_label_offset = max((a.width / 2 + 1.5 for a in agents), default=1.5)
        
        padding_x = max(8.0, max_length / 2 + max_vx * arrow_scale + 3.0)
        padding_y = max(8.0, max_label_offset + max_vy * arrow_scale + 3.0)
        
        data_min_x = min(all_x) - padding_x
        data_max_x = max(all_x) + padding_x
        data_min_y = min(all_y) - padding_y
        data_max_y = max(all_y) + padding_y
        
        max_dx = max(abs(x_center - data_min_x), abs(data_max_x - x_center))
        max_dy = max(abs(y_center - data_min_y), abs(data_max_y - y_center))
        
        span_x = max(140, 2 * max_dx + 10, 
                    self.config.OBS_RANGE_AHEAD + self.config.OBS_RANGE_BEHIND + 40)
        span_y = max(60, 2 * max_dy + 8,
                    self.config.OBS_RANGE_LEFT + self.config.OBS_RANGE_RIGHT + 20)
        half_x = span_x / 2
        half_y = span_y / 2
        
        x_min = x_center - half_x
        x_max = x_center + half_x
        y_min = y_center - half_y
        y_max = y_center + half_y
        
        if bg_extent:
            x_min = min(x_min, bg_extent[0])
            x_max = max(x_max, bg_extent[1])
            y_min = min(y_min, bg_extent[2])
            y_max = max(y_max, bg_extent[3])
            
            # Slight margin beyond background so edges are visible
            margin_x = 0.05 * (x_max - x_min)
            margin_y = 0.05 * (y_max - y_min)
            x_min -= margin_x
            x_max += margin_x
            y_min -= margin_y
            y_max += margin_y
        else:
            # Draw lane markings if no background - matching GVF/SVO
            lane_centers = self.loader.lane_structure.get('lane_centers', []) if self.loader else []
            for lc in lane_centers:
                ax.axhline(lc, color='white', linestyle='--', alpha=0.4, linewidth=1, zorder=1)
        
        # Draw occlusion shadows first (so vehicles are on top)
        for occ in occlusions:
            occluder = next((a for a in agents if a.id == occ.occluder_id), None)
            blocked = next((a for a in agents if a.id == occ.blocked_id), None)
            occluded = next((a for a in agents if a.id == occ.occluded_id), None)
            
            if all([occluder, blocked, occluded]):
                shadow_color = '#E74C3C' if occ.occlusion_type == OcclusionType.FULL else '#F39C12'
                
                # Draw shadow cone
                ax.plot([blocked.x, occluder.x], [blocked.y, occluder.y],
                       color=shadow_color, linestyle=':', alpha=0.6, linewidth=1.5, zorder=2)
                ax.plot([occluder.x, occluded.x], [occluder.y, occluded.y],
                       color=shadow_color, linestyle=':', alpha=0.6, linewidth=1.5, zorder=2)
                
                # Filled triangle for shadow
                triangle = plt.Polygon(
                    [[blocked.x, blocked.y], [occluder.x, occluder.y], [occluded.x, occluded.y]],
                    closed=True, facecolor=shadow_color, alpha=0.1, edgecolor='none', zorder=1
                )
                ax.add_patch(triangle)
        
        # Draw vehicles - matching GVF/SVO style
        for agent in agents:
            is_ego = (agent.id == ego_id)
            self._draw_vehicle(ax, agent, is_ego=is_ego, show_role=True)
            
            # Velocity arrow - matching GVF/SVO style
            arrow_color = 'yellow' if is_ego else 'cyan'
            head_width = 1 if is_ego else 0.8
            head_length = 0.5 if is_ego else 0.4
            alpha = 1.0 if is_ego else 0.7
            
            ax.arrow(agent.x, agent.y, 
                    agent.vx * arrow_scale, agent.vy * arrow_scale,
                    head_width=head_width, head_length=head_length,
                    fc=arrow_color, ec=arrow_color, alpha=alpha, zorder=5)
        
        # Legend - matching GVF/SVO style
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
                          linewidth=2, label='Truck'),
        ]
        ax.legend(handles=legend_elements, loc='upper left', fontsize=8,
                 facecolor=self.config.BG_PANEL, edgecolor='white', 
                 labelcolor='white')
        
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_xlabel('X (m)', color='white')
        ax.set_ylabel('Y (m)', color='white')
        ax.set_title('Traffic Snapshot with Roles & Occlusions', 
                    fontsize=11, fontweight='bold', color='white')
        ax.tick_params(colors='white')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.2)
        
        for spine in ax.spines.values():
            spine.set_color(self.config.SPINE_COLOR)
    
    def _draw_vehicle(self, ax, agent: AgentState, is_ego: bool = False, 
                      show_role: bool = True):
        """Draw a vehicle with role-based coloring - matching GVF/SVO style."""
        is_truck = agent.vehicle_class in self.config.HEAVY_VEHICLE_CLASSES
        
        # Color selection matching GVF/SVO
        if is_ego:
            color = self.config.COLORS.get(agent.vehicle_class, '#E74C3C')
            edgecolor = 'white'
            linewidth = 2
            alpha = 1.0
        elif is_truck:
            color = self.config.COLORS.get(agent.vehicle_class, '#E74C3C')
            edgecolor = 'white'
            linewidth = 2
            alpha = 0.9
        else:
            # Color by role for cars
            color = self.config.ROLE_COLORS.get(agent.role.value, '#3498DB')
            edgecolor = 'white'
            linewidth = 1
            alpha = 0.8
        
        # Rotated rectangle - matching GVF/SVO _get_rotated_rect and _draw_vehicle
        corners = self._get_rotated_rect(
            agent.x, agent.y, agent.length, agent.width, agent.heading
        )
        
        rect = plt.Polygon(corners, closed=True, facecolor=color,
                          edgecolor=edgecolor, linewidth=linewidth, 
                          alpha=alpha, zorder=4)
        ax.add_patch(rect)
        
        # Label - matching GVF/SVO style
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
        
        label = "\n".join(label_parts)
        text_color = 'white' if is_ego else 'yellow'
        fontsize = 8
        fontweight = 'bold' if is_ego else 'normal'
        
        ax.text(agent.x, agent.y + agent.width/2 + 1.5, label,
               ha='center', va='bottom', fontsize=fontsize,
               color=text_color, fontweight=fontweight, zorder=6)
    
    def _get_rotated_rect(self, cx, cy, length, width, heading):
        """Get corners of rotated rectangle - matching GVF/SVO exactly."""
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
    
    def _plot_role_distribution(self, ax, agents: List[AgentState]):
        """Plot role distribution bar chart - matching GVF/SVO style."""
        ax.set_facecolor(self.config.BG_PANEL)
        
        role_counts = defaultdict(int)
        for agent in agents:
            role_counts[agent.role.value] += 1
        
        roles = list(role_counts.keys())
        counts = [role_counts[r] for r in roles]
        colors = [self.config.ROLE_COLORS.get(r, '#95A5A6') for r in roles]
        
        bars = ax.bar(roles, counts, color=colors, edgecolor='white', alpha=0.8)
        
        # Add count labels
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
    
    def _plot_occlusion_diagram(self, ax, agents: List[AgentState],
                                occlusions: List[OcclusionEvent],
                                ego: AgentState):
        """Plot ego-centric occlusion diagram - matching GVF/SVO relative view style."""
        ax.set_facecolor(self.config.BG_PANEL)
        
        # Transform to ego frame
        cos_h = np.cos(-ego.heading)
        sin_h = np.sin(-ego.heading)
        
        # Draw ego at center - matching GVF/SVO FancyBboxPatch style
        is_truck = ego.vehicle_class in self.config.HEAVY_VEHICLE_CLASSES
        ego_color = self.config.COLORS.get(ego.vehicle_class, '#E74C3C')
        
        ego_rect = mpatches.FancyBboxPatch(
            (-ego.length/2, -ego.width/2), ego.length, ego.width,
            boxstyle="round,pad=0.02",
            facecolor=ego_color,
            edgecolor='white', linewidth=2, zorder=4
        )
        ax.add_patch(ego_rect)
        ax.text(0, -ego.width/2 - 2, "EGO", ha='center', va='top', 
               color='white', fontsize=9, fontweight='bold')
        
        # Draw other vehicles in ego frame
        max_ahead = max_back = max_left = max_right = 0.0
        for agent in agents:
            if agent.id == ego.id:
                continue
                
            dx = agent.x - ego.x
            dy = agent.y - ego.y
            dx_rel = dx * cos_h - dy * sin_h
            dy_rel = dx * sin_h + dy * cos_h
            
            is_other_truck = agent.vehicle_class in self.config.HEAVY_VEHICLE_CLASSES
            if is_other_truck:
                color = self.config.COLORS.get(agent.vehicle_class, '#E74C3C')
            else:
                color = self.config.ROLE_COLORS.get(agent.role.value, '#3498DB')
            
            long_half = agent.length / 2
            lat_half = agent.width / 2 + 1.0
            if dx_rel >= 0:
                max_ahead = max(max_ahead, dx_rel + long_half)
            else:
                max_back = max(max_back, abs(dx_rel) + long_half)
            if dy_rel >= 0:
                max_left = max(max_left, dy_rel + lat_half)
            else:
                max_right = max(max_right, abs(dy_rel) + lat_half)
            
            rect = mpatches.FancyBboxPatch(
                (dx_rel - agent.length/2, dy_rel - agent.width/2),
                agent.length, agent.width,
                boxstyle="round,pad=0.02",
                facecolor=color, edgecolor='white', linewidth=1, alpha=0.8, zorder=3
            )
            ax.add_patch(rect)
            ax.text(dx_rel, dy_rel + agent.width/2 + 1, str(agent.id),
                   ha='center', va='bottom', fontsize=7, color='yellow')
        
        # Draw occlusion lines involving ego
        for occ in occlusions:
            if occ.blocked_id != ego.id:
                continue
                
            occluder = next((a for a in agents if a.id == occ.occluder_id), None)
            occluded = next((a for a in agents if a.id == occ.occluded_id), None)
            
            if occluder and occluded:
                # Transform positions
                dx1 = occluder.x - ego.x
                dy1 = occluder.y - ego.y
                dx1_rel = dx1 * cos_h - dy1 * sin_h
                dy1_rel = dx1 * sin_h + dy1 * cos_h
                
                dx2 = occluded.x - ego.x
                dy2 = occluded.y - ego.y
                dx2_rel = dx2 * cos_h - dy2 * sin_h
                dy2_rel = dx2 * sin_h + dy2 * cos_h
                
                occ_color = '#E74C3C' if occ.occlusion_type == OcclusionType.FULL else '#F39C12'
                
                # Shadow cone from ego
                ax.plot([0, dx1_rel], [0, dy1_rel], color=occ_color, 
                       linestyle='-', alpha=0.5, linewidth=2, zorder=2)
                ax.plot([dx1_rel, dx2_rel], [dy1_rel, dy2_rel], color=occ_color,
                       linestyle='--', alpha=0.7, linewidth=2, zorder=2)
                
                # X marker on occluded vehicle
                ax.scatter([dx2_rel], [dy2_rel], marker='x', s=100, 
                          c=occ_color, zorder=5, linewidths=2)
        
        # Lane markings in ego frame
        ax.axhline(3.5, color='white', linestyle='--', alpha=0.5)
        ax.axhline(-3.5, color='white', linestyle='--', alpha=0.5)
        ax.axhline(7, color='white', linestyle='-', alpha=0.5)
        ax.axhline(-7, color='white', linestyle='-', alpha=0.5)
        
        # Set limits
        margin_long = 5.0
        margin_lat = 3.0
        ahead_limit = max(self.config.OBS_RANGE_AHEAD, max_ahead + margin_long)
        back_limit = max(self.config.OBS_RANGE_BEHIND, max_back + margin_long)
        left_limit = max(self.config.OBS_RANGE_LEFT, max_left + margin_lat)
        right_limit = max(self.config.OBS_RANGE_RIGHT, max_right + margin_lat)
        
        ax.set_xlim(-back_limit, ahead_limit)
        ax.set_ylim(-right_limit, left_limit)
        ax.set_xlabel('Longitudinal (m)', color='white')
        ax.set_ylabel('Lateral (m)', color='white')
        ax.set_title('Ego-Centric Occlusion View', fontsize=11, fontweight='bold', color='white')
        ax.tick_params(colors='white')
        ax.set_aspect('equal')
        
        for spine in ax.spines.values():
            spine.set_color(self.config.SPINE_COLOR)
    
    def _plot_occlusion_details(self, ax, agents: List[AgentState],
                                occlusions: List[OcclusionEvent]):
        """Plot occlusion details as table - matching GVF/SVO summary style."""
        ax.set_facecolor(self.config.BG_PANEL)
        ax.axis('off')
        
        if not occlusions:
            ax.text(0.5, 0.5, 'No Occlusions Detected', ha='center', va='center',
                   color='white', fontsize=12)
            for spine in ax.spines.values():
                spine.set_color(self.config.SPINE_COLOR)
            return
        
        # Create table data
        headers = ['Truck', 'Blocks', 'From', 'Type', 'Ratio']
        rows = []
        
        for occ in occlusions[:8]:  # Limit to 8 rows
            type_str = 'FULL' if occ.occlusion_type == OcclusionType.FULL else 'PARTIAL'
            rows.append([
                str(occ.occluder_id),
                str(occ.occluded_id),
                str(occ.blocked_id),
                type_str,
                f'{occ.occlusion_ratio:.0%}'
            ])
        
        table = ax.table(
            cellText=rows,
            colLabels=headers,
            loc='center',
            cellLoc='center',
            colColours=['#2C3E50'] * len(headers),
            cellColours=[['#34495E'] * len(headers)] * len(rows)
        )
        
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)
        
        for key, cell in table.get_celld().items():
            cell.set_text_props(color='white')
            cell.set_edgecolor(self.config.SPINE_COLOR)
        
        ax.set_title('Occlusion Details', fontsize=11, fontweight='bold', 
                    color='white', pad=20)
    
    def _plot_summary(self, ax, agents: List[AgentState], 
                      occlusions: List[OcclusionEvent],
                      ego_id: Optional[int], frame: int):
        """Plot summary statistics - matching GVF/SVO _plot_svo_summary style."""
        ax.set_facecolor(self.config.BG_PANEL)
        ax.axis('off')
        
        # Gather statistics
        n_trucks = sum(1 for a in agents if a.vehicle_class in self.config.HEAVY_VEHICLE_CLASSES)
        n_cars = sum(1 for a in agents if a.vehicle_class in self.config.CAR_CLASSES)
        n_merging = sum(1 for a in agents if a.role == AgentRole.MERGING)
        n_exiting = sum(1 for a in agents if a.role == AgentRole.EXITING)
        n_yielding = sum(1 for a in agents if a.role == AgentRole.YIELDING)
        
        n_full_occ = sum(1 for o in occlusions if o.occlusion_type == OcclusionType.FULL)
        n_partial_occ = sum(1 for o in occlusions if o.occlusion_type == OcclusionType.PARTIAL)
        
        max_urgency = max((a.urgency for a in agents), default=0)
        avg_speed = np.mean([a.speed for a in agents]) * 3.6  # km/h
        
        # Ego info
        ego_info = "Not selected"
        if ego_id is not None:
            ego_agent = next((a for a in agents if a.id == ego_id), None)
            if ego_agent:
                ego_info = f"{ego_agent.vehicle_class.title()} | {ego_agent.speed*3.6:.1f} km/h | {ego_agent.role.value}"
        
        # Summary text matching GVF/SVO monospace style
        summary_lines = [
            "ROLE & OCCLUSION SUMMARY",
            f"Ego Vehicle: {ego_info}",
            f"Frame: {frame}",
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
            f"  Normal:     {len(agents) - n_merging - n_exiting - n_yielding}",
            "",
            "Occlusions",
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
               fontsize=10, color='white', family='monospace',
               verticalalignment='top')
        
        for spine in ax.spines.values():
            spine.set_color(self.config.SPINE_COLOR)
    
    def _plot_placeholder(self, ax, text: str):
        """Plot placeholder for empty panels."""
        ax.set_facecolor(self.config.BG_PANEL)
        ax.axis('off')
        ax.text(0.5, 0.5, text, ha='center', va='center', 
               color='white', fontsize=12)
        for spine in ax.spines.values():
            spine.set_color(self.config.SPINE_COLOR)


# =============================================================================
# Main Analysis Pipeline
# =============================================================================

def analyze_recording(data_dir: str, recording_id: int,
                      ego_id: Optional[int] = None,
                      frame: Optional[int] = None,
                      output_dir: str = './output_roles') -> Dict:
    """
    Analyze a recording for roles and occlusions.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 60)
    logger.info("Role & Occlusion Analysis")
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
        else:
            logger.warning("No heavy vehicles found")
    
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
    
    # Classify agents
    agents = loader.classify_frame_agents(frame)
    logger.info(f"Classified {len(agents)} agents")
    
    # Find occlusions
    agent_dicts = [
        {'id': a.id, 'x': a.x, 'y': a.y, 'heading': a.heading,
         'width': a.width, 'length': a.length, 'class': a.vehicle_class}
        for a in agents
    ]
    occlusions = loader.occlusion_detector.find_all_occlusions(agent_dicts)
    logger.info(f"Found {len(occlusions)} occlusions")
    
    # Create visualization
    viz = RoleOcclusionVisualizer(loader=loader)
    output_file = output_path / f'role_occlusion_rec{recording_id}_ego{ego_id}_frame{frame}.png'
    viz.create_combined_figure(agents, occlusions, ego_id, frame, str(output_file))
    
    # Find occlusion scenarios
    logger.info("Finding occlusion scenarios...")
    occ_scenarios = loader.find_occlusion_scenarios(min_frames=25)
    logger.info(f"Found {len(occ_scenarios)} occlusion scenarios")
    
    # Role distribution
    role_counts = defaultdict(int)
    for agent in agents:
        role_counts[agent.role.value] += 1
    
    summary = {
        'recording_id': recording_id,
        'frame': frame,
        'ego_id': ego_id,
        'total_agents': len(agents),
        'occlusions': len(occlusions),
        'occlusion_scenarios': len(occ_scenarios),
        'role_distribution': dict(role_counts),
        'output_file': str(output_file)
    }
    
    logger.info(f"Role distribution: {dict(role_counts)}")
    logger.info(f"Output saved to: {output_file}")
    
    return summary


# =============================================================================
# Command-line Interface
# =============================================================================

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Agent Role and Occlusion Analysis')
    parser.add_argument('--data_dir', type=str, default='./data',
                       help='Path to exiD data directory')
    parser.add_argument('--recording', type=int, default=25,
                       help='Recording ID to analyze')
    parser.add_argument('--ego_id', type=int, default=None,
                       help='Ego vehicle ID (auto-select if not provided)')
    parser.add_argument('--frame', type=int, default=None,
                       help='Frame to analyze (auto-select if not provided)')
    parser.add_argument('--output_dir', type=str, default='./output_roles',
                       help='Output directory for visualizations')
    
    args = parser.parse_args()
    
    summary = analyze_recording(
        args.data_dir, 
        args.recording, 
        args.ego_id, 
        args.frame,
        args.output_dir
    )
    
    print(f"\n{'='*50}")
    print("Analysis Complete")
    print(f"{'='*50}")
    for key, value in summary.items():
        print(f"  {key}: {value}")
