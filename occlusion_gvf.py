"""
exiD Occlusion-Aware GVF Visualization (Microscopic Analysis)
=============================================================
Focused analysis on truck-induced occlusion scenarios:

Scenario A - EGO_BLOCKS_REAR:
    Ego truck drives in front of/parallel to a car, blocking that car's 
    view of vehicles in the merging lane. The rear/adjacent car cannot 
    see merging vehicles and may be surprised by merge conflicts.
    
    Merging lane:    🚙 Merging car (OCCLUDED)
    Main lane:       🚛 EGO TRUCK ─── 🚗 Rear car (BLOCKED)

Scenario B - MERGE_TWO_WAY (Mutual Occlusion):
    After a sedan merges onto the ramp, its rear view is blocked by a truck.
    The sedan cannot see main road vehicles behind, while main road vehicles 
    also cannot see the merging sedan. Both lack awareness of potential 
    lane change conflict.
    
    Main lane:       🚗 Main car (BLOCKED - can't see sedan)
                     🚛 Truck (mutual occluder)
    Merge lane:      🚙 Sedan (BLOCKED - can't see main cars)

Features:
- Maximum 6 surrounding vehicles for microscopic analysis
- Visibility-weighted GVF field
- Dynamic animation showing field evolution
- Clear visualization of blocked/occluded relationships

Usage:
    python exid_occlusion_gvf_microscopic.py --data_dir ./data --recording 25
    python exid_occlusion_gvf_microscopic.py --demo
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Polygon, FancyBboxPatch, FancyArrowPatch
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.animation import FuncAnimation, PillowWriter
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Set
from collections import defaultdict
from enum import Enum
import warnings
import argparse
import logging

from numpy.linalg import inv
from sklearn.metrics.pairwise import rbf_kernel

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class Config:
    """Configuration for microscopic occlusion analysis."""
    
    HEAVY_VEHICLE_CLASSES: Set[str] = field(default_factory=lambda: {'truck', 'bus', 'van', 'trailer'})
    CAR_CLASSES: Set[str] = field(default_factory=lambda: {'car'})
    
    # Maximum surrounding vehicles for microscopic analysis
    MAX_SURROUNDING_VEHICLES: int = 6
    
    # Observation range (meters)
    OBS_RANGE_AHEAD: float = 60.0
    OBS_RANGE_BEHIND: float = 40.0
    OBS_RANGE_LATERAL: float = 15.0
    
    # GVF parameters
    GVF_GRID_X: int = 60
    GVF_GRID_Y: int = 30
    SIGMA_X: float = 10.0
    SIGMA_Y: float = 2.0
    
    # Occlusion parameters
    OCCLUSION_RANGE: float = 80.0
    MIN_VISIBILITY: float = 0.1
    VISIBILITY_ACCEL_SENSITIVITY: float = 0.15
    SAME_DIRECTION_THRESHOLD: float = np.pi / 2
    
    # Physical
    MASS_HV: float = 15000.0
    MASS_PC: float = 3000.0
    
    # Lane structure (typical highway)
    LANE_WIDTH: float = 3.5
    MAIN_LANE_Y_MAX: float = 5.0  # Above this is merge/accel lane
    
    # Visualization
    COLORS: Dict[str, str] = field(default_factory=lambda: {
        'truck': '#E74C3C',
        'car': '#3498DB',
        'bus': '#F39C12',
        'van': '#9B59B6',
        'ego': '#E74C3C',
        'blocked': '#3498DB',
        'occluded': '#F39C12',
        'occluder': '#C0392B',
    })
    
    BG_DARK: str = '#0D1117'
    BG_PANEL: str = '#161B22'
    SPINE_COLOR: str = '#30363D'
    
    # Occlusion zone colors
    SHADOW_COLOR: str = '#E74C3C'
    SHADOW_ALPHA: float = 0.15


class OcclusionScenario(Enum):
    """Focused occlusion scenarios."""
    EGO_BLOCKS_REAR = "ego_blocks_rear"      # Ego truck blocks rear car's view of merge lane
    MERGE_TWO_WAY = "merge_two_way"          # Mutual occlusion: merge & main can't see each other
    TRUCK_BLOCKS_MERGE = "truck_blocks_merge" # Truck blocks merging car's rear view
    NONE = "none"


# =============================================================================
# Occlusion Detection
# =============================================================================

class OcclusionDetector:
    """Focused detection for truck-induced occlusion scenarios."""
    
    def __init__(self, config: Config = None):
        self.config = config or Config()
    
    def compute_vehicle_corners(self, vehicle: Dict) -> np.ndarray:
        """Compute 4 corners of vehicle bounding box."""
        cx, cy = vehicle['x'], vehicle['y']
        heading = vehicle.get('heading', 0)
        length = vehicle.get('length', 4.5)
        width = vehicle.get('width', 1.8)
        
        half_l, half_w = length / 2, width / 2
        corners_local = np.array([
            [-half_l, -half_w], [half_l, -half_w],
            [half_l, half_w], [-half_l, half_w]
        ])
        
        cos_h, sin_h = np.cos(heading), np.sin(heading)
        R = np.array([[cos_h, -sin_h], [sin_h, cos_h]])
        
        return corners_local @ R.T + np.array([cx, cy])
    
    def compute_shadow_polygon(self, observer: Dict, occluder: Dict, 
                                fov_range: float = 80.0) -> np.ndarray:
        """Compute shadow polygon cast by occluder from observer's view."""
        obs_pos = np.array([observer['x'], observer['y']])
        corners = self.compute_vehicle_corners(occluder)
        
        # Find tangent points (extreme angles)
        angles = []
        for corner in corners:
            dx, dy = corner[0] - obs_pos[0], corner[1] - obs_pos[1]
            angles.append((np.arctan2(dy, dx), corner))
        angles.sort(key=lambda x: x[0])
        
        # Get left and right tangent points
        left_pt, right_pt = angles[-1][1], angles[0][1]
        
        # Extend to far points
        left_dir = left_pt - obs_pos
        right_dir = right_pt - obs_pos
        left_norm = left_dir / (np.linalg.norm(left_dir) + 1e-6)
        right_norm = right_dir / (np.linalg.norm(right_dir) + 1e-6)
        
        left_far = obs_pos + left_norm * fov_range
        right_far = obs_pos + right_norm * fov_range
        
        return np.array([right_pt, left_pt, left_far, right_far])
    
    def compute_occlusion_ratio(self, observer: Dict, target: Dict, 
                                 occluder: Dict) -> float:
        """Compute fraction of target occluded from observer's view."""
        dx_t = target['x'] - observer['x']
        dy_t = target['y'] - observer['y']
        dist_target = np.sqrt(dx_t**2 + dy_t**2)
        
        dx_o = occluder['x'] - observer['x']
        dy_o = occluder['y'] - observer['y']
        dist_occluder = np.sqrt(dx_o**2 + dy_o**2)
        
        # Occluder must be between observer and target
        if dist_occluder >= dist_target or dist_target < 1.0 or dist_occluder < 1.0:
            return 0.0
        
        # Angular extents
        target_angle = np.arctan2(dy_t, dx_t)
        target_width = target.get('width', 1.8) + target.get('length', 4.5) * 0.2
        target_half = np.arctan2(target_width / 2, dist_target)
        
        occluder_angle = np.arctan2(dy_o, dx_o)
        occluder_width = occluder.get('length', 12.0) * 0.4 + occluder.get('width', 2.5)
        occluder_half = np.arctan2(occluder_width / 2, dist_occluder)
        
        # Compute overlap
        target_range = (target_angle - target_half, target_angle + target_half)
        shadow_range = (occluder_angle - occluder_half, occluder_angle + occluder_half)
        
        overlap = max(0, min(target_range[1], shadow_range[1]) - 
                        max(target_range[0], shadow_range[0]))
        target_span = target_range[1] - target_range[0]
        
        return np.clip(overlap / target_span if target_span > 0 else 0, 0, 1)
    
    def is_in_merge_lane(self, vehicle: Dict) -> bool:
        """Check if vehicle is in merge/acceleration lane."""
        return vehicle['y'] > self.config.MAIN_LANE_Y_MAX
    
    def is_in_main_lane(self, vehicle: Dict) -> bool:
        """Check if vehicle is in main lane."""
        return vehicle['y'] <= self.config.MAIN_LANE_Y_MAX
    
    def detect_ego_blocks_rear(self, ego: Dict, vehicles: List[Dict]) -> List[Dict]:
        """
        Scenario A: Ego truck blocks rear car's view of merging vehicles.
        
        Find cases where:
        - A car is behind/beside the ego truck (in main lane)
        - That car cannot see a merging vehicle because ego blocks the view
        """
        events = []
        
        # Ego must be a truck
        if ego.get('class', '').lower() not in self.config.HEAVY_VEHICLE_CLASSES:
            return events
        
        # Find cars behind or beside ego in main lane
        rear_cars = []
        for v in vehicles:
            if v['id'] == ego['id']:
                continue
            if v.get('class', '').lower() not in self.config.CAR_CLASSES:
                continue
            if not self.is_in_main_lane(v):
                continue
            
            # Check if behind or beside ego
            dx = v['x'] - ego['x']
            dy = abs(v['y'] - ego['y'])
            
            if dx < 10 and dy < self.config.LANE_WIDTH * 2:  # Behind or adjacent
                rear_cars.append(v)
        
        # Find merging vehicles
        merge_vehicles = [v for v in vehicles if self.is_in_merge_lane(v) and v['id'] != ego['id']]
        
        # Check if ego blocks rear car's view of any merging vehicle
        for rear_car in rear_cars:
            for merge_v in merge_vehicles:
                occ_ratio = self.compute_occlusion_ratio(rear_car, merge_v, ego)
                
                if occ_ratio > 0.3:  # Significant occlusion
                    events.append({
                        'scenario': OcclusionScenario.EGO_BLOCKS_REAR,
                        'occluder': ego,
                        'occluder_id': ego['id'],
                        'blocked': rear_car,          # Can't see
                        'blocked_id': rear_car['id'],
                        'occluded': merge_v,          # Is hidden
                        'occluded_id': merge_v['id'],
                        'occlusion_ratio': occ_ratio,
                        'shadow_polygon': self.compute_shadow_polygon(rear_car, ego),
                    })
        
        return events
    
    def detect_merge_two_way(self, vehicles: List[Dict], trucks: List[Dict]) -> List[Dict]:
        """
        Scenario B: Mutual occlusion between merging and main lane vehicles.
        
        Find cases where:
        - A truck is between merge lane and main lane
        - A merging car can't see main lane vehicles (rear view blocked)
        - Main lane vehicles can't see the merging car
        """
        events = []
        
        merge_vehicles = [v for v in vehicles if self.is_in_merge_lane(v)]
        main_vehicles = [v for v in vehicles if self.is_in_main_lane(v) and 
                        v.get('class', '').lower() in self.config.CAR_CLASSES]
        
        for truck in trucks:
            # Check each merging car
            for merge_v in merge_vehicles:
                if merge_v['id'] == truck['id']:
                    continue
                
                # Check each main lane car
                for main_v in main_vehicles:
                    if main_v['id'] == truck['id']:
                        continue
                    
                    # Check if truck blocks merge car's view of main car
                    occ_merge_to_main = self.compute_occlusion_ratio(merge_v, main_v, truck)
                    
                    # Check if truck blocks main car's view of merge car
                    occ_main_to_merge = self.compute_occlusion_ratio(main_v, merge_v, truck)
                    
                    # Mutual occlusion: both directions have significant blocking
                    if occ_merge_to_main > 0.3 and occ_main_to_merge > 0.3:
                        events.append({
                            'scenario': OcclusionScenario.MERGE_TWO_WAY,
                            'occluder': truck,
                            'occluder_id': truck['id'],
                            'merge_vehicle': merge_v,
                            'merge_vehicle_id': merge_v['id'],
                            'main_vehicle': main_v,
                            'main_vehicle_id': main_v['id'],
                            'occ_merge_to_main': occ_merge_to_main,
                            'occ_main_to_merge': occ_main_to_merge,
                            'shadow_from_merge': self.compute_shadow_polygon(merge_v, truck),
                            'shadow_from_main': self.compute_shadow_polygon(main_v, truck),
                        })
        
        return events
    
    def detect_truck_blocks_merge_rear(self, vehicles: List[Dict], trucks: List[Dict]) -> List[Dict]:
        """
        Variant of Scenario B: Truck blocks merging car's rear view.
        
        The merging sedan can't see main road vehicles behind due to truck.
        """
        events = []
        
        merge_vehicles = [v for v in vehicles if self.is_in_merge_lane(v) and
                         v.get('class', '').lower() in self.config.CAR_CLASSES]
        main_vehicles = [v for v in vehicles if self.is_in_main_lane(v)]
        
        for merge_v in merge_vehicles:
            for truck in trucks:
                if truck['id'] == merge_v['id']:
                    continue
                
                # Truck should be behind or beside the merging car
                dx = truck['x'] - merge_v['x']
                if dx > 5:  # Truck should not be far ahead
                    continue
                
                # Check which main lane vehicles are blocked
                for main_v in main_vehicles:
                    if main_v['id'] == truck['id']:
                        continue
                    
                    occ_ratio = self.compute_occlusion_ratio(merge_v, main_v, truck)
                    
                    if occ_ratio > 0.3:
                        events.append({
                            'scenario': OcclusionScenario.TRUCK_BLOCKS_MERGE,
                            'occluder': truck,
                            'occluder_id': truck['id'],
                            'blocked': merge_v,
                            'blocked_id': merge_v['id'],
                            'occluded': main_v,
                            'occluded_id': main_v['id'],
                            'occlusion_ratio': occ_ratio,
                            'shadow_polygon': self.compute_shadow_polygon(merge_v, truck),
                        })
        
        return events
    
    def detect_all_scenarios(self, ego: Dict, vehicles: List[Dict]) -> Dict[str, List[Dict]]:
        """Detect all occlusion scenarios."""
        all_vehicles = [ego] + vehicles
        trucks = [v for v in all_vehicles if v.get('class', '').lower() in self.config.HEAVY_VEHICLE_CLASSES]
        
        return {
            'ego_blocks_rear': self.detect_ego_blocks_rear(ego, all_vehicles),
            'merge_two_way': self.detect_merge_two_way(all_vehicles, trucks),
            'truck_blocks_merge': self.detect_truck_blocks_merge_rear(all_vehicles, trucks),
        }


# =============================================================================
# Visibility Model
# =============================================================================

class VisibilityModel:
    """Compute visibility weights based on occlusion."""
    
    def __init__(self, config: Config = None):
        self.config = config or Config()
        self.history: Dict[Tuple[int, int], List[float]] = defaultdict(list)
    
    def compute_weights(self, observer: Dict, vehicles: List[Dict],
                        occlusion_events: Dict[str, List[Dict]]) -> Dict[int, Dict]:
        """
        Compute visibility weight for each vehicle from observer's perspective.
        
        The weight affects how much that vehicle contributes to the GVF field.
        """
        visibility_data = {}
        
        # Flatten all events
        all_events = []
        for events in occlusion_events.values():
            all_events.extend(events)
        
        # Also need to directly check occlusions from observer to each vehicle
        occlusion_detector = OcclusionDetector(self.config)
        
        # Find trucks that could be occluders
        all_vehicles = [observer] + vehicles
        trucks = [v for v in all_vehicles if v.get('class', '').lower() in self.config.HEAVY_VEHICLE_CLASSES]
        
        for vehicle in vehicles:
            vid = vehicle['id']
            
            # Find if this vehicle is occluded from observer's view
            max_occlusion = 0.0
            occluder = None
            scenario = OcclusionScenario.NONE
            
            # Check from event logs first
            for event in all_events:
                # Check different event structures
                if event.get('blocked_id') == observer['id'] and event.get('occluded_id') == vid:
                    occ = event.get('occlusion_ratio', 0)
                    if occ > max_occlusion:
                        max_occlusion = occ
                        occluder = event.get('occluder')
                        scenario = event.get('scenario', OcclusionScenario.NONE)
                
                # MERGE_TWO_WAY has different structure
                if event.get('scenario') == OcclusionScenario.MERGE_TWO_WAY:
                    if observer['id'] == event.get('main_vehicle_id') and vid == event.get('merge_vehicle_id'):
                        occ = event.get('occ_main_to_merge', 0)
                        if occ > max_occlusion:
                            max_occlusion = occ
                            occluder = event.get('occluder')
                            scenario = OcclusionScenario.MERGE_TWO_WAY
                    elif observer['id'] == event.get('merge_vehicle_id') and vid == event.get('main_vehicle_id'):
                        occ = event.get('occ_merge_to_main', 0)
                        if occ > max_occlusion:
                            max_occlusion = occ
                            occluder = event.get('occluder')
                            scenario = OcclusionScenario.MERGE_TWO_WAY
            
            # Also directly check: can observer see this vehicle past any truck?
            for truck in trucks:
                if truck['id'] == vid or truck['id'] == observer['id']:
                    continue
                
                occ_ratio = occlusion_detector.compute_occlusion_ratio(observer, vehicle, truck)
                if occ_ratio > max_occlusion:
                    max_occlusion = occ_ratio
                    occluder = truck
                    scenario = OcclusionScenario.EGO_BLOCKS_REAR  # Generic occlusion from ego's view
            
            # Compute visibility weight
            base_visibility = 1.0 - max_occlusion
            
            # Dynamic adjustment based on observer acceleration
            accel = observer.get('ax', 0)
            accel_factor = 1.0 + self.config.VISIBILITY_ACCEL_SENSITIVITY * accel
            accel_factor = np.clip(accel_factor, 0.7, 1.3)
            
            dynamic_visibility = base_visibility * accel_factor
            
            # Temporal smoothing
            key = (observer['id'], vid)
            self.history[key].append(dynamic_visibility)
            if len(self.history[key]) > 10:
                self.history[key] = self.history[key][-10:]
            
            weight = np.mean(self.history[key])
            weight = np.clip(weight, self.config.MIN_VISIBILITY, 1.0)
            
            # Status
            if weight > 0.8:
                status = 'full'
            elif weight > 0.4:
                status = 'partial'
            else:
                status = 'minimal'
            
            visibility_data[vid] = {
                'weight': weight,
                'status': status,
                'occlusion_ratio': max_occlusion,
                'occluder': occluder,
                'scenario': scenario,
            }
        
        return visibility_data


# =============================================================================
# Vehicle Selection
# =============================================================================

def select_surrounding_vehicles(ego: Dict, all_vehicles: List[Dict], 
                                 config: Config) -> List[Dict]:
    """
    Select up to MAX_SURROUNDING_VEHICLES most relevant vehicles.
    
    Priority:
    1. Vehicles involved in occlusion scenarios
    2. Closest vehicles in observation range
    """
    candidates = []
    
    for v in all_vehicles:
        if v['id'] == ego['id']:
            continue
        
        dx = v['x'] - ego['x']
        dy = v['y'] - ego['y']
        dist = np.sqrt(dx**2 + dy**2)
        
        # Check if in observation range
        if not (-config.OBS_RANGE_BEHIND <= dx <= config.OBS_RANGE_AHEAD and
                abs(dy) <= config.OBS_RANGE_LATERAL):
            continue
        
        # Priority score: trucks and merging vehicles get higher priority
        priority = 0
        if v.get('class', '').lower() in config.HEAVY_VEHICLE_CLASSES:
            priority += 100  # Trucks are important
        if v['y'] > config.MAIN_LANE_Y_MAX:
            priority += 50   # Merging vehicles are important
        if abs(dy) < config.LANE_WIDTH:
            priority += 30   # Same lane vehicles
        
        # Closer vehicles get higher priority
        priority += 100 / (dist + 1)
        
        candidates.append((priority, dist, v))
    
    # Sort by priority (descending)
    candidates.sort(key=lambda x: -x[0])
    
    # Return top N
    return [c[2] for c in candidates[:config.MAX_SURROUNDING_VEHICLES]]


# =============================================================================
# GVF Computation
# =============================================================================

def construct_visibility_weighted_gvf(
    ego: Dict,
    vehicles: List[Dict],
    visibility_weights: Dict[int, float],
    x_range: Tuple[float, float],
    y_range: Tuple[float, float],
    config: Config
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Construct visibility-weighted Gaussian Velocity Field."""
    
    nx, ny = config.GVF_GRID_X, config.GVF_GRID_Y
    X = np.linspace(x_range[0], x_range[1], nx)
    Y = np.linspace(y_range[0], y_range[1], ny)
    X_mesh, Y_mesh = np.meshgrid(X, Y)
    
    VX_field = np.zeros_like(X_mesh)
    VY_field = np.zeros_like(Y_mesh)
    Weight_sum = np.ones_like(X_mesh) * 1e-6
    
    cos_h = np.cos(-ego['heading'])
    sin_h = np.sin(-ego['heading'])
    
    for vehicle in vehicles:
        vid = vehicle['id']
        vis_weight = visibility_weights.get(vid, 1.0)
        
        # Relative position
        dx = vehicle['x'] - ego['x']
        dy = vehicle['y'] - ego['y']
        dx_rel = dx * cos_h - dy * sin_h
        dy_rel = dx * sin_h + dy * cos_h
        
        # Relative velocity
        dvx = vehicle.get('vx', 0) - ego.get('vx', 0)
        dvy = vehicle.get('vy', 0) - ego.get('vy', 0)
        dvx_rel = dvx * cos_h - dvy * sin_h
        dvy_rel = dvx * sin_h + dvy * cos_h
        
        # Gaussian kernel
        kernel = np.exp(-((X_mesh - dx_rel)**2 / (2 * config.SIGMA_X**2) +
                         (Y_mesh - dy_rel)**2 / (2 * config.SIGMA_Y**2)))
        
        # Weight by visibility
        weighted_kernel = vis_weight * kernel
        
        VX_field += weighted_kernel * dvx_rel
        VY_field += weighted_kernel * dvy_rel
        Weight_sum += weighted_kernel
    
    VX_field /= Weight_sum
    VY_field /= Weight_sum
    
    return X_mesh, Y_mesh, VX_field, VY_field


# =============================================================================
# Data Loader
# =============================================================================

class ExiDMicroscopicLoader:
    """Load exiD data for microscopic analysis."""
    
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.config = Config()
        self.occlusion_detector = OcclusionDetector(self.config)
        self.visibility_model = VisibilityModel(self.config)
        
        self.tracks_df = None
        self.tracks_meta_df = None
        self.recording_id = None
    
    def load_recording(self, recording_id: int) -> bool:
        """Load recording data."""
        prefix = f"{recording_id:02d}_"
        self.recording_id = recording_id
        
        try:
            self.tracks_df = pd.read_csv(self.data_dir / f"{prefix}tracks.csv")
            self.tracks_meta_df = pd.read_csv(self.data_dir / f"{prefix}tracksMeta.csv")
            
            self.tracks_df = self.tracks_df.merge(
                self.tracks_meta_df[['trackId', 'class', 'width', 'length']],
                on='trackId', how='left', suffixes=('', '_meta')
            )
            
            if 'width_meta' in self.tracks_df.columns:
                self.tracks_df['width'] = self.tracks_df['width'].fillna(self.tracks_df['width_meta'])
                self.tracks_df['length'] = self.tracks_df['length'].fillna(self.tracks_df['length_meta'])
            
            # Infer main lane boundary
            y_vals = self.tracks_df['yCenter'].dropna()
            self.config.MAIN_LANE_Y_MAX = np.percentile(y_vals, 75)
            
            logger.info(f"Loaded recording {recording_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading: {e}")
            return False
    
    def get_frame_data(self, frame: int, ego_id: int) -> Dict:
        """Get frame data with occlusion analysis."""
        frame_data = self.tracks_df[self.tracks_df['frame'] == frame]
        
        if frame_data.empty:
            return {'ego': None, 'vehicles': [], 'occlusions': {}, 'visibility': {}}
        
        all_vehicles = []
        ego = None
        
        for _, row in frame_data.iterrows():
            vclass = str(row.get('class', 'car')).lower()
            
            vehicle = {
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
                'mass': self.config.MASS_HV if vclass in self.config.HEAVY_VEHICLE_CLASSES else self.config.MASS_PC,
            }
            
            all_vehicles.append(vehicle)
            if vehicle['id'] == ego_id:
                ego = vehicle
        
        if ego is None:
            return {'ego': None, 'vehicles': [], 'occlusions': {}, 'visibility': {}}
        
        # Select surrounding vehicles
        surrounding = select_surrounding_vehicles(ego, all_vehicles, self.config)
        
        # Detect occlusions
        occlusions = self.occlusion_detector.detect_all_scenarios(ego, surrounding)
        
        # Compute visibility
        visibility = self.visibility_model.compute_weights(ego, surrounding, occlusions)
        
        return {
            'ego': ego,
            'vehicles': surrounding,
            'occlusions': occlusions,
            'visibility': visibility,
            'frame': frame,
        }
    
    def get_heavy_vehicles(self) -> List[int]:
        """Get list of heavy vehicle IDs."""
        mask = self.tracks_meta_df['class'].str.lower().isin(self.config.HEAVY_VEHICLE_CLASSES)
        return self.tracks_meta_df[mask]['trackId'].tolist()
    
    def find_best_frame(self, ego_id: int) -> int:
        """Find frame with most occlusion events."""
        ego_frames = self.tracks_df[self.tracks_df['trackId'] == ego_id]['frame'].unique()
        
        best_frame = int(np.median(ego_frames))
        best_score = 0
        
        for frame in ego_frames[::10]:
            data = self.get_frame_data(int(frame), ego_id)
            n_occ = sum(len(v) for v in data['occlusions'].values())
            n_veh = len(data['vehicles'])
            score = n_occ * 10 + n_veh
            
            if score > best_score:
                best_score = score
                best_frame = int(frame)
        
        return best_frame


# =============================================================================
# Visualization
# =============================================================================

class MicroscopicVisualizer:
    """Visualize microscopic occlusion-aware GVF."""
    
    def __init__(self, config: Config = None):
        self.config = config or Config()
    
    def create_figure(self, data: Dict, output_path: Optional[str] = None):
        """Create comparison figure."""
        ego = data['ego']
        vehicles = data['vehicles']
        visibility = data['visibility']
        occlusions = data['occlusions']
        frame = data['frame']
        
        if ego is None:
            logger.error("No ego vehicle")
            return
        
        vis_weights = {vid: d['weight'] for vid, d in visibility.items()}
        
        fig = plt.figure(figsize=(20, 10))
        fig.patch.set_facecolor(self.config.BG_DARK)
        
        gs = fig.add_gridspec(2, 3, width_ratios=[1.2, 1.2, 0.7],
                              height_ratios=[1, 1], hspace=0.25, wspace=0.2)
        
        # Row 1
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_gvf(ax1, ego, vehicles, {v['id']: 1.0 for v in vehicles},
                       "Standard GVF (All w=1.0)", visibility_data=None)
        
        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_gvf(ax2, ego, vehicles, vis_weights,
                       "Occlusion-Aware GVF", visibility_data=visibility)
        
        ax3 = fig.add_subplot(gs[0, 2])
        self._plot_scenario_legend(ax3, ego, vehicles, visibility, occlusions)
        
        # Row 2
        ax4 = fig.add_subplot(gs[1, 0])
        self._plot_occlusion_geometry(ax4, ego, vehicles, visibility, occlusions)
        
        ax5 = fig.add_subplot(gs[1, 1])
        self._plot_field_difference(ax5, ego, vehicles, vis_weights)
        
        ax6 = fig.add_subplot(gs[1, 2])
        self._plot_visibility_bars(ax6, vehicles, visibility)
        
        # Title
        n_occ = sum(len(v) for v in occlusions.values())
        n_affected = sum(1 for d in visibility.values() if d['status'] != 'full')
        fig.suptitle(
            f"Microscopic Occlusion Analysis | Frame: {frame} | "
            f"Ego: {ego['class'].title()} (ID:{ego['id']}) | "
            f"Vehicles: {len(vehicles)}/{self.config.MAX_SURROUNDING_VEHICLES} | "
            f"Occlusion Events: {n_occ} | Affected: {n_affected}",
            fontsize=13, fontweight='bold', color='white', y=0.98
        )
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        if output_path:
            fig.savefig(output_path, dpi=150, bbox_inches='tight',
                       facecolor=fig.get_facecolor())
            logger.info(f"Saved: {output_path}")
            plt.close(fig)
        
        return fig
    
    def _compute_range(self, ego: Dict, vehicles: List[Dict]) -> Tuple[Tuple, Tuple]:
        """Compute plot range in ego frame."""
        cos_h = np.cos(-ego['heading'])
        sin_h = np.sin(-ego['heading'])
        
        rel_x, rel_y = [0], [0]
        for v in vehicles:
            dx = v['x'] - ego['x']
            dy = v['y'] - ego['y']
            rel_x.append(dx * cos_h - dy * sin_h)
            rel_y.append(dx * sin_h + dy * cos_h)
        
        margin = 12
        x_range = (min(-35, min(rel_x) - margin), max(55, max(rel_x) + margin))
        y_range = (min(-12, min(rel_y) - margin), max(12, max(rel_y) + margin))
        
        return x_range, y_range
    
    def _plot_gvf(self, ax, ego: Dict, vehicles: List[Dict],
                  vis_weights: Dict[int, float], title: str,
                  visibility_data: Optional[Dict] = None):
        """Plot GVF field."""
        ax.set_facecolor(self.config.BG_PANEL)
        
        x_range, y_range = self._compute_range(ego, vehicles)
        
        X, Y, VX, VY = construct_visibility_weighted_gvf(
            ego, vehicles, vis_weights, x_range, y_range, self.config
        )
        
        V_mag = np.sqrt(VX**2 + VY**2)
        
        pcm = ax.pcolormesh(X, Y, V_mag, cmap='viridis', shading='gouraud', alpha=0.85)
        
        # Quiver
        skip = 4
        ax.quiver(X[::skip, ::skip], Y[::skip, ::skip],
                 VX[::skip, ::skip], VY[::skip, ::skip],
                 color='white', alpha=0.5, scale=120, width=0.003)
        
        # Draw vehicles
        self._draw_vehicles(ax, ego, vehicles, vis_weights, visibility_data)
        
        # Lane markings
        for ly in [-self.config.LANE_WIDTH, 0, self.config.LANE_WIDTH]:
            ax.axhline(ly, color='white', linestyle='--', alpha=0.3, linewidth=0.8)
        ax.axhline(self.config.MAIN_LANE_Y_MAX - ego['y'], color='yellow', 
                   linestyle='-', alpha=0.5, linewidth=1.5)
        
        ax.set_xlim(x_range)
        ax.set_ylim(y_range)
        ax.set_xlabel('Longitudinal (m)', color='white', fontsize=9)
        ax.set_ylabel('Lateral (m)', color='white', fontsize=9)
        ax.set_title(title, fontsize=10, fontweight='bold', color='white')
        ax.tick_params(colors='white', labelsize=8)
        ax.set_aspect('equal')
        
        cbar = plt.colorbar(pcm, ax=ax, shrink=0.6, pad=0.02)
        cbar.set_label('|V| (m/s)', color='white', fontsize=8)
        cbar.ax.tick_params(colors='white', labelsize=7)
        
        for spine in ax.spines.values():
            spine.set_color(self.config.SPINE_COLOR)
    
    def _draw_vehicles(self, ax, ego: Dict, vehicles: List[Dict],
                       vis_weights: Dict[int, float],
                       visibility_data: Optional[Dict] = None):
        """Draw vehicles with occlusion styling."""
        cos_h = np.cos(-ego['heading'])
        sin_h = np.sin(-ego['heading'])
        
        # Ego at origin
        ego_rect = FancyBboxPatch(
            (-ego['length']/2, -ego['width']/2),
            ego['length'], ego['width'],
            boxstyle="round,pad=0.02",
            facecolor=self.config.COLORS['ego'],
            edgecolor='white', linewidth=2.5
        )
        ax.add_patch(ego_rect)
        ax.text(0, -ego['width']/2 - 1.8, f"EGO\n({ego['id']})",
               ha='center', va='top', fontsize=8, color='white', fontweight='bold')
        
        # Other vehicles
        for v in vehicles:
            vid = v['id']
            weight = vis_weights.get(vid, 1.0)
            
            dx = v['x'] - ego['x']
            dy = v['y'] - ego['y']
            dx_rel = dx * cos_h - dy * sin_h
            dy_rel = dx * sin_h + dy * cos_h
            
            # Determine styling
            is_truck = v.get('class', '').lower() in self.config.HEAVY_VEHICLE_CLASSES
            
            if visibility_data and vid in visibility_data:
                status = visibility_data[vid]['status']
                scenario = visibility_data[vid].get('scenario', OcclusionScenario.NONE)
            else:
                status = 'full'
                scenario = OcclusionScenario.NONE
            
            # Color and style
            if is_truck:
                facecolor = self.config.COLORS['truck']
            elif status == 'minimal':
                facecolor = self.config.COLORS['occluded']
            elif status == 'partial':
                facecolor = '#E67E22'
            else:
                facecolor = self.config.COLORS['car']
            
            if status == 'full':
                edgecolor = 'white'
                linestyle = '-'
                linewidth = 1
            elif status == 'partial':
                edgecolor = 'yellow'
                linestyle = '--'
                linewidth = 1.5
            else:
                edgecolor = 'red'
                linestyle = '--'
                linewidth = 2
            
            alpha = max(0.4, weight)
            
            rect = FancyBboxPatch(
                (dx_rel - v['length']/2, dy_rel - v['width']/2),
                v['length'], v['width'],
                boxstyle="round,pad=0.02",
                facecolor=facecolor,
                edgecolor=edgecolor,
                linewidth=linewidth,
                linestyle=linestyle,
                alpha=alpha
            )
            ax.add_patch(rect)
            
            # Label
            label = f"{vid}"
            if weight < 1.0:
                label += f"\nw={weight:.2f}"
            
            label_color = 'red' if status == 'minimal' else ('yellow' if status == 'partial' else 'white')
            ax.text(dx_rel, dy_rel + v['width']/2 + 1.2, label,
                   ha='center', va='bottom', fontsize=7, color=label_color)
    
    def _plot_scenario_legend(self, ax, ego: Dict, vehicles: List[Dict],
                               visibility: Dict, occlusions: Dict):
        """Plot scenario explanation."""
        ax.set_facecolor(self.config.BG_PANEL)
        ax.axis('off')
        
        lines = [
            "OCCLUSION SCENARIOS",
            "=" * 28,
            "",
            "A) EGO_BLOCKS_REAR:",
            f"   Events: {len(occlusions.get('ego_blocks_rear', []))}",
            "   Ego truck blocks rear car's",
            "   view of merging vehicles",
            "",
            "B) MERGE_TWO_WAY:",
            f"   Events: {len(occlusions.get('merge_two_way', []))}",
            "   Mutual occlusion: merge &",
            "   main lane can't see each other",
            "",
            "C) TRUCK_BLOCKS_MERGE:",
            f"   Events: {len(occlusions.get('truck_blocks_merge', []))}",
            "   Truck blocks merging car's",
            "   rear view of main lane",
            "",
            "-" * 28,
            "VISIBILITY STATUS:",
        ]
        
        for v in vehicles:
            vid = v['id']
            data = visibility.get(vid, {'weight': 1.0, 'status': 'full'})
            icon = {'full': '●', 'partial': '◐', 'minimal': '○'}[data['status']]
            vclass = 'T' if v['class'] in self.config.HEAVY_VEHICLE_CLASSES else 'C'
            lines.append(f"  {icon} {vid}({vclass}): w={data['weight']:.2f}")
        
        ax.text(0.05, 0.95, "\n".join(lines), transform=ax.transAxes,
               fontsize=8, color='white', family='monospace',
               verticalalignment='top')
        
        for spine in ax.spines.values():
            spine.set_color(self.config.SPINE_COLOR)
    
    def _plot_occlusion_geometry(self, ax, ego: Dict, vehicles: List[Dict],
                                  visibility: Dict, occlusions: Dict):
        """Plot occlusion geometry with shadow zones."""
        ax.set_facecolor(self.config.BG_PANEL)
        
        x_range, y_range = self._compute_range(ego, vehicles)
        cos_h = np.cos(-ego['heading'])
        sin_h = np.sin(-ego['heading'])
        
        # Draw shadow zones
        all_events = []
        for events in occlusions.values():
            all_events.extend(events)
        
        for event in all_events:
            shadow = None
            if 'shadow_polygon' in event and event['shadow_polygon'] is not None:
                shadow = event['shadow_polygon']
            elif 'shadow_from_merge' in event and event['shadow_from_merge'] is not None:
                shadow = event['shadow_from_merge']
            
            if shadow is not None and len(shadow) > 0:
                # Transform to ego frame
                shadow_rel = []
                for pt in shadow:
                    dx = pt[0] - ego['x']
                    dy = pt[1] - ego['y']
                    shadow_rel.append([dx * cos_h - dy * sin_h, dx * sin_h + dy * cos_h])
                
                shadow_patch = Polygon(shadow_rel, alpha=self.config.SHADOW_ALPHA,
                                       facecolor=self.config.SHADOW_COLOR,
                                       edgecolor='red', linestyle='--', linewidth=1)
                ax.add_patch(shadow_patch)
        
        # Draw vehicles
        vis_weights = {vid: d['weight'] for vid, d in visibility.items()}
        self._draw_vehicles(ax, ego, vehicles, vis_weights, visibility)
        
        # Draw occlusion arrows
        for event in all_events:
            if 'blocked' in event and 'occluded' in event:
                blocked = event['blocked']
                occluded = event['occluded']
                
                # Transform positions
                b_dx = blocked['x'] - ego['x']
                b_dy = blocked['y'] - ego['y']
                b_rel = (b_dx * cos_h - b_dy * sin_h, b_dx * sin_h + b_dy * cos_h)
                
                o_dx = occluded['x'] - ego['x']
                o_dy = occluded['y'] - ego['y']
                o_rel = (o_dx * cos_h - o_dy * sin_h, o_dx * sin_h + o_dy * cos_h)
                
                # Draw "can't see" arrow
                ax.annotate('', xy=o_rel, xytext=b_rel,
                           arrowprops=dict(arrowstyle='->', color='red',
                                         linestyle='--', alpha=0.6, lw=1.5))
        
        ax.set_xlim(x_range)
        ax.set_ylim(y_range)
        ax.set_xlabel('Longitudinal (m)', color='white', fontsize=9)
        ax.set_ylabel('Lateral (m)', color='white', fontsize=9)
        ax.set_title("Occlusion Geometry (Red = Shadow Zones)", fontsize=10,
                    fontweight='bold', color='white')
        ax.tick_params(colors='white', labelsize=8)
        ax.set_aspect('equal')
        
        for spine in ax.spines.values():
            spine.set_color(self.config.SPINE_COLOR)
    
    def _plot_field_difference(self, ax, ego: Dict, vehicles: List[Dict],
                                vis_weights: Dict[int, float]):
        """Plot difference between standard and occlusion-aware GVF."""
        ax.set_facecolor(self.config.BG_PANEL)
        
        x_range, y_range = self._compute_range(ego, vehicles)
        
        # Standard
        X1, Y1, VX1, VY1 = construct_visibility_weighted_gvf(
            ego, vehicles, {v['id']: 1.0 for v in vehicles}, x_range, y_range, self.config
        )
        V_std = np.sqrt(VX1**2 + VY1**2)
        
        # Occlusion-aware
        X2, Y2, VX2, VY2 = construct_visibility_weighted_gvf(
            ego, vehicles, vis_weights, x_range, y_range, self.config
        )
        V_occ = np.sqrt(VX2**2 + VY2**2)
        
        diff = V_std - V_occ
        max_diff = max(abs(diff.min()), abs(diff.max()), 0.1)
        
        cmap = LinearSegmentedColormap.from_list('diff', ['#3498DB', '#1A1A2E', '#E74C3C'])
        pcm = ax.pcolormesh(X1, Y1, diff, cmap=cmap, shading='gouraud',
                           vmin=-max_diff, vmax=max_diff, alpha=0.85)
        
        self._draw_vehicles(ax, ego, vehicles, vis_weights, None)
        
        ax.set_xlim(x_range)
        ax.set_ylim(y_range)
        ax.set_xlabel('Longitudinal (m)', color='white', fontsize=9)
        ax.set_ylabel('Lateral (m)', color='white', fontsize=9)
        ax.set_title("Field Difference (Std - OccAware)\nRed=Overestimated", fontsize=10,
                    fontweight='bold', color='white')
        ax.tick_params(colors='white', labelsize=8)
        ax.set_aspect('equal')
        
        cbar = plt.colorbar(pcm, ax=ax, shrink=0.6, pad=0.02)
        cbar.set_label('ΔV (m/s)', color='white', fontsize=8)
        cbar.ax.tick_params(colors='white', labelsize=7)
        
        for spine in ax.spines.values():
            spine.set_color(self.config.SPINE_COLOR)
    
    def _plot_visibility_bars(self, ax, vehicles: List[Dict], visibility: Dict):
        """Plot visibility weight bar chart."""
        ax.set_facecolor(self.config.BG_PANEL)
        
        if not vehicles:
            ax.text(0.5, 0.5, 'No vehicles', ha='center', va='center', color='white')
            ax.axis('off')
            return
        
        vids = [v['id'] for v in vehicles]
        weights = [visibility.get(vid, {'weight': 1.0})['weight'] for vid in vids]
        statuses = [visibility.get(vid, {'status': 'full'})['status'] for vid in vids]
        
        colors = []
        for s in statuses:
            if s == 'full':
                colors.append('#27AE60')
            elif s == 'partial':
                colors.append('#F39C12')
            else:
                colors.append('#E74C3C')
        
        y_pos = np.arange(len(vids))
        bars = ax.barh(y_pos, weights, color=colors, edgecolor='white', alpha=0.8)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels([f"ID:{vid}" for vid in vids], color='white', fontsize=8)
        ax.set_xlim(0, 1.1)
        ax.set_xlabel('Visibility Weight', color='white', fontsize=9)
        ax.set_title('Per-Vehicle Visibility', fontsize=10, fontweight='bold', color='white')
        ax.tick_params(colors='white', labelsize=8)
        
        # Add value labels
        for i, (bar, w) in enumerate(zip(bars, weights)):
            ax.text(w + 0.02, i, f'{w:.2f}', va='center', color='white', fontsize=8)
        
        # Reference line
        ax.axvline(1.0, color='white', linestyle='--', alpha=0.3)
        
        for spine in ax.spines.values():
            spine.set_color(self.config.SPINE_COLOR)
    
    def create_animation(self, loader, ego_id: int, frames: List[int],
                         output_path: str, fps: int = 10):
        """Create animation of occlusion-aware GVF over time with FIXED view."""
        
        logger.info(f"Creating animation with {len(frames)} frames...")
        
        # Precompute all frame data
        precomputed = []
        for frame in frames:
            data = loader.get_frame_data(frame, ego_id)
            if data['ego'] is not None:
                precomputed.append(data)
        
        if not precomputed:
            logger.error("No valid frames")
            return
        
        # Compute FIXED view range from all frames (global extent)
        all_x, all_y = [], []
        for data in precomputed:
            ego = data['ego']
            cos_h = np.cos(-ego['heading'])
            sin_h = np.sin(-ego['heading'])
            for v in data['vehicles']:
                dx = v['x'] - ego['x']
                dy = v['y'] - ego['y']
                all_x.append(dx * cos_h - dy * sin_h)
                all_y.append(dx * sin_h + dy * cos_h)
        
        # Fixed range for entire animation
        margin = 15
        fixed_x_range = (-40, 70)
        fixed_y_range = (-15, 15)
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.patch.set_facecolor(self.config.BG_DARK)
        plt.subplots_adjust(left=0.05, right=0.95, top=0.88, bottom=0.1, wspace=0.15)
        
        def update(idx):
            for ax in axes:
                ax.clear()
            
            data = precomputed[idx]
            ego = data['ego']
            vehicles = data['vehicles']
            visibility = data['visibility']
            vis_weights = {vid: d['weight'] for vid, d in visibility.items()}
            frame = data['frame']
            occlusions = data['occlusions']
            
            # Standard GVF with FIXED range
            self._plot_gvf_fixed(axes[0], ego, vehicles, {v['id']: 1.0 for v in vehicles},
                                f"Standard GVF", None, fixed_x_range, fixed_y_range)
            
            # Occlusion-Aware GVF with FIXED range
            self._plot_gvf_fixed(axes[1], ego, vehicles, vis_weights,
                                f"Occlusion-Aware GVF", visibility, fixed_x_range, fixed_y_range)
            
            # Occlusion Geometry with FIXED range
            self._plot_geometry_fixed(axes[2], ego, vehicles, visibility, occlusions,
                                     fixed_x_range, fixed_y_range)
            
            n_occ = sum(len(v) for v in occlusions.values())
            n_affected = sum(1 for d in visibility.values() if d['status'] != 'full')
            fig.suptitle(
                f"Frame: {frame} | Ego: {ego['id']} | "
                f"Vehicles: {len(vehicles)} | Occlusions: {n_occ} | Affected: {n_affected}",
                fontsize=12, fontweight='bold', color='white', y=0.96
            )
            
            return []
        
        anim = FuncAnimation(fig, update, frames=len(precomputed),
                            interval=1000//fps, blit=False, repeat=True)
        
        try:
            writer = PillowWriter(fps=fps)
            anim.save(output_path, writer=writer, dpi=100,
                     savefig_kwargs={'facecolor': fig.get_facecolor()})
            logger.info(f"Saved: {output_path}")
        except Exception as e:
            logger.error(f"Animation failed: {e}")
        
        plt.close(fig)
    
    def _plot_gvf_fixed(self, ax, ego: Dict, vehicles: List[Dict],
                        vis_weights: Dict[int, float], title: str,
                        visibility_data: Optional[Dict],
                        x_range: Tuple[float, float], y_range: Tuple[float, float]):
        """Plot GVF with fixed view range."""
        ax.set_facecolor(self.config.BG_PANEL)
        
        X, Y, VX, VY = construct_visibility_weighted_gvf(
            ego, vehicles, vis_weights, x_range, y_range, self.config
        )
        
        V_mag = np.sqrt(VX**2 + VY**2)
        
        pcm = ax.pcolormesh(X, Y, V_mag, cmap='viridis', shading='gouraud', 
                           alpha=0.85, vmin=0, vmax=15)  # Fixed color scale
        
        # Quiver
        skip = 4
        ax.quiver(X[::skip, ::skip], Y[::skip, ::skip],
                 VX[::skip, ::skip], VY[::skip, ::skip],
                 color='white', alpha=0.5, scale=120, width=0.003)
        
        # Draw vehicles
        self._draw_vehicles(ax, ego, vehicles, vis_weights, visibility_data)
        
        # Lane markings
        for ly in [-self.config.LANE_WIDTH, 0, self.config.LANE_WIDTH]:
            ax.axhline(ly, color='white', linestyle='--', alpha=0.3, linewidth=0.8)
        ax.axhline(self.config.MAIN_LANE_Y_MAX - ego['y'], color='yellow', 
                   linestyle='-', alpha=0.5, linewidth=1.5)
        
        ax.set_xlim(x_range)
        ax.set_ylim(y_range)
        ax.set_xlabel('Longitudinal (m)', color='white', fontsize=9)
        ax.set_ylabel('Lateral (m)', color='white', fontsize=9)
        ax.set_title(title, fontsize=10, fontweight='bold', color='white')
        ax.tick_params(colors='white', labelsize=8)
        ax.set_aspect('equal')
        
        for spine in ax.spines.values():
            spine.set_color(self.config.SPINE_COLOR)
    
    def _plot_geometry_fixed(self, ax, ego: Dict, vehicles: List[Dict],
                             visibility: Dict, occlusions: Dict,
                             x_range: Tuple[float, float], y_range: Tuple[float, float]):
        """Plot occlusion geometry with fixed view range."""
        ax.set_facecolor(self.config.BG_PANEL)
        
        cos_h = np.cos(-ego['heading'])
        sin_h = np.sin(-ego['heading'])
        
        # Draw shadow zones
        all_events = []
        for events in occlusions.values():
            all_events.extend(events)
        
        for event in all_events:
            shadow = None
            if 'shadow_polygon' in event and event['shadow_polygon'] is not None:
                shadow = event['shadow_polygon']
            elif 'shadow_from_merge' in event and event['shadow_from_merge'] is not None:
                shadow = event['shadow_from_merge']
            
            if shadow is not None and len(shadow) > 0:
                shadow_rel = []
                for pt in shadow:
                    dx = pt[0] - ego['x']
                    dy = pt[1] - ego['y']
                    shadow_rel.append([dx * cos_h - dy * sin_h, dx * sin_h + dy * cos_h])
                
                shadow_patch = Polygon(shadow_rel, alpha=self.config.SHADOW_ALPHA,
                                       facecolor=self.config.SHADOW_COLOR,
                                       edgecolor='red', linestyle='--', linewidth=1)
                ax.add_patch(shadow_patch)
        
        # Draw vehicles
        vis_weights = {vid: d['weight'] for vid, d in visibility.items()}
        self._draw_vehicles(ax, ego, vehicles, vis_weights, visibility)
        
        ax.set_xlim(x_range)
        ax.set_ylim(y_range)
        ax.set_xlabel('Longitudinal (m)', color='white', fontsize=9)
        ax.set_ylabel('Lateral (m)', color='white', fontsize=9)
        ax.set_title("Occlusion Geometry", fontsize=10, fontweight='bold', color='white')
        ax.tick_params(colors='white', labelsize=8)
        ax.set_aspect('equal')
        
        for spine in ax.spines.values():
            spine.set_color(self.config.SPINE_COLOR)


# =============================================================================
# Demo
# =============================================================================

def create_demo_scenario() -> Tuple[Dict, List[Dict]]:
    """
    Create demo scenario illustrating both occlusion types:
    
    A) Ego truck blocks rear car's view of merging vehicle
    B) Mutual occlusion: merge car & main car can't see each other
    """
    
    # Ego truck in main lane
    ego = {
        'id': 1,
        'x': 50.0,
        'y': 0.0,
        'vx': 22.0,
        'vy': 0.0,
        'ax': 0.2,
        'ay': 0.0,
        'heading': 0.0,
        'speed': 22.0,
        'length': 12.0,
        'width': 2.5,
        'class': 'truck',
        'mass': 15000.0
    }
    
    # 6 surrounding vehicles for microscopic analysis
    vehicles = [
        # Car behind ego (BLOCKED - can't see merging car due to ego)
        {
            'id': 2,
            'x': 35.0,
            'y': 0.0,
            'vx': 24.0,
            'vy': 0.0,
            'ax': 0.0,
            'ay': 0.0,
            'heading': 0.0,
            'speed': 24.0,
            'length': 4.5,
            'width': 1.8,
            'class': 'car',
            'mass': 1500.0
        },
        # Merging car (OCCLUDED from car 2's view by ego truck)
        {
            'id': 3,
            'x': 55.0,
            'y': 7.0,  # In merge lane
            'vx': 18.0,
            'vy': -0.8,  # Moving toward main lane
            'ax': 1.2,
            'ay': 0.0,
            'heading': -0.04,
            'speed': 18.0,
            'length': 4.5,
            'width': 1.8,
            'class': 'car',
            'mass': 1500.0
        },
        # Another truck (causes MERGE_TWO_WAY occlusion)
        {
            'id': 4,
            'x': 75.0,
            'y': 3.5,
            'vx': 20.0,
            'vy': 0.0,
            'ax': 0.0,
            'ay': 0.0,
            'heading': 0.0,
            'speed': 20.0,
            'length': 10.0,
            'width': 2.5,
            'class': 'truck',
            'mass': 12000.0
        },
        # Merged sedan (rear view blocked by truck 4 - can't see car 6)
        {
            'id': 5,
            'x': 85.0,
            'y': 6.5,  # Just merged / merging
            'vx': 19.0,
            'vy': -0.5,
            'ax': 0.8,
            'ay': 0.0,
            'heading': -0.02,
            'speed': 19.0,
            'length': 4.5,
            'width': 1.8,
            'class': 'car',
            'mass': 1500.0
        },
        # Main lane car (can't see sedan 5 due to truck 4 - MUTUAL)
        {
            'id': 6,
            'x': 65.0,
            'y': 0.0,
            'vx': 23.0,
            'vy': 0.0,
            'ax': 0.0,
            'ay': 0.0,
            'heading': 0.0,
            'speed': 23.0,
            'length': 4.5,
            'width': 1.8,
            'class': 'car',
            'mass': 1500.0
        },
        # Car ahead in main lane (visible)
        {
            'id': 7,
            'x': 100.0,
            'y': 0.0,
            'vx': 25.0,
            'vy': 0.0,
            'ax': 0.0,
            'ay': 0.0,
            'heading': 0.0,
            'speed': 25.0,
            'length': 4.5,
            'width': 1.8,
            'class': 'car',
            'mass': 1500.0
        },
    ]
    
    return ego, vehicles


def run_demo():
    """Run demo without exiD data."""
    
    logger.info("=" * 60)
    logger.info("DEMO: Microscopic Occlusion Analysis")
    logger.info("=" * 60)
    
    output_dir = Path('./output_microscopic')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    config = Config()
    config.MAIN_LANE_Y_MAX = 5.0
    
    ego, vehicles = create_demo_scenario()
    
    # Detect and analyze
    detector = OcclusionDetector(config)
    visibility_model = VisibilityModel(config)
    
    # Limit to 6 vehicles
    vehicles = vehicles[:config.MAX_SURROUNDING_VEHICLES]
    
    occlusions = detector.detect_all_scenarios(ego, vehicles)
    visibility = visibility_model.compute_weights(ego, vehicles, occlusions)
    
    # Log results
    logger.info(f"Ego: {ego['class']} (ID:{ego['id']})")
    logger.info(f"Surrounding vehicles: {len(vehicles)}")
    
    for scenario, events in occlusions.items():
        logger.info(f"  {scenario}: {len(events)} events")
        for e in events:
            if 'blocked_id' in e:
                logger.info(f"    - Blocked:{e['blocked_id']} can't see Occluded:{e['occluded_id']} "
                           f"(by {e['occluder_id']}, ratio={e.get('occlusion_ratio', 0):.2f})")
            elif 'merge_vehicle_id' in e:
                logger.info(f"    - Merge:{e['merge_vehicle_id']} <-> Main:{e['main_vehicle_id']} "
                           f"(by {e['occluder_id']})")
    
    logger.info("Visibility weights:")
    for vid, data in visibility.items():
        logger.info(f"  Vehicle {vid}: w={data['weight']:.2f} ({data['status']})")
    
    # Create data dict
    data = {
        'ego': ego,
        'vehicles': vehicles,
        'occlusions': occlusions,
        'visibility': visibility,
        'frame': 100,
    }
    
    # Visualize
    viz = MicroscopicVisualizer(config)
    viz.create_figure(data, str(output_dir / 'microscopic_occlusion_demo.png'))
    
    # Animation with FIXED view
    logger.info("Creating demo animation with fixed view...")
    
    # Fixed view range for entire animation
    fixed_x_range = (-40, 70)
    fixed_y_range = (-15, 15)
    
    frames_data = []
    for t in range(60):
        sim_ego = ego.copy()
        sim_ego['x'] = ego['x'] + ego['vx'] * t * 0.08
        
        sim_vehicles = []
        for v in vehicles:
            sv = v.copy()
            sv['x'] = v['x'] + v['vx'] * t * 0.08
            sv['y'] = v['y'] + v['vy'] * t * 0.08
            sim_vehicles.append(sv)
        
        sim_occ = detector.detect_all_scenarios(sim_ego, sim_vehicles)
        sim_vis = visibility_model.compute_weights(sim_ego, sim_vehicles, sim_occ)
        
        frames_data.append({
            'ego': sim_ego,
            'vehicles': sim_vehicles,
            'occlusions': sim_occ,
            'visibility': sim_vis,
            'frame': 100 + t,
        })
    
    # Create animation with fixed view
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.patch.set_facecolor(config.BG_DARK)
    plt.subplots_adjust(left=0.05, right=0.95, top=0.88, bottom=0.1, wspace=0.15)
    
    def update(idx):
        for ax in axes:
            ax.clear()
        
        data = frames_data[idx]
        ego_t = data['ego']
        vehicles_t = data['vehicles']
        vis = data['visibility']
        vis_w = {vid: d['weight'] for vid, d in vis.items()}
        occ = data['occlusions']
        frame = data['frame']
        
        # Use fixed range methods
        viz._plot_gvf_fixed(axes[0], ego_t, vehicles_t, {v['id']: 1.0 for v in vehicles_t},
                           "Standard GVF", None, fixed_x_range, fixed_y_range)
        viz._plot_gvf_fixed(axes[1], ego_t, vehicles_t, vis_w,
                           "Occlusion-Aware GVF", vis, fixed_x_range, fixed_y_range)
        viz._plot_geometry_fixed(axes[2], ego_t, vehicles_t, vis, occ,
                                fixed_x_range, fixed_y_range)
        
        n_occ = sum(len(v) for v in occ.values())
        n_aff = sum(1 for d in vis.values() if d['status'] != 'full')
        fig.suptitle(
            f"Frame: {frame} | Ego:{ego_t['id']} | Veh:{len(vehicles_t)} | "
            f"Occ:{n_occ} | Affected:{n_aff}",
            fontsize=12, fontweight='bold', color='white', y=0.96
        )
        return []
    
    anim = FuncAnimation(fig, update, frames=len(frames_data), interval=100, blit=False)
    
    anim_path = output_dir / 'microscopic_occlusion_animation.gif'
    try:
        writer = PillowWriter(fps=10)
        anim.save(str(anim_path), writer=writer, dpi=100,
                 savefig_kwargs={'facecolor': fig.get_facecolor()})
        logger.info(f"Saved animation: {anim_path}")
    except Exception as e:
        logger.error(f"Animation failed: {e}")
    
    plt.close(fig)
    
    logger.info("=" * 60)
    logger.info(f"Output saved to: {output_dir}")
    logger.info("=" * 60)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Microscopic Occlusion-Aware GVF Analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Occlusion Scenarios:
  A) EGO_BLOCKS_REAR: Ego truck blocks rear car's view of merge lane
  B) MERGE_TWO_WAY: Mutual occlusion between merge and main lane vehicles
  
Examples:
  python exid_occlusion_gvf_microscopic.py --demo
  python exid_occlusion_gvf_microscopic.py --data_dir ./data --recording 25
  python exid_occlusion_gvf_microscopic.py --data_dir ./data --recording 25 --animate
        """
    )
    
    parser.add_argument('--data_dir', type=str, default=None)
    parser.add_argument('--recording', type=int, default=None)
    parser.add_argument('--ego_id', type=int, default=None)
    parser.add_argument('--frame', type=int, default=None)
    parser.add_argument('--output_dir', type=str, default='./output_microscopic')
    parser.add_argument('--demo', action='store_true', help='Run demo mode')
    parser.add_argument('--animate', action='store_true')
    parser.add_argument('--anim_frames', type=int, default=80)
    parser.add_argument('--anim_step', type=int, default=2)
    parser.add_argument('--fps', type=int, default=10)
    
    args = parser.parse_args()
    
    if args.demo:
        run_demo()
        return
    
    if args.data_dir is None or args.recording is None:
        parser.error("--data_dir and --recording required (or use --demo)")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 60)
    logger.info("Microscopic Occlusion Analysis")
    logger.info("=" * 60)
    
    loader = ExiDMicroscopicLoader(args.data_dir)
    if not loader.load_recording(args.recording):
        return
    
    ego_id = args.ego_id
    if ego_id is None:
        heavy = loader.get_heavy_vehicles()
        if not heavy:
            logger.error("No heavy vehicles found")
            return
        ego_id = heavy[0]
        logger.info(f"Auto-selected ego: {ego_id}")
    
    viz = MicroscopicVisualizer()
    
    if args.animate:
        ego_frames = loader.tracks_df[loader.tracks_df['trackId'] == ego_id]['frame'].unique()
        frames = sorted(ego_frames)[::args.anim_step][:args.anim_frames]
        
        output_path = output_dir / f'microscopic_anim_rec{args.recording}_ego{ego_id}.gif'
        viz.create_animation(loader, ego_id, frames, str(output_path), fps=args.fps)
    else:
        frame = args.frame or loader.find_best_frame(ego_id)
        logger.info(f"Using frame: {frame}")
        
        data = loader.get_frame_data(frame, ego_id)
        
        if data['ego'] is None:
            logger.error(f"Ego {ego_id} not in frame {frame}")
            return
        
        output_path = output_dir / f'microscopic_rec{args.recording}_ego{ego_id}_frame{frame}.png'
        viz.create_figure(data, str(output_path))
    
    logger.info(f"Output: {output_dir}")


if __name__ == '__main__':
    main()