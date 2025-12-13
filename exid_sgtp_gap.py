"""
exiD Dataset: Safety-Guaranteed Gap Generation and Risk Assessment
===================================================================
Implements the risk assessment and gap generation from:
Xu et al. (2025) "SGTP: A Safety-Guaranteed Trajectory Planning Algorithm 
for Autonomous Vehicles Using Gap-Oriented Spatio-Temporal Corridor"
IEEE Transactions on Vehicular Technology

Key Concepts:
1. Elliptical risk field based on longitudinal/lateral dynamics (Eq. 1-6)
2. Rectangular risk field inflation (Eq. 7-9)
3. Safety-guaranteed gap generation between vehicles
4. Spatio-temporal gap distribution map

Mathematical Model:
- Distance vector: r_i = [s_i - s_0, l_i - l_0]
- Longitudinal component: r_si = |r_i| * cos(θ_i)
- Lateral component: r_li = |r_i| * sin(θ_i)
- Safety distances: D_si, D_li based on relative velocity and heading
- Elliptical risk field: L_i = D_si²/r_si² + D_li²/r_li²
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.collections import PatchCollection
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Set
import warnings
import argparse
import logging

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class SGTPConfig:
    """Configuration for SGTP risk assessment and gap generation."""
    
    HEAVY_VEHICLE_CLASSES: Set[str] = field(default_factory=lambda: {'truck', 'bus', 'van', 'trailer'})
    CAR_CLASSES: Set[str] = field(default_factory=lambda: {'car'})
    
    # Observation range
    OBS_RANGE_AHEAD: float = 100.0
    OBS_RANGE_BEHIND: float = 50.0
    OBS_RANGE_LEFT: float = 20.0
    OBS_RANGE_RIGHT: float = 20.0
    
    # Risk assessment parameters
    MIN_DECEL: float = 4.0        # Minimum deceleration (m/s²) - a_min in paper
    DESIRED_GAP: float = 5.0      # Desired gap G (m) between vehicles
    
    # Vehicle dimensions
    TRUCK_LENGTH: float = 12.0
    TRUCK_WIDTH: float = 2.5
    CAR_LENGTH: float = 4.5
    CAR_WIDTH: float = 1.8
    
    # Gap generation
    MIN_GAP_LENGTH: float = 8.0   # Minimum gap length to be considered valid
    LANE_WIDTH: float = 3.5       # Standard lane width
    
    # Time discretization for spatio-temporal map
    TIME_HORIZON: float = 8.0     # Planning horizon (s)
    TIME_STEP: float = 0.5        # Time step for gap sampling (s)
    
    # Grid for visualization
    FIELD_GRID_X: int = 100
    FIELD_GRID_Y: int = 50
    
    # Visualization colors
    COLORS: Dict[str, str] = field(default_factory=lambda: {
        'truck': '#E74C3C',
        'car': '#3498DB',
        'gap': '#2ECC71',
        'risk_high': '#E74C3C',
        'risk_low': '#27AE60',
        'corridor': '#F39C12',
    })


# =============================================================================
# SGTP Risk Assessment Model (Equations 1-9 from paper)
# =============================================================================

def compute_distance_vector(ego: Dict, other: Dict) -> Tuple[float, float, float, float]:
    """
    Compute distance vector between ego and other vehicle (Equations 1-2).
    
    r_i = [s_i - s_0, l_i - l_0]  (in Frenet frame, approximated here)
    r_si = |r_i| * cos(θ_i)  - longitudinal component
    r_li = |r_i| * sin(θ_i)  - lateral component
    
    Returns:
        (r_s, r_l, distance, theta) - longitudinal, lateral components, total distance, angle
    """
    # Position difference
    dx = other['x'] - ego['x']
    dy = other['y'] - ego['y']
    
    # Transform to ego's local frame (approximation of Frenet)
    cos_h = np.cos(-ego['heading'])
    sin_h = np.sin(-ego['heading'])
    
    # Longitudinal (s) and lateral (l) in ego frame
    r_s = dx * cos_h - dy * sin_h  # Along ego's heading
    r_l = dx * sin_h + dy * cos_h  # Perpendicular to ego's heading
    
    distance = np.sqrt(r_s**2 + r_l**2)
    
    # Angle θ_i - heading deviation between vehicles
    theta = other['heading'] - ego['heading']
    # Normalize to [-pi, pi]
    theta = np.arctan2(np.sin(theta), np.cos(theta))
    
    return r_s, r_l, distance, theta


def compute_corner_distances(ego: Dict, other: Dict, config: SGTPConfig) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute distances considering vehicle geometry (Equation 3).
    
    Uses the four corner points of vehicles to get boundary distances.
    
    Returns:
        (r_s_corners, r_l_corners) - arrays of corner distances
    """
    # Get base distance
    r_s, r_l, _, theta = compute_distance_vector(ego, other)
    
    # Vehicle half-dimensions
    a_i = other['length'] / 2  # Half length of other
    b_i = other['width'] / 2   # Half width of other
    a_0 = ego['length'] / 2    # Half length of ego
    b_0 = ego['width'] / 2     # Half width of ego
    
    # Corner offsets for other vehicle (in its local frame)
    # S = {[1,1], [-1,1], [-1,-1], [1,-1]}
    corner_offsets = np.array([
        [1, 1], [-1, 1], [-1, -1], [1, -1]
    ]) * np.array([a_i, b_i])
    
    # Rotation matrix for heading difference
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    R = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
    
    # Transform corner offsets
    rotated_offsets = corner_offsets @ R.T
    
    # Compute corner distances (positive linear function - keep only positive parts)
    r_s_corners = []
    r_l_corners = []
    
    for offset in rotated_offsets:
        # Add ego vehicle dimensions
        r_s_corner = abs(r_s) - a_0 - abs(offset[0])
        r_l_corner = abs(r_l) - b_0 - abs(offset[1])
        
        # poslin function: keep positive, zero otherwise
        r_s_corners.append(max(r_s_corner, 0.1))
        r_l_corners.append(max(r_l_corner, 0.1))
    
    return np.array(r_s_corners), np.array(r_l_corners)


def compute_safety_distances(ego: Dict, other: Dict, config: SGTPConfig) -> Tuple[float, float]:
    """
    Compute longitudinal and lateral safety distances (Equations 4-5).
    
    D_si = [v_0 - sign(cos θ)*(cos θ)² * v_i] * cos θ / (2*a_min) + G
    D_li = [v_0 * sin θ > 0] * (sin θ)² * v_i / (2*a_min)
    
    Returns:
        (D_s, D_l) - longitudinal and lateral safety distances
    """
    v_0 = ego['speed']
    v_i = other['speed']
    
    # Heading difference
    theta = other['heading'] - ego['heading']
    theta = np.arctan2(np.sin(theta), np.cos(theta))
    
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    
    # Longitudinal safety distance (Equation 4)
    # D_si = [v_0 - sign(cos θ)*(cos θ)² * v_i] * |cos θ| / (2*a_min) + G
    sign_cos = np.sign(cos_t) if abs(cos_t) > 0.01 else 0
    
    v_term = v_0 - sign_cos * (cos_t ** 2) * v_i
    D_s = max(0, v_term) * abs(cos_t) / (2 * config.MIN_DECEL) + config.DESIRED_GAP
    
    # Lateral safety distance (Equation 5)
    # D_li = [v_0 * sin θ > 0] * (sin θ)² * v_i / (2*a_min)
    if v_0 * sin_t > 0:
        D_l = (sin_t ** 2) * v_i / (2 * config.MIN_DECEL)
    else:
        D_l = 0.5  # Minimum lateral safety margin
    
    # Add vehicle dimensions to safety distances
    D_s = D_s + (ego['length'] + other['length']) / 2
    D_l = D_l + (ego['width'] + other['width']) / 2
    
    return max(D_s, 1.0), max(D_l, 0.5)


def compute_elliptical_risk_field(ego: Dict, other: Dict, config: SGTPConfig) -> float:
    """
    Compute elliptical risk field value (Equation 6).
    
    L_i = D_si²/r_si² + D_li²/r_li²
    
    Returns:
        Risk field value (higher = more dangerous)
    """
    r_s, r_l, distance, theta = compute_distance_vector(ego, other)
    
    if distance < 0.1:
        return float('inf')  # Collision
    
    # Safety distances
    D_s, D_l = compute_safety_distances(ego, other, config)
    
    # Avoid division by zero
    r_s = max(abs(r_s), 0.1)
    r_l = max(abs(r_l), 0.1)
    
    # Elliptical risk field (Equation 6)
    L_i = (D_s ** 2) / (r_s ** 2) + (D_l ** 2) / (r_l ** 2)
    
    return L_i


def compute_rectangular_risk_bounds(ego: Dict, other: Dict, config: SGTPConfig) -> Tuple[float, float, float, float]:
    """
    Inflate elliptical risk field to rectangular bounds (Equations 7-9).
    
    â = max(L_i) - s_i  (major axis - longitudinal extent)
    b̂ = max(L_i) - l_i  (minor axis - lateral extent)
    
    Returns:
        (s_min, s_max, l_min, l_max) - rectangular risk bounds in ego frame
    """
    r_s, r_l, distance, theta = compute_distance_vector(ego, other)
    D_s, D_l = compute_safety_distances(ego, other, config)
    
    # Risk field value
    L_i = compute_elliptical_risk_field(ego, other, config)
    
    # Scale factor based on risk (higher risk = larger inflation)
    risk_scale = min(np.sqrt(L_i), 3.0)  # Cap at 3x
    
    # Rectangular bounds centered on other vehicle
    # â and b̂ represent the half-extents of the rectangle
    a_hat = D_s * risk_scale  # Longitudinal half-extent
    b_hat = D_l * risk_scale  # Lateral half-extent
    
    # Bounds in ego frame
    s_min = r_s - a_hat
    s_max = r_s + a_hat
    l_min = r_l - b_hat
    l_max = r_l + b_hat
    
    return s_min, s_max, l_min, l_max


def compute_risk_field_grid(ego: Dict, vehicles: List[Dict], 
                            x_range: Tuple[float, float],
                            y_range: Tuple[float, float],
                            grid_size: Tuple[int, int],
                            config: SGTPConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute risk field over a spatial grid.
    
    Returns:
        (X_mesh, Y_mesh, risk_field)
    """
    nx, ny = grid_size
    X = np.linspace(x_range[0], x_range[1], nx)
    Y = np.linspace(y_range[0], y_range[1], ny)
    X_mesh, Y_mesh = np.meshgrid(X, Y)
    
    risk_field = np.zeros_like(X_mesh)
    
    for other in vehicles:
        # Get rectangular risk bounds
        s_min, s_max, l_min, l_max = compute_rectangular_risk_bounds(ego, other, config)
        
        # Add risk to grid points within bounds
        for i in range(ny):
            for j in range(nx):
                x_local = X_mesh[i, j]
                y_local = Y_mesh[i, j]
                
                # Check if within rectangular risk region
                if s_min <= x_local <= s_max and l_min <= y_local <= l_max:
                    # Compute distance-based risk intensity
                    r_s, r_l, _, _ = compute_distance_vector(ego, other)
                    dx = x_local - r_s
                    dy = y_local - r_l
                    dist_to_vehicle = np.sqrt(dx**2 + dy**2)
                    
                    # Risk decreases with distance from vehicle center
                    D_s, D_l = compute_safety_distances(ego, other, config)
                    norm_dist = dist_to_vehicle / max(D_s, D_l)
                    local_risk = max(0, 1 - norm_dist) * compute_elliptical_risk_field(ego, other, config)
                    
                    risk_field[i, j] += local_risk
    
    return X_mesh, Y_mesh, risk_field


# =============================================================================
# Gap Generation (Algorithm I from paper)
# =============================================================================

@dataclass
class Gap:
    """Represents a drivable gap between vehicles."""
    id: int
    center_s: float       # Longitudinal center in Frenet frame
    center_l: float       # Lateral center (lane center)
    length: float         # Gap length
    width: float          # Gap width (lane width)
    lane_id: int          # Lane identifier
    front_vehicle_id: Optional[int] = None
    rear_vehicle_id: Optional[int] = None
    time_step: int = 0    # Time step index for spatio-temporal map
    
    def contains_point(self, s: float, l: float) -> bool:
        """Check if a point is within this gap."""
        s_min = self.center_s - self.length / 2
        s_max = self.center_s + self.length / 2
        l_min = self.center_l - self.width / 2
        l_max = self.center_l + self.width / 2
        return s_min <= s <= s_max and l_min <= l <= l_max


def assign_lanes(ego: Dict, vehicles: List[Dict], config: SGTPConfig) -> Dict[int, List[Dict]]:
    """
    Assign vehicles to lanes based on lateral position.
    
    Returns:
        Dictionary mapping lane_id to list of vehicles in that lane
    """
    lanes = {}
    
    # Determine lane boundaries based on ego position
    ego_lane_center = 0  # Ego is at lane center (l=0 in Frenet)
    
    for v in vehicles:
        r_s, r_l, _, _ = compute_distance_vector(ego, v)
        
        # Assign to lane based on lateral position
        lane_id = int(round(r_l / config.LANE_WIDTH))
        
        if lane_id not in lanes:
            lanes[lane_id] = []
        
        v_with_frenet = v.copy()
        v_with_frenet['s'] = r_s
        v_with_frenet['l'] = r_l
        lanes[lane_id].append(v_with_frenet)
    
    # Sort vehicles in each lane by longitudinal position
    for lane_id in lanes:
        lanes[lane_id].sort(key=lambda v: v['s'])
    
    return lanes


def generate_gaps(ego: Dict, vehicles: List[Dict], config: SGTPConfig,
                  time_step: int = 0) -> List[Gap]:
    """
    Generate safety-guaranteed gaps (Algorithm I from paper).
    
    Args:
        ego: Ego vehicle state
        vehicles: List of surrounding vehicles
        config: Configuration
        time_step: Current time step for spatio-temporal indexing
    
    Returns:
        List of Gap objects
    """
    gaps = []
    gap_id = 0
    
    # Assign vehicles to lanes
    lanes = assign_lanes(ego, vehicles, config)
    
    # Add ego's lane if not present
    if 0 not in lanes:
        lanes[0] = []
    
    # Process each lane
    for lane_id in sorted(lanes.keys()):
        lane_vehicles = lanes[lane_id]
        lane_center = lane_id * config.LANE_WIDTH
        
        # Compute inflated occupied regions for vehicles in this lane
        occupied_regions = []
        
        for v in lane_vehicles:
            s_min, s_max, l_min, l_max = compute_rectangular_risk_bounds(ego, v, config)
            occupied_regions.append({
                'vehicle': v,
                's_min': s_min,
                's_max': s_max,
                'vehicle_id': v['id']
            })
        
        # Sort by longitudinal position
        occupied_regions.sort(key=lambda r: r['s_min'])
        
        # Generate gaps between occupied regions
        # Gap at rear (behind all vehicles)
        if occupied_regions:
            first_region = occupied_regions[0]
            if first_region['s_min'] > -config.OBS_RANGE_BEHIND:
                gap_length = first_region['s_min'] - (-config.OBS_RANGE_BEHIND)
                if gap_length >= config.MIN_GAP_LENGTH:
                    gaps.append(Gap(
                        id=gap_id,
                        center_s=(-config.OBS_RANGE_BEHIND + first_region['s_min']) / 2,
                        center_l=lane_center,
                        length=gap_length,
                        width=config.LANE_WIDTH,
                        lane_id=lane_id,
                        front_vehicle_id=first_region['vehicle_id'],
                        rear_vehicle_id=None,
                        time_step=time_step
                    ))
                    gap_id += 1
        
        # Gaps between vehicles
        for i in range(len(occupied_regions) - 1):
            rear_region = occupied_regions[i]
            front_region = occupied_regions[i + 1]
            
            gap_start = rear_region['s_max']
            gap_end = front_region['s_min']
            gap_length = gap_end - gap_start
            
            if gap_length >= config.MIN_GAP_LENGTH:
                gaps.append(Gap(
                    id=gap_id,
                    center_s=(gap_start + gap_end) / 2,
                    center_l=lane_center,
                    length=gap_length,
                    width=config.LANE_WIDTH,
                    lane_id=lane_id,
                    front_vehicle_id=front_region['vehicle_id'],
                    rear_vehicle_id=rear_region['vehicle_id'],
                    time_step=time_step
                ))
                gap_id += 1
        
        # Gap at front (ahead of all vehicles)
        if occupied_regions:
            last_region = occupied_regions[-1]
            if last_region['s_max'] < config.OBS_RANGE_AHEAD:
                gap_length = config.OBS_RANGE_AHEAD - last_region['s_max']
                if gap_length >= config.MIN_GAP_LENGTH:
                    gaps.append(Gap(
                        id=gap_id,
                        center_s=(last_region['s_max'] + config.OBS_RANGE_AHEAD) / 2,
                        center_l=lane_center,
                        length=gap_length,
                        width=config.LANE_WIDTH,
                        lane_id=lane_id,
                        front_vehicle_id=None,
                        rear_vehicle_id=last_region['vehicle_id'],
                        time_step=time_step
                    ))
                    gap_id += 1
        else:
            # Empty lane - entire lane is a gap
            gaps.append(Gap(
                id=gap_id,
                center_s=0,
                center_l=lane_center,
                length=config.OBS_RANGE_AHEAD + config.OBS_RANGE_BEHIND,
                width=config.LANE_WIDTH,
                lane_id=lane_id,
                front_vehicle_id=None,
                rear_vehicle_id=None,
                time_step=time_step
            ))
            gap_id += 1
    
    return gaps


def generate_spatio_temporal_gaps(ego: Dict, vehicles: List[Dict], 
                                   config: SGTPConfig) -> Dict[int, List[Gap]]:
    """
    Generate spatio-temporal gap distribution map.
    
    Predicts gap evolution over time horizon using constant velocity assumption.
    
    Returns:
        Dictionary mapping time_step to list of gaps
    """
    n_steps = int(config.TIME_HORIZON / config.TIME_STEP) + 1
    st_gaps = {}
    
    for t_idx in range(n_steps):
        t = t_idx * config.TIME_STEP
        
        # Predict vehicle positions at time t
        predicted_vehicles = []
        for v in vehicles:
            pred_v = v.copy()
            pred_v['x'] = v['x'] + v['vx'] * t
            pred_v['y'] = v['y'] + v['vy'] * t
            predicted_vehicles.append(pred_v)
        
        # Predict ego position
        pred_ego = ego.copy()
        pred_ego['x'] = ego['x'] + ego['vx'] * t
        pred_ego['y'] = ego['y'] + ego['vy'] * t
        
        # Generate gaps at this time step
        gaps = generate_gaps(pred_ego, predicted_vehicles, config, time_step=t_idx)
        st_gaps[t_idx] = gaps
    
    return st_gaps


# =============================================================================
# Data Loader
# =============================================================================

class ExiDLoader:
    """Load exiD data for SGTP analysis."""
    
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.config = SGTPConfig()
        self.tracks_df = None
        self.tracks_meta_df = None
        self.background_image = None
        self.ortho_px_to_meter = 0.1
    
    def load_recording(self, recording_id: int) -> bool:
        prefix = f"{recording_id:02d}_"
        
        try:
            self.tracks_df = pd.read_csv(self.data_dir / f"{prefix}tracks.csv")
            self.tracks_meta_df = pd.read_csv(self.data_dir / f"{prefix}tracksMeta.csv")
            
            rec_meta_path = self.data_dir / f"{prefix}recordingMeta.csv"
            if rec_meta_path.exists():
                rec_meta_df = pd.read_csv(rec_meta_path)
                if not rec_meta_df.empty:
                    self.ortho_px_to_meter = float(rec_meta_df.iloc[0].get('orthoPxToMeter', 0.1))
            
            bg_path = self.data_dir / f"{prefix}background.png"
            if bg_path.exists():
                self.background_image = plt.imread(str(bg_path))
            
            self.tracks_df = self.tracks_df.merge(
                self.tracks_meta_df[['trackId', 'class', 'width', 'length']],
                on='trackId', how='left', suffixes=('', '_meta')
            )
            
            if 'width_meta' in self.tracks_df.columns:
                self.tracks_df['width'] = self.tracks_df['width'].fillna(self.tracks_df['width_meta'])
                self.tracks_df['length'] = self.tracks_df['length'].fillna(self.tracks_df['length_meta'])
            
            logger.info(f"Loaded recording {recording_id}")
            return True
        except Exception as e:
            logger.error(f"Error: {e}")
            return False
    
    def get_snapshot(self, ego_id: int, frame: int, heading_tol_deg: float = 90.0) -> Optional[Dict]:
        """Get snapshot of ego and surrounding vehicles."""
        
        frame_data = self.tracks_df[self.tracks_df['frame'] == frame]
        if frame_data.empty:
            return None
        
        ego_row = frame_data[frame_data['trackId'] == ego_id]
        if ego_row.empty:
            return None
        
        ego_row = ego_row.iloc[0]
        vclass = str(ego_row.get('class', 'car')).lower()
        
        ego = {
            'id': ego_id,
            'x': float(ego_row['xCenter']),
            'y': float(ego_row['yCenter']),
            'heading': np.radians(float(ego_row.get('heading', 0))),
            'vx': float(ego_row.get('xVelocity', 0)),
            'vy': float(ego_row.get('yVelocity', 0)),
            'ax': float(ego_row.get('xAcceleration', 0)),
            'ay': float(ego_row.get('yAcceleration', 0)),
            'speed': np.sqrt(ego_row.get('xVelocity', 0)**2 + ego_row.get('yVelocity', 0)**2),
            'width': float(ego_row.get('width', 2.5)),
            'length': float(ego_row.get('length', 12.0)),
            'class': vclass,
        }
        
        surrounding = []
        for _, row in frame_data.iterrows():
            if row['trackId'] == ego_id:
                continue
            
            other_class = str(row.get('class', 'car')).lower()
            other = {
                'id': int(row['trackId']),
                'x': float(row['xCenter']),
                'y': float(row['yCenter']),
                'heading': np.radians(float(row.get('heading', 0))),
                'vx': float(row.get('xVelocity', 0)),
                'vy': float(row.get('yVelocity', 0)),
                'ax': float(row.get('xAcceleration', 0)),
                'ay': float(row.get('yAcceleration', 0)),
                'speed': np.sqrt(row.get('xVelocity', 0)**2 + row.get('yVelocity', 0)**2),
                'width': float(row.get('width', 1.8)),
                'length': float(row.get('length', 4.5)),
                'class': other_class,
            }
            
            # Check range
            dx = other['x'] - ego['x']
            dy = other['y'] - ego['y']
            
            if (-self.config.OBS_RANGE_BEHIND <= dx <= self.config.OBS_RANGE_AHEAD and
                -self.config.OBS_RANGE_RIGHT <= dy <= self.config.OBS_RANGE_LEFT):
                # Check heading tolerance
                d_heading = abs(np.arctan2(np.sin(other['heading'] - ego['heading']),
                                          np.cos(other['heading'] - ego['heading'])))
                if d_heading <= np.radians(heading_tol_deg):
                    surrounding.append(other)
        
        return {'ego': ego, 'surrounding': surrounding, 'frame': frame}
    
    def find_best_frame(self, ego_id: int, min_vehicles: int = 5) -> Optional[int]:
        """Find frame with good interaction density."""
        ego_data = self.tracks_df[self.tracks_df['trackId'] == ego_id]
        if ego_data.empty:
            return None
        
        frames = ego_data['frame'].values
        best_frame, best_count = None, -1
        
        for frame in frames[::5]:
            snapshot = self.get_snapshot(ego_id, frame)
            if snapshot and len(snapshot['surrounding']) > best_count:
                best_count = len(snapshot['surrounding'])
                best_frame = frame
            if best_count >= min_vehicles:
                break
        
        return best_frame if best_frame else int(np.median(frames))
    
    def get_heavy_vehicles(self) -> List[int]:
        mask = self.tracks_meta_df['class'].str.lower().isin(self.config.HEAVY_VEHICLE_CLASSES)
        return self.tracks_meta_df[mask]['trackId'].tolist()


# =============================================================================
# Visualization
# =============================================================================

class SGTPVisualizer:
    """Visualizer for SGTP risk assessment and gap generation."""
    
    def __init__(self, config: SGTPConfig = None, loader: ExiDLoader = None):
        self.config = config or SGTPConfig()
        self.loader = loader
    
    def create_analysis_figure(self, snapshot: Dict, output_path: str = None):
        """Create comprehensive SGTP analysis figure."""
        
        ego = snapshot['ego']
        surrounding = snapshot['surrounding']
        
        if not surrounding:
            logger.warning("No surrounding vehicles")
            return
        
        fig = plt.figure(figsize=(24, 16))
        fig.patch.set_facecolor('#0D1117')
        
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.25)
        
        ax1 = fig.add_subplot(gs[0, 0])  # Risk field
        ax2 = fig.add_subplot(gs[0, 1])  # Rectangular risk bounds
        ax3 = fig.add_subplot(gs[0, 2])  # Gap distribution (current)
        ax4 = fig.add_subplot(gs[1, :2])  # Spatio-temporal gap map
        ax5 = fig.add_subplot(gs[1, 2])  # Risk assessment details
        ax6 = fig.add_subplot(gs[2, 0])  # Gap statistics
        ax7 = fig.add_subplot(gs[2, 1])  # Safety distances
        ax8 = fig.add_subplot(gs[2, 2])  # Summary
        
        # Compute bounds
        rel_positions = []
        cos_h = np.cos(-ego['heading'])
        sin_h = np.sin(-ego['heading'])
        for v in surrounding:
            dx = v['x'] - ego['x']
            dy = v['y'] - ego['y']
            rel_positions.append([dx * cos_h - dy * sin_h, dx * sin_h + dy * cos_h])
        rel_positions = np.array(rel_positions) if rel_positions else np.array([[0, 0]])
        
        margin = 15
        x_range = (-self.config.OBS_RANGE_BEHIND - margin, self.config.OBS_RANGE_AHEAD + margin)
        y_range = (-max(self.config.OBS_RANGE_LEFT, abs(rel_positions[:, 1]).max()) - margin,
                   max(self.config.OBS_RANGE_RIGHT, abs(rel_positions[:, 1]).max()) + margin)
        
        # Generate gaps
        gaps = generate_gaps(ego, surrounding, self.config)
        st_gaps = generate_spatio_temporal_gaps(ego, surrounding, self.config)
        
        # 1. Risk field visualization
        self._plot_risk_field(ax1, ego, surrounding, x_range, y_range)
        
        # 2. Rectangular risk bounds
        self._plot_risk_bounds(ax2, ego, surrounding, x_range, y_range)
        
        # 3. Current gap distribution
        self._plot_gaps(ax3, ego, surrounding, gaps, x_range, y_range)
        
        # 4. Spatio-temporal gap map
        self._plot_st_gap_map(ax4, ego, st_gaps)
        
        # 5. Risk assessment details
        self._plot_risk_details(ax5, ego, surrounding)
        
        # 6. Gap statistics
        self._plot_gap_statistics(ax6, gaps)
        
        # 7. Safety distances
        self._plot_safety_distances(ax7, ego, surrounding)
        
        # 8. Summary
        self._plot_summary(ax8, ego, surrounding, gaps, st_gaps)
        
        fig.suptitle(
            f"SGTP Risk Assessment & Gap Generation: {ego['class'].title()} (ID: {ego['id']}) | "
            f"Frame: {snapshot['frame']} | Vehicles: {len(surrounding)} | Gaps: {len(gaps)}\n"
            f"Based on Xu et al. (2025) IEEE TVT",
            fontsize=14, fontweight='bold', color='white', y=0.98
        )
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        if output_path:
            fig.savefig(output_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
            logger.info(f"Saved: {output_path}")
            plt.close(fig)
        else:
            plt.show()
    
    def _plot_risk_field(self, ax, ego: Dict, vehicles: List[Dict],
                         x_range: Tuple, y_range: Tuple):
        """Plot elliptical risk field."""
        ax.set_facecolor('#1A1A2E')
        
        X, Y, risk = compute_risk_field_grid(
            ego, vehicles, x_range, y_range,
            (self.config.FIELD_GRID_X, self.config.FIELD_GRID_Y),
            self.config
        )
        
        risk_norm = np.log1p(risk)
        
        cmap = LinearSegmentedColormap.from_list('risk',
            ['#1A1A2E', '#2D4A3E', '#4A7C59', '#F4A261', '#E76F51', '#E63946'])
        pcm = ax.pcolormesh(X, Y, risk_norm, cmap=cmap, shading='gouraud', alpha=0.9)
        
        # Draw vehicles
        self._draw_vehicles_local(ax, ego, vehicles)
        
        ax.set_xlim(x_range)
        ax.set_ylim(y_range)
        ax.set_xlabel('Longitudinal s (m)', color='white')
        ax.set_ylabel('Lateral l (m)', color='white')
        ax.set_title('Elliptical Risk Field (Eq. 6)', fontsize=10, fontweight='bold', color='white')
        ax.tick_params(colors='white')
        ax.set_aspect('equal')
        
        cbar = plt.colorbar(pcm, ax=ax, shrink=0.7)
        cbar.set_label('log(1 + L)', color='white', fontsize=8)
        cbar.ax.tick_params(colors='white', labelsize=7)
        
        for spine in ax.spines.values():
            spine.set_color('#4A4A6A')
    
    def _plot_risk_bounds(self, ax, ego: Dict, vehicles: List[Dict],
                          x_range: Tuple, y_range: Tuple):
        """Plot rectangular risk bounds (inflated occupied areas)."""
        ax.set_facecolor('#1A1A2E')
        
        # Draw lane markings
        for lane_offset in range(-3, 4):
            y = lane_offset * self.config.LANE_WIDTH
            ls = '-' if lane_offset in [-3, 3] else '--'
            ax.axhline(y, color='white', linestyle=ls, alpha=0.3, linewidth=0.5)
        
        # Draw rectangular risk bounds for each vehicle
        patches = []
        colors = []
        
        for v in vehicles:
            s_min, s_max, l_min, l_max = compute_rectangular_risk_bounds(ego, v, self.config)
            
            # Risk bound rectangle
            width = s_max - s_min
            height = l_max - l_min
            rect = mpatches.Rectangle((s_min, l_min), width, height,
                                       linewidth=1.5, edgecolor='#E74C3C',
                                       facecolor='#E74C3C', alpha=0.3)
            ax.add_patch(rect)
            
            # Vehicle rectangle
            r_s, r_l, _, _ = compute_distance_vector(ego, v)
            v_rect = mpatches.FancyBboxPatch(
                (r_s - v['length']/2, r_l - v['width']/2),
                v['length'], v['width'],
                boxstyle="round,pad=0.02",
                facecolor=self.config.COLORS['car'],
                edgecolor='white', linewidth=1
            )
            ax.add_patch(v_rect)
            ax.text(r_s, r_l + v['width']/2 + 1, str(v['id']),
                   ha='center', fontsize=7, color='yellow')
        
        # Ego vehicle
        ego_rect = mpatches.FancyBboxPatch(
            (-ego['length']/2, -ego['width']/2),
            ego['length'], ego['width'],
            boxstyle="round,pad=0.02",
            facecolor=self.config.COLORS['truck'],
            edgecolor='white', linewidth=2
        )
        ax.add_patch(ego_rect)
        ax.text(0, -ego['width']/2 - 2, 'EGO', ha='center', fontsize=8,
               color='white', fontweight='bold')
        
        ax.set_xlim(x_range)
        ax.set_ylim(y_range)
        ax.set_xlabel('Longitudinal s (m)', color='white')
        ax.set_ylabel('Lateral l (m)', color='white')
        ax.set_title('Rectangular Risk Bounds (Eq. 7-9)', fontsize=10, fontweight='bold', color='white')
        ax.tick_params(colors='white')
        ax.set_aspect('equal')
        
        for spine in ax.spines.values():
            spine.set_color('#4A4A6A')
    
    def _plot_gaps(self, ax, ego: Dict, vehicles: List[Dict], gaps: List[Gap],
                   x_range: Tuple, y_range: Tuple):
        """Plot current gap distribution."""
        ax.set_facecolor('#1A1A2E')
        
        # Draw lane markings
        for lane_offset in range(-3, 4):
            y = lane_offset * self.config.LANE_WIDTH
            ls = '-' if lane_offset in [-3, 3] else '--'
            ax.axhline(y, color='white', linestyle=ls, alpha=0.3, linewidth=0.5)
        
        # Draw gaps
        for gap in gaps:
            s_min = gap.center_s - gap.length / 2
            l_min = gap.center_l - gap.width / 2
            
            rect = mpatches.FancyBboxPatch(
                (s_min, l_min), gap.length, gap.width,
                boxstyle="round,pad=0.02",
                facecolor=self.config.COLORS['gap'],
                edgecolor='white', linewidth=1, alpha=0.4
            )
            ax.add_patch(rect)
            
            # Gap label
            ax.text(gap.center_s, gap.center_l, f'G{gap.id}\n{gap.length:.0f}m',
                   ha='center', va='center', fontsize=7, color='white',
                   bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))
        
        # Draw vehicles with risk bounds
        for v in vehicles:
            s_min, s_max, l_min, l_max = compute_rectangular_risk_bounds(ego, v, self.config)
            
            # Risk bound
            rect = mpatches.Rectangle((s_min, l_min), s_max-s_min, l_max-l_min,
                                       linewidth=1, edgecolor='#E74C3C',
                                       facecolor='#E74C3C', alpha=0.2)
            ax.add_patch(rect)
            
            # Vehicle
            r_s, r_l, _, _ = compute_distance_vector(ego, v)
            v_rect = mpatches.FancyBboxPatch(
                (r_s - v['length']/2, r_l - v['width']/2),
                v['length'], v['width'],
                boxstyle="round,pad=0.02",
                facecolor=self.config.COLORS['car'],
                edgecolor='white', linewidth=1
            )
            ax.add_patch(v_rect)
        
        # Ego
        ego_rect = mpatches.FancyBboxPatch(
            (-ego['length']/2, -ego['width']/2),
            ego['length'], ego['width'],
            boxstyle="round,pad=0.02",
            facecolor=self.config.COLORS['truck'],
            edgecolor='white', linewidth=2
        )
        ax.add_patch(ego_rect)
        
        ax.set_xlim(x_range)
        ax.set_ylim(y_range)
        ax.set_xlabel('Longitudinal s (m)', color='white')
        ax.set_ylabel('Lateral l (m)', color='white')
        ax.set_title(f'Safety-Guaranteed Gaps (N={len(gaps)})', fontsize=10, fontweight='bold', color='white')
        ax.tick_params(colors='white')
        ax.set_aspect('equal')
        
        for spine in ax.spines.values():
            spine.set_color('#4A4A6A')
    
    def _plot_st_gap_map(self, ax, ego: Dict, st_gaps: Dict[int, List[Gap]]):
        """Plot spatio-temporal gap distribution map (Fig. 4 from paper)."""
        ax.set_facecolor('#1A1A2E')
        
        # Create 3D-like visualization in 2D (s-t projection for each lane)
        n_steps = len(st_gaps)
        
        # Collect all lanes
        all_lanes = set()
        for gaps in st_gaps.values():
            for gap in gaps:
                all_lanes.add(gap.lane_id)
        
        if not all_lanes:
            ax.text(0.5, 0.5, 'No gaps generated', ha='center', va='center',
                   color='white', transform=ax.transAxes)
            return
        
        # Color map for lanes
        lane_colors = plt.cm.Set2(np.linspace(0, 1, len(all_lanes)))
        lane_color_map = {lane: lane_colors[i] for i, lane in enumerate(sorted(all_lanes))}
        
        # Plot gaps as rectangles in s-t space
        for t_idx, gaps in st_gaps.items():
            t = t_idx * self.config.TIME_STEP
            
            for gap in gaps:
                s_min = gap.center_s - gap.length / 2
                s_max = gap.center_s + gap.length / 2
                
                # Offset by lane for visibility
                t_offset = gap.lane_id * 0.15
                
                rect = mpatches.Rectangle(
                    (s_min, t + t_offset), gap.length, self.config.TIME_STEP * 0.8,
                    facecolor=lane_color_map[gap.lane_id],
                    edgecolor='white', linewidth=0.5, alpha=0.6
                )
                ax.add_patch(rect)
        
        # Draw current ego position line
        ax.axvline(0, color='yellow', linestyle='--', linewidth=2, label='Ego position')
        
        # Legend for lanes
        from matplotlib.lines import Line2D
        legend_elements = [Line2D([0], [0], marker='s', color='w', 
                                  markerfacecolor=lane_color_map[l], markersize=10,
                                  label=f'Lane {l}', linestyle='None')
                         for l in sorted(all_lanes)]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=8)
        
        ax.set_xlim(-self.config.OBS_RANGE_BEHIND, self.config.OBS_RANGE_AHEAD)
        ax.set_ylim(0, self.config.TIME_HORIZON)
        ax.set_xlabel('Longitudinal s (m)', color='white')
        ax.set_ylabel('Time t (s)', color='white')
        ax.set_title('Spatio-Temporal Gap Distribution Map (Fig. 4)', 
                    fontsize=10, fontweight='bold', color='white')
        ax.tick_params(colors='white')
        ax.grid(True, alpha=0.2)
        
        for spine in ax.spines.values():
            spine.set_color('#4A4A6A')
    
    def _plot_risk_details(self, ax, ego: Dict, vehicles: List[Dict]):
        """Plot risk assessment details for each vehicle."""
        ax.set_facecolor('#1A1A2E')
        
        if not vehicles:
            ax.text(0.5, 0.5, 'No vehicles', ha='center', va='center', color='white')
            return
        
        # Compute risk for each vehicle
        risk_data = []
        for v in vehicles:
            r_s, r_l, dist, theta = compute_distance_vector(ego, v)
            D_s, D_l = compute_safety_distances(ego, v, self.config)
            L = compute_elliptical_risk_field(ego, v, self.config)
            
            risk_data.append({
                'id': v['id'],
                'distance': dist,
                'r_s': r_s,
                'r_l': r_l,
                'D_s': D_s,
                'D_l': D_l,
                'risk': L
            })
        
        # Sort by risk
        risk_data.sort(key=lambda x: x['risk'], reverse=True)
        
        n = min(len(risk_data), 10)
        x = np.arange(n)
        risks = [d['risk'] for d in risk_data[:n]]
        labels = [f"V{d['id']}" for d in risk_data[:n]]
        
        # Color by risk level
        colors = ['#E74C3C' if r > 1 else '#F4A261' if r > 0.5 else '#2ECC71' for r in risks]
        
        bars = ax.barh(x, risks, color=colors, edgecolor='white', alpha=0.8)
        
        ax.set_yticks(x)
        ax.set_yticklabels(labels, color='white', fontsize=8)
        ax.set_xlabel('Risk Level (L)', color='white')
        ax.set_title('Vehicle Risk Assessment', fontsize=10, fontweight='bold', color='white')
        ax.tick_params(colors='white')
        ax.axvline(1.0, color='#E74C3C', linestyle='--', alpha=0.7, label='High risk')
        ax.legend(loc='lower right', fontsize=7)
        
        for spine in ax.spines.values():
            spine.set_color('#4A4A6A')
    
    def _plot_gap_statistics(self, ax, gaps: List[Gap]):
        """Plot gap statistics."""
        ax.set_facecolor('#1A1A2E')
        
        if not gaps:
            ax.text(0.5, 0.5, 'No gaps', ha='center', va='center', color='white')
            return
        
        # Group by lane
        lane_gaps = {}
        for gap in gaps:
            if gap.lane_id not in lane_gaps:
                lane_gaps[gap.lane_id] = []
            lane_gaps[gap.lane_id].append(gap)
        
        lanes = sorted(lane_gaps.keys())
        n_gaps = [len(lane_gaps[l]) for l in lanes]
        avg_lengths = [np.mean([g.length for g in lane_gaps[l]]) for l in lanes]
        
        x = np.arange(len(lanes))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, n_gaps, width, label='N Gaps', 
                      color='#3498DB', edgecolor='white')
        
        ax2 = ax.twinx()
        bars2 = ax2.bar(x + width/2, avg_lengths, width, label='Avg Length (m)',
                       color='#2ECC71', edgecolor='white')
        
        ax.set_xticks(x)
        ax.set_xticklabels([f'Lane {l}' for l in lanes], color='white', fontsize=8)
        ax.set_ylabel('Number of Gaps', color='#3498DB')
        ax2.set_ylabel('Avg Gap Length (m)', color='#2ECC71')
        ax.set_title('Gap Statistics by Lane', fontsize=10, fontweight='bold', color='white')
        ax.tick_params(colors='white')
        ax2.tick_params(colors='white')
        
        ax.legend(loc='upper left', fontsize=7)
        ax2.legend(loc='upper right', fontsize=7)
        
        for spine in ax.spines.values():
            spine.set_color('#4A4A6A')
        for spine in ax2.spines.values():
            spine.set_color('#4A4A6A')
    
    def _plot_safety_distances(self, ax, ego: Dict, vehicles: List[Dict]):
        """Plot safety distances comparison."""
        ax.set_facecolor('#1A1A2E')
        
        if not vehicles:
            return
        
        # Compute safety distances
        data = []
        for v in vehicles:
            D_s, D_l = compute_safety_distances(ego, v, self.config)
            r_s, r_l, dist, _ = compute_distance_vector(ego, v)
            
            data.append({
                'id': v['id'],
                'D_s': D_s,
                'D_l': D_l,
                'actual_s': abs(r_s),
                'actual_l': abs(r_l)
            })
        
        # Sort by longitudinal safety margin
        data.sort(key=lambda x: x['actual_s'] - x['D_s'])
        
        n = min(len(data), 8)
        labels = [f"V{d['id']}" for d in data[:n]]
        
        x = np.arange(n)
        width = 0.2
        
        # Longitudinal
        ax.bar(x - width*1.5, [d['D_s'] for d in data[:n]], width, 
              label='D_s (required)', color='#E74C3C', alpha=0.7)
        ax.bar(x - width*0.5, [d['actual_s'] for d in data[:n]], width,
              label='r_s (actual)', color='#3498DB', alpha=0.7)
        
        # Lateral
        ax.bar(x + width*0.5, [d['D_l'] for d in data[:n]], width,
              label='D_l (required)', color='#F4A261', alpha=0.7)
        ax.bar(x + width*1.5, [d['actual_l'] for d in data[:n]], width,
              label='r_l (actual)', color='#2ECC71', alpha=0.7)
        
        ax.set_xticks(x)
        ax.set_xticklabels(labels, color='white', fontsize=8)
        ax.set_ylabel('Distance (m)', color='white')
        ax.set_title('Safety Distance vs Actual Distance', fontsize=10, fontweight='bold', color='white')
        ax.tick_params(colors='white')
        ax.legend(loc='upper right', fontsize=7, ncol=2)
        
        for spine in ax.spines.values():
            spine.set_color('#4A4A6A')
    
    def _plot_summary(self, ax, ego: Dict, vehicles: List[Dict], 
                      gaps: List[Gap], st_gaps: Dict):
        """Plot summary panel."""
        ax.set_facecolor('#1A1A2E')
        ax.axis('off')
        
        # Compute statistics
        risks = [compute_elliptical_risk_field(ego, v, self.config) for v in vehicles]
        high_risk = sum(1 for r in risks if r > 1.0)
        
        total_gap_length = sum(g.length for g in gaps)
        n_lanes = len(set(g.lane_id for g in gaps))
        
        lines = [
            "═" * 42,
            "   SGTP ANALYSIS SUMMARY",
            "═" * 42,
            "",
            f"EGO VEHICLE: {ego['class'].upper()}",
            f"  ID: {ego['id']}",
            f"  Speed: {ego['speed']:.1f} m/s ({ego['speed']*3.6:.1f} km/h)",
            f"  Dimensions: {ego['length']:.1f}m × {ego['width']:.1f}m",
            "",
            "─" * 42,
            "SURROUNDING VEHICLES",
            "─" * 42,
            f"  Total: {len(vehicles)}",
            f"  High risk (L>1): {high_risk}",
            f"  Mean risk: {np.mean(risks):.2f}" if risks else "  Mean risk: N/A",
            f"  Max risk: {max(risks):.2f}" if risks else "  Max risk: N/A",
            "",
            "─" * 42,
            "GAP GENERATION",
            "─" * 42,
            f"  Current gaps: {len(gaps)}",
            f"  Lanes with gaps: {n_lanes}",
            f"  Total gap length: {total_gap_length:.1f}m",
            f"  Mean gap length: {total_gap_length/len(gaps):.1f}m" if gaps else "",
            "",
            "─" * 42,
            "SPATIO-TEMPORAL MAP",
            "─" * 42,
            f"  Time horizon: {self.config.TIME_HORIZON}s",
            f"  Time steps: {len(st_gaps)}",
            f"  Total gaps (all t): {sum(len(g) for g in st_gaps.values())}",
            "",
            "═" * 42,
            "Model: Xu et al. (2025) IEEE TVT",
            "Risk: L = D_s²/r_s² + D_l²/r_l²",
            "═" * 42,
        ]
        
        ax.text(0.02, 0.98, "\n".join(lines), transform=ax.transAxes,
               fontsize=8, color='white', family='monospace',
               verticalalignment='top')
        
        for spine in ax.spines.values():
            spine.set_color('#4A4A6A')
    
    def _draw_vehicles_local(self, ax, ego: Dict, vehicles: List[Dict]):
        """Draw vehicles in ego-relative frame."""
        cos_h = np.cos(-ego['heading'])
        sin_h = np.sin(-ego['heading'])
        
        # Ego at origin
        ego_rect = mpatches.FancyBboxPatch(
            (-ego['length']/2, -ego['width']/2), ego['length'], ego['width'],
            boxstyle="round,pad=0.02", facecolor=self.config.COLORS['truck'],
            edgecolor='white', linewidth=2
        )
        ax.add_patch(ego_rect)
        ax.text(0, -ego['width']/2 - 2, 'EGO', ha='center', color='white', fontsize=8, fontweight='bold')
        
        for v in vehicles:
            dx = v['x'] - ego['x']
            dy = v['y'] - ego['y']
            dx_rel = dx * cos_h - dy * sin_h
            dy_rel = dx * sin_h + dy * cos_h
            
            rect = mpatches.FancyBboxPatch(
                (dx_rel - v['length']/2, dy_rel - v['width']/2),
                v['length'], v['width'],
                boxstyle="round,pad=0.02", facecolor=self.config.COLORS['car'],
                edgecolor='white', linewidth=1, alpha=0.8
            )
            ax.add_patch(rect)
            ax.text(dx_rel, dy_rel + v['width']/2 + 1, str(v['id']),
                   ha='center', fontsize=7, color='yellow')


# =============================================================================
# Main
# =============================================================================

def main(data_dir: str, recording_id: int, ego_id: Optional[int] = None,
         frame: Optional[int] = None, output_dir: str = './output_sgtp'):
    """Main entry point."""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 60)
    logger.info("SGTP Risk Assessment & Gap Generation")
    logger.info("Based on Xu et al. (2025) IEEE TVT")
    logger.info("=" * 60)
    
    loader = ExiDLoader(data_dir)
    if not loader.load_recording(recording_id):
        return
    
    if ego_id is None:
        heavy_ids = loader.get_heavy_vehicles()
        if not heavy_ids:
            logger.error("No heavy vehicles found")
            return
        ego_id = heavy_ids[0]
        logger.info(f"Auto-selected ego: {ego_id}")
    
    if frame is None:
        frame = loader.find_best_frame(ego_id, min_vehicles=5)
        if frame is None:
            logger.error("Could not find suitable frame")
            return
        logger.info(f"Auto-selected frame: {frame}")
    
    snapshot = loader.get_snapshot(ego_id, frame)
    if snapshot is None:
        logger.error("Could not get snapshot")
        return
    
    logger.info(f"Ego: {snapshot['ego']['class']} (ID: {ego_id})")
    logger.info(f"Surrounding: {len(snapshot['surrounding'])} vehicles")
    
    viz = SGTPVisualizer(loader=loader)
    
    output_file = output_path / f'sgtp_rec{recording_id}_ego{ego_id}_frame{snapshot["frame"]}.png'
    viz.create_analysis_figure(snapshot, str(output_file))
    
    logger.info(f"Output saved to: {output_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='SGTP Risk Assessment & Gap Generation (Xu et al. 2025)')
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--recording', type=int, default=25)
    parser.add_argument('--ego_id', type=int, default=None)
    parser.add_argument('--frame', type=int, default=None)
    parser.add_argument('--output_dir', type=str, default='./output_sgtp')
    
    args = parser.parse_args()
    main(args.data_dir, args.recording, args.ego_id, args.frame, args.output_dir)