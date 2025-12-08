"""
exiD Dataset: Complete Visualization with Animation & Corrected SVO
===================================================================
Features:
1. Animated GIF visualization (like original)
2. Hysteresis-based tracking (prevents abrupt vehicle disappearances)
3. Independent bidirectional SVO computation (corrected)
4. All static analysis plots

Key Fixes:
- Surrounding vehicles tracked with 10-frame hysteresis
- Distance-based fallback when slot detection fails
- SVO computed independently for each vehicle (not complementary)
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
from collections import defaultdict
import warnings
import argparse
import logging
from scipy.signal import savgol_filter

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class Config:
    """Complete configuration."""
    
    HEAVY_VEHICLE_CLASSES: Set[str] = field(default_factory=lambda: {'truck', 'bus', 'van', 'trailer'})
    CAR_CLASSES: Set[str] = field(default_factory=lambda: {'car'})
    
    # Aggressiveness parameters (Hu et al.)
    MU_1: float = 0.2
    MU_2: float = 0.21
    SIGMA: float = 0.1
    DELTA: float = 0.0005
    TAU_1: float = 0.2
    TAU_2: float = 0.1
    BETA: float = 0.05
    
    # Vehicle masses
    MASS_HV: float = 15000.0
    MASS_PC: float = 3000.0
    
    # Reference values for SVO normalization
    V_REF: float = 30.0
    DIST_REF: float = 30.0
    
    # SVO behavioral weights
    WEIGHT_AGGR_RATIO: float = 0.4
    WEIGHT_DECEL: float = 0.3
    WEIGHT_YIELDING: float = 0.3
    
    # === HYSTERESIS TRACKING (FIXES ABRUPT DISAPPEARANCE) ===
    TRACKING_HYSTERESIS: int = 10  # Frames to keep tracking after slot loss
    MAX_TRACKING_DISTANCE: float = 100.0  # Max distance for fallback tracking
    INTERPOLATION_DECAY: float = 0.85  # Decay factor for interpolated values
    
    # Interaction detection
    ROI_RADIUS: float = 80.0
    MIN_INTERACTION_FRAMES: int = 50
    
    # Smoothing
    SMOOTH_WINDOW: int = 15
    SMOOTH_POLY: int = 2
    MAX_SVO_DELTA: float = 2.5
    
    # Visualization
    TRAIL_LENGTH: int = 75
    FPS: int = 25
    GRID_RESOLUTION: int = 60
    
    # Position colors
    POSITION_COLORS: Dict[str, str] = field(default_factory=lambda: {
        'lead': '#E74C3C', 'rear': '#3498DB',
        'leftLead': '#9B59B6', 'leftAlongside': '#E67E22', 'leftRear': '#1ABC9C',
        'rightLead': '#F1C40F', 'rightAlongside': '#2ECC71', 'rightRear': '#34495E',
        'tracked': '#95A5A6',  # For hysteresis-tracked vehicles
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


@dataclass
class TrackedVehicle:
    """Vehicle with tracking metadata for hysteresis."""
    state: VehicleState
    position_label: str
    last_slot_frame: int  # Last frame seen in slot detection
    frames_since_slot: int = 0  # How many frames since lost from slot
    is_interpolated: bool = False


@dataclass
class SVOState:
    """SVO state for ego vehicle considering all surrounding vehicles."""
    ego_id: int
    frame: int
    
    # Per-vehicle SVOs (ego's SVO toward each)
    ego_svo_per_vehicle: Dict[int, float] = field(default_factory=dict)
    
    # Per-vehicle SVOs (each vehicle's SVO toward ego)
    vehicle_svo_to_ego: Dict[int, float] = field(default_factory=dict)
    
    # Aggressiveness
    aggr_ego_to_others: Dict[int, float] = field(default_factory=dict)
    aggr_others_to_ego: Dict[int, float] = field(default_factory=dict)
    
    # Aggregates
    mean_ego_svo: float = 45.0
    mean_others_svo: float = 45.0
    total_exerted: float = 0.0
    total_suffered: float = 0.0


# =============================================================================
# Hysteresis-Based Vehicle Tracker
# =============================================================================

class HysteresisTracker:
    """
    Tracks surrounding vehicles with hysteresis to prevent abrupt disappearances.
    
    When a vehicle leaves slot detection:
    1. Continue tracking for TRACKING_HYSTERESIS frames
    2. Use distance-based detection as fallback
    3. Interpolate state if vehicle temporarily undetectable
    """
    
    def __init__(self, config: Config):
        self.config = config
        
        # Track vehicles by ID -> TrackedVehicle
        self.tracked_vehicles: Dict[int, TrackedVehicle] = {}
        
        # History of which vehicles were in which slots
        self.slot_history: Dict[str, List[int]] = defaultdict(list)
    
    def update(
        self,
        frame: int,
        ego_state: VehicleState,
        slot_vehicles: Dict[str, Optional[VehicleState]],
        all_frame_vehicles: Dict[int, VehicleState]
    ) -> Dict[str, TrackedVehicle]:
        """
        Update tracking with hysteresis.
        
        Args:
            frame: Current frame number
            ego_state: Ego vehicle state
            slot_vehicles: Vehicles detected in slots (from exiD)
            all_frame_vehicles: All vehicles in this frame (for fallback)
        
        Returns:
            Dict mapping position labels to tracked vehicles
        """
        
        result = {}
        seen_ids = set()
        
        # 1. Process slot-detected vehicles (primary detection)
        for slot_name, vehicle in slot_vehicles.items():
            if vehicle is None:
                continue
            
            seen_ids.add(vehicle.track_id)
            
            tracked = TrackedVehicle(
                state=vehicle,
                position_label=slot_name,
                last_slot_frame=frame,
                frames_since_slot=0,
                is_interpolated=False
            )
            
            self.tracked_vehicles[vehicle.track_id] = tracked
            result[slot_name] = tracked
        
        # 2. Check previously tracked vehicles (hysteresis)
        vehicles_to_remove = []
        
        for vid, tracked in self.tracked_vehicles.items():
            if vid in seen_ids:
                continue  # Already processed
            
            if vid == ego_state.track_id:
                continue  # Skip ego
            
            frames_since = frame - tracked.last_slot_frame
            
            if frames_since > self.config.TRACKING_HYSTERESIS:
                # Too long since seen, stop tracking
                vehicles_to_remove.append(vid)
                continue
            
            # Try to find vehicle in all_frame_vehicles (distance-based fallback)
            if vid in all_frame_vehicles:
                new_state = all_frame_vehicles[vid]
                
                # Check if still within tracking distance
                dist = np.sqrt(
                    (new_state.x - ego_state.x)**2 + 
                    (new_state.y - ego_state.y)**2
                )
                
                if dist <= self.config.MAX_TRACKING_DISTANCE:
                    # Update state but mark as "tracked" (not in slot)
                    tracked.state = new_state
                    tracked.frames_since_slot = frames_since
                    tracked.is_interpolated = False
                    
                    # Use a unique label for tracked vehicles
                    label = f"tracked_{vid}"
                    result[label] = tracked
                    seen_ids.add(vid)
                else:
                    vehicles_to_remove.append(vid)
            else:
                # Vehicle not found, interpolate if recent
                if frames_since <= 3:  # Only interpolate for very recent loss
                    interpolated = self._interpolate_state(tracked, frames_since)
                    if interpolated:
                        tracked.state = interpolated
                        tracked.frames_since_slot = frames_since
                        tracked.is_interpolated = True
                        
                        label = f"interpolated_{vid}"
                        result[label] = tracked
                        seen_ids.add(vid)
                else:
                    vehicles_to_remove.append(vid)
        
        # Clean up old tracked vehicles
        for vid in vehicles_to_remove:
            del self.tracked_vehicles[vid]
        
        return result
    
    def _interpolate_state(
        self, 
        tracked: TrackedVehicle, 
        frames_elapsed: int
    ) -> Optional[VehicleState]:
        """Interpolate vehicle state based on last known velocity."""
        
        old = tracked.state
        dt = frames_elapsed / self.config.FPS
        
        # Simple linear extrapolation
        new_x = old.x + old.x_velocity * dt
        new_y = old.y + old.y_velocity * dt
        
        # Apply decay to velocity (vehicle likely slowing)
        decay = self.config.INTERPOLATION_DECAY ** frames_elapsed
        
        return VehicleState(
            track_id=old.track_id,
            frame=old.frame + frames_elapsed,
            x=new_x,
            y=new_y,
            heading=old.heading,
            width=old.width,
            length=old.length,
            x_velocity=old.x_velocity * decay,
            y_velocity=old.y_velocity * decay,
            x_acceleration=old.x_acceleration * decay,
            y_acceleration=old.y_acceleration * decay,
            vehicle_class=old.vehicle_class,
            mass=old.mass
        )
    
    def reset(self):
        """Reset tracker state."""
        self.tracked_vehicles = {}
        self.slot_history = defaultdict(list)
    
    def get_tracking_stats(self) -> Dict:
        """Get tracking statistics."""
        n_slot = sum(1 for t in self.tracked_vehicles.values() if t.frames_since_slot == 0)
        n_hysteresis = sum(1 for t in self.tracked_vehicles.values() if t.frames_since_slot > 0 and not t.is_interpolated)
        n_interpolated = sum(1 for t in self.tracked_vehicles.values() if t.is_interpolated)
        
        return {
            'slot_detected': n_slot,
            'hysteresis_tracked': n_hysteresis,
            'interpolated': n_interpolated,
            'total': len(self.tracked_vehicles)
        }


# =============================================================================
# Independent SVO Calculator
# =============================================================================

class IndependentSVOCalculator:
    """Computes INDEPENDENT SVO for each vehicle."""
    
    def __init__(self, config: Config):
        self.config = config
        self.svo_history: Dict[int, List[float]] = defaultdict(list)  # Per-vehicle history
    
    def compute_svo_state(
        self,
        ego: VehicleState,
        tracked_vehicles: Dict[str, TrackedVehicle]
    ) -> SVOState:
        """Compute SVO state considering all tracked vehicles."""
        
        state = SVOState(ego_id=ego.track_id, frame=ego.frame)
        
        ego_svos = []
        others_svos = []
        
        for label, tracked in tracked_vehicles.items():
            other = tracked.state
            if other.track_id == ego.track_id:
                continue
            
            # Compute bidirectional aggressiveness
            aggr_ego_to_other = self._compute_aggressiveness(ego, other)
            aggr_other_to_ego = self._compute_aggressiveness(other, ego)
            
            state.aggr_ego_to_others[other.track_id] = aggr_ego_to_other
            state.aggr_others_to_ego[other.track_id] = aggr_other_to_ego
            state.total_exerted += aggr_ego_to_other
            state.total_suffered += aggr_other_to_ego
            
            # Compute INDEPENDENT SVOs
            ego_svo = self._compute_independent_svo(ego, other, aggr_ego_to_other)
            other_svo = self._compute_independent_svo(other, ego, aggr_other_to_ego)
            
            state.ego_svo_per_vehicle[other.track_id] = ego_svo
            state.vehicle_svo_to_ego[other.track_id] = other_svo
            
            ego_svos.append(ego_svo)
            others_svos.append(other_svo)
        
        # Compute means
        if ego_svos:
            state.mean_ego_svo = np.mean(ego_svos)
        if others_svos:
            state.mean_others_svo = np.mean(others_svos)
        
        return state
    
    def _compute_independent_svo(
        self,
        vehicle: VehicleState,
        other: VehicleState,
        aggr_exerted: float
    ) -> float:
        """
        Compute SVO based on vehicle's OWN behavior.
        Independent of other's SVO!
        """
        
        # 1. Normalized aggressiveness
        potential = self._compute_potential_aggressiveness(vehicle, other)
        if potential > 1e-6:
            normalized_aggr = min(1.0, aggr_exerted / potential)
        else:
            normalized_aggr = 0.5
        
        # 2. Deceleration behavior
        decel = max(0, -vehicle.acceleration)
        normalized_decel = min(1.0, decel / 3.0)
        
        # 3. Speed ratio (yielding = slower than free flow)
        speed_ratio = min(1.0, vehicle.speed / self.config.V_REF)
        normalized_yielding = 0.5 * normalized_decel + 0.5 * (1 - speed_ratio)
        
        # 4. Combine into SVO
        # Low aggression → high SVO
        svo_aggr = 90 - 135 * normalized_aggr
        
        # Deceleration → higher SVO
        svo_decel = 45 * normalized_decel
        
        # Yielding → higher SVO
        svo_yield = -22.5 + 45 * normalized_yielding
        
        w1 = self.config.WEIGHT_AGGR_RATIO
        w2 = self.config.WEIGHT_DECEL
        w3 = self.config.WEIGHT_YIELDING
        
        svo = w1 * svo_aggr + w2 * svo_decel + w3 * svo_yield
        
        return np.clip(svo, -45, 90)
    
    def _compute_aggressiveness(
        self,
        aggressor: VehicleState,
        sufferer: VehicleState
    ) -> float:
        """Compute asymmetric aggressiveness."""
        
        dx = sufferer.x - aggressor.x
        dy = sufferer.y - aggressor.y
        dist = np.sqrt(dx**2 + dy**2)
        
        if dist < 0.1:
            return 0.0
        
        pos_unit_x = dx / dist
        pos_unit_y = dy / dist
        
        v_i = aggressor.speed
        v_j = sufferer.speed
        
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
        
        # Pseudo-distance
        phi = aggressor.heading
        x_local = dx * np.cos(phi) + dy * np.sin(phi)
        y_local = -dx * np.sin(phi) + dy * np.cos(phi)
        
        expansion = np.exp(2 * self.config.BETA * v_i)
        tau_1 = self.config.TAU_1 * expansion
        tau_2 = self.config.TAU_2
        
        if tau_1 > 0 and tau_2 > 0:
            r_pseudo = np.sqrt((x_local**2)/(tau_1**2) + (y_local**2)/(tau_2**2))
            r_pseudo = r_pseudo * min(tau_1, tau_2)
        else:
            r_pseudo = dist
        r_pseudo = max(r_pseudo, 0.1)
        
        xi_1 = self.config.MU_1 * v_i * cos_theta_i + self.config.MU_2 * v_j * cos_theta_j
        xi_2 = -self.config.SIGMA * (aggressor.mass ** -1) * r_pseudo
        
        mass_term = (aggressor.mass * v_i) / (2 * self.config.DELTA * sufferer.mass)
        omega = mass_term * np.exp(xi_1 + xi_2)
        
        return np.clip(omega, 0, 2000)
    
    def _compute_potential_aggressiveness(
        self,
        vehicle: VehicleState,
        other: VehicleState
    ) -> float:
        """Maximum possible aggressiveness for normalization."""
        
        v_ref = self.config.V_REF
        dist_ref = self.config.DIST_REF
        
        mass_term = (vehicle.mass * v_ref) / (2 * self.config.DELTA * other.mass)
        xi_1_max = self.config.MU_1 * v_ref + self.config.MU_2 * other.speed
        xi_2_min = -self.config.SIGMA * (vehicle.mass ** -1) * dist_ref
        
        return mass_term * np.exp(xi_1_max + xi_2_min)
    
    def reset(self):
        self.svo_history = defaultdict(list)


# =============================================================================
# Potential Field Generator
# =============================================================================

class PotentialFieldGenerator:
    """Generates potential field visualization."""
    
    def __init__(self, config: Config):
        self.config = config
    
    def compute_field(
        self,
        x_grid: np.ndarray,
        y_grid: np.ndarray,
        ego: VehicleState,
        tracked_vehicles: Dict[str, TrackedVehicle]
    ) -> np.ndarray:
        """Compute combined potential field from all vehicles."""
        
        field = np.zeros_like(x_grid)
        
        # Ego vehicle field
        field += self._vehicle_field(x_grid, y_grid, ego, is_ego=True)
        
        # Surrounding vehicle fields
        for label, tracked in tracked_vehicles.items():
            other = tracked.state
            # Reduce field strength for interpolated vehicles
            scale = 0.5 if tracked.is_interpolated else 1.0
            field += scale * self._vehicle_field(x_grid, y_grid, other, is_ego=False)
        
        return field
    
    def _vehicle_field(
        self,
        x_grid: np.ndarray,
        y_grid: np.ndarray,
        vehicle: VehicleState,
        is_ego: bool = False
    ) -> np.ndarray:
        """Compute anisotropic field for single vehicle."""
        
        dx = x_grid - vehicle.x
        dy = y_grid - vehicle.y
        
        cos_h = np.cos(vehicle.heading)
        sin_h = np.sin(vehicle.heading)
        
        dx_local = dx * cos_h + dy * sin_h
        dy_local = -dx * sin_h + dy * cos_h
        
        v = vehicle.speed
        
        # Anisotropic sigma
        sigma_x_fwd = max(vehicle.length, 5.0) + v * 1.5
        sigma_x_bwd = max(vehicle.length, 5.0) * 0.5
        sigma_y = max(vehicle.width, 2.0) * 1.5
        
        sigma_x = np.where(dx_local > 0, sigma_x_fwd, sigma_x_bwd)
        
        # Mass-based amplitude
        amplitude = vehicle.mass / self.config.MASS_PC * 10.0
        
        exponent = -((dx_local**2) / (2 * sigma_x**2) + 
                     (dy_local**2) / (2 * sigma_y**2))
        
        return amplitude * np.exp(exponent)


# =============================================================================
# Data Loader
# =============================================================================

class ExiDLoader:
    """Loads exiD dataset with all necessary data."""
    
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
    
    def load_recording(self, recording_id: int) -> bool:
        prefix = f"{recording_id:02d}_"
        
        try:
            logger.info(f"Loading recording {recording_id}...")
            
            self.tracks_df = pd.read_csv(self.data_dir / f"{prefix}tracks.csv")
            self.tracks_meta_df = pd.read_csv(self.data_dir / f"{prefix}tracksMeta.csv")
            rec_meta_df = pd.read_csv(self.data_dir / f"{prefix}recordingMeta.csv")
            self.recording_meta = rec_meta_df.iloc[0]
            
            self.ortho_px_to_meter = self.recording_meta.get('orthoPxToMeter', 0.1)
            
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
            
            location_id = int(self.recording_meta.get('locationId', -1))
            location_name = self.LOCATION_INFO.get(location_id, {}).get('name', 'Unknown')
            
            logger.info(f"  Location: {location_id} ({location_name})")
            logger.info(f"  Duration: {self.recording_meta.get('duration', 0):.1f}s")
            
            heavy_mask = self.tracks_meta_df['class'].str.lower().isin(self.config.HEAVY_VEHICLE_CLASSES)
            logger.info(f"  Heavy vehicles: {heavy_mask.sum()}")
            
            return True
        except Exception as e:
            logger.error(f"Error: {e}")
            return False
    
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
        """Get ALL vehicles in a frame (for hysteresis fallback)."""
        frame_data = self.tracks_df[self.tracks_df['frame'] == frame]
        vehicles = {}
        
        for _, row in frame_data.iterrows():
            vid = int(row['trackId'])
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
# Complete Visualizer with Animation
# =============================================================================

class CompleteVisualizer:
    """Complete visualizer with animation and all fixes."""
    
    def __init__(self, loader: ExiDLoader):
        self.loader = loader
        self.config = Config()
        self.tracker = HysteresisTracker(self.config)
        self.svo_calc = IndependentSVOCalculator(self.config)
        self.field_gen = PotentialFieldGenerator(self.config)
        
        self.fig = None
        self.axes = {}
        self.elements = {}
    
    def animate_interaction(
        self,
        interaction: Dict,
        save_path: Optional[str] = None,
        show_field: bool = True
    ) -> Optional[animation.FuncAnimation]:
        """Create animated visualization with hysteresis tracking."""
        
        logger.info(f"Preparing animation for ego vehicle {interaction['ego_id']}...")
        
        # Reset
        self.tracker.reset()
        self.svo_calc.reset()
        
        # Prepare frame data
        frames_data = self._prepare_frames(interaction)
        
        if not frames_data:
            logger.error("No valid frame data.")
            return None
        
        bounds = self._calculate_bounds(frames_data)
        
        # Create figure
        self.fig = plt.figure(figsize=(20, 12))
        self.fig.patch.set_facecolor('#0D1117')
        
        # Layout
        self.axes['main'] = self.fig.add_axes([0.03, 0.35, 0.50, 0.60])
        self.axes['svo_chrono'] = self.fig.add_axes([0.56, 0.55, 0.42, 0.40])
        self.axes['aggr_balance'] = self.fig.add_axes([0.56, 0.08, 0.20, 0.40])
        self.axes['tracking_stats'] = self.fig.add_axes([0.78, 0.08, 0.20, 0.40])
        self.axes['info'] = self.fig.add_axes([0.03, 0.03, 0.50, 0.28])
        
        # Setup panels
        self._setup_main_panel(bounds, interaction, show_field)
        self._setup_svo_chronograph(frames_data)
        self._setup_aggr_balance()
        self._setup_tracking_stats()
        self._setup_info_panel(interaction)
        
        # Initialize elements
        self.elements = self._init_elements(show_field)
        
        # Field grid
        if show_field:
            x = np.linspace(bounds[0], bounds[1], self.config.GRID_RESOLUTION)
            y = np.linspace(bounds[2], bounds[3], self.config.GRID_RESOLUTION)
            self.field_grid_x, self.field_grid_y = np.meshgrid(x, y)
        
        # Create animation
        ani = animation.FuncAnimation(
            self.fig,
            self._update_frame,
            frames=len(frames_data),
            fargs=(frames_data, show_field),
            interval=1000 // self.config.FPS,
            blit=False,
            repeat=True
        )
        
        if save_path:
            logger.info(f"Saving animation to {save_path}...")
            writer = animation.PillowWriter(fps=min(self.config.FPS, 12))
            ani.save(save_path, writer=writer, dpi=100)
            logger.info("Animation saved!")
        
        return ani
    
    def _prepare_frames(self, interaction: Dict) -> List[Dict]:
        """Prepare data for each frame with hysteresis tracking."""
        
        frames_data = []
        ego_id = interaction['ego_id']
        
        # Accumulators
        all_times = []
        all_ego_svo = []
        all_others_svo = []
        all_exerted = []
        all_suffered = []
        all_n_slot = []
        all_n_tracked = []
        
        for i, frame in enumerate(interaction['frames']):
            ego_state = self.loader.get_vehicle_state(ego_id, frame)
            if ego_state is None:
                continue
            
            # Get slot-based surrounding vehicles
            slot_vehicles = self.loader.get_surrounding_vehicles_by_slot(ego_id, frame)
            
            # Get ALL vehicles in frame for hysteresis fallback
            all_frame_vehicles = self.loader.get_all_vehicles_in_frame(frame)
            
            # Update tracker with hysteresis
            tracked_vehicles = self.tracker.update(
                frame, ego_state, slot_vehicles, all_frame_vehicles
            )
            
            # Get tracking stats
            tracking_stats = self.tracker.get_tracking_stats()
            
            # Compute SVO
            svo_state = self.svo_calc.compute_svo_state(ego_state, tracked_vehicles)
            
            time_s = i / self.config.FPS
            
            # Accumulate
            all_times.append(time_s)
            all_ego_svo.append(svo_state.mean_ego_svo)
            all_others_svo.append(svo_state.mean_others_svo)
            all_exerted.append(svo_state.total_exerted)
            all_suffered.append(svo_state.total_suffered)
            all_n_slot.append(tracking_stats['slot_detected'])
            all_n_tracked.append(tracking_stats['total'])
            
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
                'all_times': all_times.copy(),
                'all_ego_svo': all_ego_svo.copy(),
                'all_others_svo': all_others_svo.copy(),
                'all_exerted': all_exerted.copy(),
                'all_suffered': all_suffered.copy(),
                'all_n_slot': all_n_slot.copy(),
                'all_n_tracked': all_n_tracked.copy(),
            })
        
        return frames_data
    
    def _calculate_bounds(self, frames_data: List[Dict]) -> Tuple:
        all_x, all_y = [], []
        
        for fd in frames_data:
            all_x.append(fd['ego'].x)
            all_y.append(fd['ego'].y)
            
            for label, tracked in fd['tracked_vehicles'].items():
                all_x.append(tracked.state.x)
                all_y.append(tracked.state.y)
        
        padding = 50
        return (min(all_x) - padding, max(all_x) + padding,
                min(all_y) - padding, max(all_y) + padding)
    
    def _setup_main_panel(self, bounds: Tuple, interaction: Dict, show_field: bool):
        ax = self.axes['main']
        ax.set_xlim(bounds[0], bounds[1])
        ax.set_ylim(bounds[2], bounds[3])
        ax.set_aspect('equal')
        
        if self.loader.background_image is not None:
            extent = self.loader.get_background_extent()
            ax.imshow(self.loader.background_image, extent=extent, alpha=0.5, aspect='auto', zorder=0)
        else:
            ax.set_facecolor('#1A1A2E')
        
        ax.set_xlabel('X (m)', fontsize=10, color='white')
        ax.set_ylabel('Y (m)', fontsize=10, color='white')
        ax.tick_params(colors='white')
        
        ax.set_title(
            f'Multi-Agent Interaction: {interaction["ego_class"].title()} (ID: {interaction["ego_id"]})\n'
            f'With Hysteresis Tracking (No Abrupt Disappearances)',
            fontsize=12, fontweight='bold', color='white'
        )
        
        for spine in ax.spines.values():
            spine.set_color('#4A4A6A')
    
    def _setup_svo_chronograph(self, frames_data: List[Dict]):
        ax = self.axes['svo_chrono']
        ax.set_facecolor('#1A1A2E')
        
        all_times = [fd['time'] for fd in frames_data]
        ax.set_xlim(0, max(all_times) if all_times else 10)
        ax.set_ylim(-50, 100)
        
        # Background zones
        ax.axhspan(60, 100, alpha=0.2, color='#27AE60', label='Altruistic')
        ax.axhspan(30, 60, alpha=0.2, color='#3498DB', label='Cooperative')
        ax.axhspan(0, 30, alpha=0.2, color='#F39C12', label='Individualistic')
        ax.axhspan(-50, 0, alpha=0.2, color='#E74C3C', label='Competitive')
        
        ax.axhline(45, color='white', linestyle='--', alpha=0.5)
        
        ax.set_xlabel('Time (s)', fontsize=10, color='white')
        ax.set_ylabel('SVO Angle (°)', color='white')
        ax.set_title('Independent Bidirectional SVO\n(Truck vs Cars - NOT Symmetric!)', 
                    fontsize=11, fontweight='bold', color='white')
        ax.tick_params(colors='white')
        ax.grid(True, alpha=0.2)
        
        for spine in ax.spines.values():
            spine.set_color('#4A4A6A')
    
    def _setup_aggr_balance(self):
        ax = self.axes['aggr_balance']
        ax.set_facecolor('#1A1A2E')
        
        ax.set_xlabel('Time (s)', fontsize=9, color='white')
        ax.set_ylabel('Aggressiveness', fontsize=9, color='white')
        ax.set_title('Exerted vs Suffered', fontsize=10, fontweight='bold', color='white')
        ax.tick_params(colors='white', labelsize=8)
        ax.grid(True, alpha=0.2)
        
        for spine in ax.spines.values():
            spine.set_color('#4A4A6A')
    
    def _setup_tracking_stats(self):
        ax = self.axes['tracking_stats']
        ax.set_facecolor('#1A1A2E')
        
        ax.set_xlabel('Time (s)', fontsize=9, color='white')
        ax.set_ylabel('Vehicle Count', fontsize=9, color='white')
        ax.set_title('Tracking Stability\n(Hysteresis Effect)', fontsize=10, fontweight='bold', color='white')
        ax.tick_params(colors='white', labelsize=8)
        ax.grid(True, alpha=0.2)
        
        for spine in ax.spines.values():
            spine.set_color('#4A4A6A')
    
    def _setup_info_panel(self, interaction: Dict):
        ax = self.axes['info']
        ax.set_facecolor('#1A1A2E')
        ax.axis('off')
        
        for spine in ax.spines.values():
            spine.set_color('#4A4A6A')
    
    def _init_elements(self, show_field: bool) -> Dict:
        elements = {}
        ax = self.axes['main']
        
        if show_field:
            elements['field_contour'] = None
        
        # Ego vehicle
        elements['ego'] = patches.FancyBboxPatch(
            (0, 0), 12, 2.5, boxstyle="round,pad=0.02",
            facecolor='#1A1A1A', edgecolor='white', linewidth=2,
            alpha=0.95, zorder=20
        )
        ax.add_patch(elements['ego'])
        
        # Dynamic surrounding vehicle patches (up to 15)
        elements['surrounding'] = []
        for i in range(15):
            patch = patches.FancyBboxPatch(
                (0, 0), 4.5, 1.8, boxstyle="round,pad=0.02",
                facecolor='#7F8C8D', edgecolor='white', linewidth=1.5,
                alpha=0.8, zorder=15, visible=False
            )
            ax.add_patch(patch)
            elements['surrounding'].append(patch)
        
        # Ego trail
        elements['ego_trail'], = ax.plot([], [], color='#E74C3C', alpha=0.6, linewidth=3, zorder=10)
        
        # Ego label
        elements['ego_label'] = ax.text(0, 0, '', fontsize=9, fontweight='bold',
            ha='center', va='bottom', color='white',
            bbox=dict(boxstyle='round', facecolor='#1A1A1A', alpha=0.8), zorder=25)
        
        # SVO lines
        ax_svo = self.axes['svo_chrono']
        elements['ego_svo_line'], = ax_svo.plot([], [], color='#E74C3C', linewidth=2.5,
            label='Truck SVO (ego)')
        elements['others_svo_line'], = ax_svo.plot([], [], color='#3498DB', linewidth=2.5,
            label='Cars SVO (mean)')
        elements['svo_marker_ego'], = ax_svo.plot([], [], 'o', color='white', markersize=8,
            markeredgecolor='#E74C3C', markeredgewidth=2)
        elements['svo_marker_others'], = ax_svo.plot([], [], 's', color='white', markersize=8,
            markeredgecolor='#3498DB', markeredgewidth=2)
        ax_svo.legend(loc='upper right', fontsize=8)
        
        # Aggressiveness lines
        ax_bal = self.axes['aggr_balance']
        elements['exerted_line'], = ax_bal.plot([], [], color='#E74C3C', linewidth=2, label='Exerted')
        elements['suffered_line'], = ax_bal.plot([], [], color='#3498DB', linewidth=2, label='Suffered')
        ax_bal.legend(loc='upper right', fontsize=8)
        
        # Tracking stats lines
        ax_track = self.axes['tracking_stats']
        elements['n_slot_line'], = ax_track.plot([], [], color='#27AE60', linewidth=2, label='Slot-based')
        elements['n_tracked_line'], = ax_track.plot([], [], color='#9B59B6', linewidth=2, label='Total tracked')
        ax_track.legend(loc='upper right', fontsize=8)
        
        # Info text
        elements['info_text'] = self.axes['info'].text(0.02, 0.95, '', ha='left', va='top',
            fontsize=10, color='white', family='monospace',
            transform=self.axes['info'].transAxes)
        
        return elements
    
    def _update_frame(self, frame_idx: int, frames_data: List[Dict], show_field: bool):
        fd = frames_data[frame_idx]
        ego = fd['ego']
        tracked_vehicles = fd['tracked_vehicles']
        svo_state = fd['svo_state']
        tracking_stats = fd['tracking_stats']
        
        ax = self.axes['main']
        
        # Update field
        if show_field and hasattr(self, 'field_grid_x'):
            if self.elements.get('field_contour') is not None:
                for coll in self.elements['field_contour'].collections:
                    coll.remove()
            
            field = self.field_gen.compute_field(
                self.field_grid_x, self.field_grid_y, ego, tracked_vehicles
            )
            
            self.elements['field_contour'] = ax.contourf(
                self.field_grid_x, self.field_grid_y, field,
                levels=15, cmap='Reds', alpha=0.3, zorder=1
            )
        
        # Update ego
        self._update_vehicle_patch(self.elements['ego'], ego)
        
        # Update surrounding vehicles
        for patch in self.elements['surrounding']:
            patch.set_visible(False)
        
        for i, (label, tracked) in enumerate(tracked_vehicles.items()):
            if i >= len(self.elements['surrounding']):
                break
            
            patch = self.elements['surrounding'][i]
            self._update_vehicle_patch(patch, tracked.state)
            
            # Color based on tracking status
            if tracked.is_interpolated:
                patch.set_facecolor('#95A5A6')  # Gray for interpolated
                patch.set_alpha(0.5)
            elif tracked.frames_since_slot > 0:
                patch.set_facecolor('#9B59B6')  # Purple for hysteresis
                patch.set_alpha(0.7)
            else:
                # Use position-based color
                base_label = label.replace('_', '').split('tracked')[0].split('interpolated')[0]
                color = self.config.POSITION_COLORS.get(base_label, '#7F8C8D')
                patch.set_facecolor(color)
                patch.set_alpha(0.8)
            
            patch.set_visible(True)
        
        # Update trail
        if len(fd['ego_history']) > 1:
            self.elements['ego_trail'].set_data(fd['ego_history'][:, 0], fd['ego_history'][:, 1])
        
        # Update ego label
        self.elements['ego_label'].set_position((ego.x, ego.y + ego.width/2 + 3))
        self.elements['ego_label'].set_text(f'{ego.vehicle_class.title()}\n{ego.speed*3.6:.0f} km/h')
        
        # Update SVO chronograph
        self.elements['ego_svo_line'].set_data(fd['all_times'], fd['all_ego_svo'])
        self.elements['others_svo_line'].set_data(fd['all_times'], fd['all_others_svo'])
        self.elements['svo_marker_ego'].set_data([fd['time']], [svo_state.mean_ego_svo])
        self.elements['svo_marker_others'].set_data([fd['time']], [svo_state.mean_others_svo])
        
        # Update aggressiveness balance
        ax_bal = self.axes['aggr_balance']
        ax_bal.set_xlim(0, max(fd['all_times']) + 0.1)
        max_aggr = max(max(fd['all_exerted'] + [1]), max(fd['all_suffered'] + [1]))
        ax_bal.set_ylim(0, max_aggr * 1.1)
        
        self.elements['exerted_line'].set_data(fd['all_times'], fd['all_exerted'])
        self.elements['suffered_line'].set_data(fd['all_times'], fd['all_suffered'])
        
        # Update tracking stats
        ax_track = self.axes['tracking_stats']
        ax_track.set_xlim(0, max(fd['all_times']) + 0.1)
        ax_track.set_ylim(0, max(max(fd['all_n_tracked']) + 1, 8))
        
        self.elements['n_slot_line'].set_data(fd['all_times'], fd['all_n_slot'])
        self.elements['n_tracked_line'].set_data(fd['all_times'], fd['all_n_tracked'])
        
        # Update info text
        n_surrounding = len(tracked_vehicles)
        
        info_text = (
            f"━━━ Frame {fd['frame']} | Time: {fd['time']:.2f}s ━━━\n\n"
            f"Ego Vehicle: {ego.vehicle_class.title()} (ID: {ego.track_id})\n"
            f"  Speed: {ego.speed*3.6:.1f} km/h\n"
            f"  Mass: {ego.mass/1000:.1f} tons\n\n"
            f"━━━ Tracking Stats ━━━\n"
            f"  Slot-detected: {tracking_stats['slot_detected']}\n"
            f"  Hysteresis-tracked: {tracking_stats['hysteresis_tracked']}\n"
            f"  Interpolated: {tracking_stats['interpolated']}\n"
            f"  Total: {tracking_stats['total']}\n\n"
            f"━━━ Aggressiveness ━━━\n"
            f"  Total Exerted: {svo_state.total_exerted:.1f}\n"
            f"  Total Suffered: {svo_state.total_suffered:.1f}\n\n"
            f"━━━ Independent SVO ━━━\n"
            f"  Truck (ego) SVO: {svo_state.mean_ego_svo:.1f}°\n"
            f"  Cars (mean) SVO: {svo_state.mean_others_svo:.1f}°\n"
            f"  Asymmetry: {svo_state.mean_others_svo - svo_state.mean_ego_svo:.1f}°\n"
        )
        
        # Zone interpretation
        if svo_state.mean_ego_svo > 60:
            zone = "ALTRUISTIC"
        elif svo_state.mean_ego_svo > 30:
            zone = "COOPERATIVE"
        elif svo_state.mean_ego_svo > 0:
            zone = "INDIVIDUALISTIC"
        else:
            zone = "COMPETITIVE"
        
        info_text += f"\n  Truck Zone: {zone}"
        
        self.elements['info_text'].set_text(info_text)
        
        return list(self.elements.values())
    
    def _update_vehicle_patch(self, patch, vehicle: VehicleState):
        half_l = vehicle.length / 2
        half_w = vehicle.width / 2
        
        cos_h = np.cos(vehicle.heading)
        sin_h = np.sin(vehicle.heading)
        
        corner_x = vehicle.x - half_l * cos_h + half_w * sin_h
        corner_y = vehicle.y - half_l * sin_h - half_w * cos_h
        
        patch.set_bounds(corner_x, corner_y, vehicle.length, vehicle.width)
        
        t = plt.matplotlib.transforms.Affine2D().rotate_around(
            vehicle.x, vehicle.y, vehicle.heading
        ) + self.axes['main'].transData
        patch.set_transform(t)
    
    def create_static_analysis(
        self,
        interaction: Dict,
        save_path: Optional[str] = None
    ):
        """Create static analysis plots."""
        
        self.tracker.reset()
        self.svo_calc.reset()
        frames_data = self._prepare_frames(interaction)
        
        if not frames_data:
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.patch.set_facecolor('#0D1117')
        
        times = [fd['time'] for fd in frames_data]
        ego_svo = [fd['svo_state'].mean_ego_svo for fd in frames_data]
        others_svo = [fd['svo_state'].mean_others_svo for fd in frames_data]
        exerted = [fd['svo_state'].total_exerted for fd in frames_data]
        suffered = [fd['svo_state'].total_suffered for fd in frames_data]
        n_slot = [fd['tracking_stats']['slot_detected'] for fd in frames_data]
        n_tracked = [fd['tracking_stats']['total'] for fd in frames_data]
        
        # Plot 1: SVO comparison
        ax = axes[0, 0]
        ax.set_facecolor('#1A1A2E')
        ax.axhspan(60, 100, alpha=0.2, color='#27AE60')
        ax.axhspan(30, 60, alpha=0.2, color='#3498DB')
        ax.axhspan(0, 30, alpha=0.2, color='#F39C12')
        ax.axhspan(-50, 0, alpha=0.2, color='#E74C3C')
        ax.plot(times, ego_svo, color='#E74C3C', linewidth=2, label='Truck SVO')
        ax.plot(times, others_svo, color='#3498DB', linewidth=2, label='Cars SVO (mean)')
        ax.axhline(45, color='white', linestyle='--', alpha=0.5)
        ax.set_xlabel('Time (s)', color='white')
        ax.set_ylabel('SVO Angle (°)', color='white')
        ax.set_title('Independent Bidirectional SVO', fontsize=12, fontweight='bold', color='white')
        ax.tick_params(colors='white')
        ax.set_ylim(-50, 100)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.2)
        
        # Plot 2: Asymmetry
        ax = axes[0, 1]
        ax.set_facecolor('#1A1A2E')
        asymmetry = [o - e for e, o in zip(ego_svo, others_svo)]
        ax.fill_between(times, asymmetry, alpha=0.3, color='#9B59B6')
        ax.plot(times, asymmetry, color='#9B59B6', linewidth=2)
        ax.axhline(0, color='white', linestyle='--', alpha=0.5)
        ax.set_xlabel('Time (s)', color='white')
        ax.set_ylabel('Cars SVO - Truck SVO (°)', color='white')
        ax.set_title('SVO Asymmetry\n(Positive = Cars more cooperative)', 
                    fontsize=12, fontweight='bold', color='white')
        ax.tick_params(colors='white')
        ax.grid(True, alpha=0.2)
        
        # Plot 3: Aggressiveness
        ax = axes[0, 2]
        ax.set_facecolor('#1A1A2E')
        ax.plot(times, exerted, color='#E74C3C', linewidth=2, label='Exerted (ego→others)')
        ax.plot(times, suffered, color='#3498DB', linewidth=2, label='Suffered (others→ego)')
        ax.fill_between(times, exerted, alpha=0.3, color='#E74C3C')
        ax.fill_between(times, suffered, alpha=0.3, color='#3498DB')
        ax.set_xlabel('Time (s)', color='white')
        ax.set_ylabel('Aggressiveness', color='white')
        ax.set_title('Asymmetric Aggressiveness', fontsize=12, fontweight='bold', color='white')
        ax.tick_params(colors='white')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.2)
        
        # Plot 4: Tracking stability
        ax = axes[1, 0]
        ax.set_facecolor('#1A1A2E')
        ax.fill_between(times, n_tracked, alpha=0.3, color='#9B59B6')
        ax.plot(times, n_tracked, color='#9B59B6', linewidth=2, label='Total tracked')
        ax.plot(times, n_slot, color='#27AE60', linewidth=2, linestyle='--', label='Slot-based only')
        ax.set_xlabel('Time (s)', color='white')
        ax.set_ylabel('Vehicle Count', color='white')
        ax.set_title('Tracking Stability (Hysteresis Effect)', 
                    fontsize=12, fontweight='bold', color='white')
        ax.tick_params(colors='white')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.2)
        
        # Plot 5: SVO distribution
        ax = axes[1, 1]
        ax.set_facecolor('#1A1A2E')
        ax.hist(ego_svo, bins=20, alpha=0.7, color='#E74C3C', 
               label=f'Truck (μ={np.mean(ego_svo):.1f}°)')
        ax.hist(others_svo, bins=20, alpha=0.7, color='#3498DB',
               label=f'Cars (μ={np.mean(others_svo):.1f}°)')
        ax.axvline(45, color='white', linestyle='--', alpha=0.5)
        ax.set_xlabel('SVO Angle (°)', color='white')
        ax.set_ylabel('Frequency', color='white')
        ax.set_title('SVO Distributions', fontsize=12, fontweight='bold', color='white')
        ax.tick_params(colors='white')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.2)
        
        # Plot 6: Statistics summary
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
            f"Hysteresis benefit: +{np.mean(n_tracked) - np.mean(n_slot):.1f}\n\n"
            "─── Key Finding ───\n"
            f"Asymmetry: {np.mean(asymmetry):.1f}°\n"
            f"(Cars {abs(np.mean(asymmetry)):.1f}° more "
            f"{'cooperative' if np.mean(asymmetry) > 0 else 'competitive'})"
        )
        
        ax.text(0.1, 0.9, stats_text, transform=ax.transAxes,
               fontsize=10, color='white', family='monospace',
               verticalalignment='top')
        
        for row in axes:
            for a in row:
                for spine in a.spines.values():
                    spine.set_color('#4A4A6A')
        
        fig.suptitle(f'Complete SVO Analysis - {interaction["ego_class"].title()} (ID: {interaction["ego_id"]})',
                    fontsize=14, fontweight='bold', color='white', y=0.98)
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
            logger.info(f"Saved analysis to {save_path}")
            plt.close(fig)
        else:
            plt.show()


# =============================================================================
# Main
# =============================================================================

def main(data_dir: str, recording_id: int, output_dir: str = './output_complete'):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 70)
    logger.info("Complete exiD SVO Visualization")
    logger.info("  - Animation with Hysteresis Tracking")
    logger.info("  - Independent Bidirectional SVO")
    logger.info("=" * 70)
    
    loader = ExiDLoader(data_dir)
    if not loader.load_recording(recording_id):
        return None
    
    detector = InteractionDetector(loader)
    interactions = detector.find_interactions()
    
    if not interactions:
        logger.warning("No interactions found.")
        return None
    
    logger.info(f"\nFound {len(interactions)} interactions")
    for i, inter in enumerate(interactions[:5]):
        logger.info(f"  {i+1}. {inter['ego_class']} (ID: {inter['ego_id']}) - "
                   f"{len(inter['frames'])} frames")
    
    best = interactions[0]
    logger.info(f"\nVisualizing: {best['ego_class']} (ID: {best['ego_id']})")
    
    visualizer = CompleteVisualizer(loader)
    
    # Create animation
    anim_path = output_path / f'recording_{recording_id}_complete.gif'
    ani = visualizer.animate_interaction(best, save_path=str(anim_path), show_field=True)
    
    # Create static analysis
    analysis_path = output_path / f'recording_{recording_id}_analysis.png'
    visualizer.create_static_analysis(best, save_path=str(analysis_path))
    
    logger.info(f"\n✓ Outputs saved to: {output_path}")
    logger.info("\nGenerated files:")
    logger.info(f"  - {anim_path.name}  (Animation)")
    logger.info(f"  - {analysis_path.name}  (Static Analysis)")
    
    return interactions


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Complete exiD SVO Visualization')
    parser.add_argument('--data_dir', type=str, default='C:\\exiD-tools\\data')
    parser.add_argument('--recording', type=int, default=25)
    parser.add_argument('--output_dir', type=str, default='./output_complete')
    
    args = parser.parse_args()
    main(args.data_dir, args.recording, args.output_dir)