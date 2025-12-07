"""
exiD Dataset: Asymmetric Aggressiveness & Multi-Agent SVO Visualization
========================================================================
Based on:
- Hu et al. (2025) "Socially Game-Theoretic Lane-Change for Autonomous Heavy 
  Vehicle based on Asymmetric Driving Aggressiveness" - IEEE TVT
- Rasidescu & Taghavifar (2024) - APF + SVO Framework

Key Features:
- Asymmetric aggressiveness model (Ω_{i→j} ≠ Ω_{j→i})
- 6-agent interaction visualization (lead, rear, left/right neighbors)
- Physics-informed SVO with proper smoothing
- Potential field visualization

Usage:
    python exid_asymmetric_aggressiveness_viz.py --data_dir "path/to/exiD" --recording 25
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap, Normalize, to_rgba
from matplotlib.cm import ScalarMappable
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Set
import warnings
import argparse
import logging
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter

warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class AsymmetricConfig:
    """Configuration based on Hu et al. (2025) asymmetric aggressiveness model."""
    
    # Vehicle classification
    HEAVY_VEHICLE_CLASSES: Set[str] = field(default_factory=lambda: {'truck', 'bus', 'van', 'trailer'})
    CAR_CLASSES: Set[str] = field(default_factory=lambda: {'car'})
    
    # Asymmetric Aggressiveness Parameters (Table II from Hu et al.)
    MU_1: float = 0.2               # Velocity angle coefficient for AGV
    MU_2: float = 0.21              # Velocity angle coefficient for SFV
    SIGMA: float = 0.1              # Distance decay coefficient
    DELTA: float = 0.0005           # Mass normalization factor
    TAU_1: float = 0.2              # Longitudinal safe distance threshold
    TAU_2: float = 0.1              # Lateral safe distance threshold
    BETA: float = 0.05              # Shape coefficient for ellipse
    
    # Vehicle masses (kg) - from Table II
    MASS_HV: float = 15000.0        # Heavy vehicle mass (15 tons)
    MASS_PC: float = 3000.0         # Passenger car mass (3 tons)
    
    # Interaction detection
    ROI_RADIUS: float = 80.0        # Region of interest radius
    MIN_INTERACTION_FRAMES: int = 50
    
    # Smoothing parameters (to prevent abrupt SVO jumps)
    SMOOTH_WINDOW: int = 25         # 1 second at 25fps
    SMOOTH_POLY_ORDER: int = 3      # Savitzky-Golay polynomial order
    MAX_SVO_DELTA: float = 3.0      # Max SVO change per frame (degrees)
    
    # SVO calculation parameters
    SVO_BASE: float = 45.0          # Neutral SVO angle
    AGGR_TO_SVO_SCALE: float = 30.0 # Scale aggressiveness to SVO degrees
    
    # Visualization
    TRAIL_LENGTH: int = 75
    FPS: int = 25
    GRID_RESOLUTION: int = 60
    
    # Surrounding vehicle slots (from exiD dataset)
    SURROUNDING_SLOTS: List[str] = field(default_factory=lambda: [
        'leadId', 'rearId', 
        'leftLeadId', 'leftAlongsideId', 'leftRearId',
        'rightLeadId', 'rightAlongsideId', 'rightRearId'
    ])
    
    # Colors for different positions
    POSITION_COLORS: Dict[str, str] = field(default_factory=lambda: {
        'lead': '#E74C3C',           # Red - front
        'rear': '#3498DB',           # Blue - behind
        'leftLead': '#9B59B6',       # Purple - left front
        'leftAlongside': '#E67E22',  # Orange - left side
        'leftRear': '#1ABC9C',       # Teal - left rear
        'rightLead': '#F1C40F',      # Yellow - right front
        'rightAlongside': '#2ECC71', # Green - right side
        'rightRear': '#34495E',      # Dark - right rear
    })
    
    COLORS: Dict[str, str] = field(default_factory=lambda: {
        'ego_truck': '#1A1A1A',
        'ego_car': '#3498DB',
        'surrounding': '#7F8C8D',
        'trail': '#E74C3C',
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
    heading: float  # radians
    width: float
    length: float
    x_velocity: float
    y_velocity: float
    x_acceleration: float
    y_acceleration: float
    vehicle_class: str
    mass: float = 3000.0  # Default mass
    
    @property
    def speed(self) -> float:
        return np.sqrt(self.x_velocity**2 + self.y_velocity**2)
    
    @property
    def velocity_angle(self) -> float:
        """Angle of velocity vector in world frame."""
        return np.arctan2(self.y_velocity, self.x_velocity)


@dataclass
class AggressivenessResult:
    """Result of asymmetric aggressiveness calculation."""
    aggressor_id: int
    sufferer_id: int
    aggressiveness: float           # Ω_{i→j}
    position_label: str             # e.g., 'lead', 'leftAlongside'
    distance: float
    relative_velocity: float


@dataclass
class MultiAgentSVOState:
    """SVO state considering multiple surrounding agents."""
    ego_id: int
    frame: int
    
    # Aggressiveness FROM ego TO others (ego as aggressor)
    ego_to_others: List[AggressivenessResult] = field(default_factory=list)
    
    # Aggressiveness FROM others TO ego (ego as sufferer)
    others_to_ego: List[AggressivenessResult] = field(default_factory=list)
    
    # Aggregate metrics
    total_suffered: float = 0.0     # Total aggressiveness ego suffers
    total_exerted: float = 0.0      # Total aggressiveness ego exerts
    
    # SVO angle (derived from balance of suffered vs exerted)
    svo_angle: float = 45.0
    cooperativeness: float = 0.5    # [0, 1]
    
    @property
    def asymmetry_ratio(self) -> float:
        """Ratio of exerted to suffered aggressiveness."""
        if self.total_suffered < 1e-6:
            return 1.0
        return self.total_exerted / self.total_suffered


# =============================================================================
# Asymmetric Aggressiveness Calculator (Hu et al. 2025)
# =============================================================================

class AsymmetricAggressivenessCalculator:
    """
    Implements the asymmetric driving aggressiveness model from Hu et al. (2025).
    
    Core equation (Eq. 12):
    Ω_{i→j} = (m_i |v_i|) / (2δ m_j) * e^(ξ_1 + ξ_2)
    
    Where:
    - ξ_1 = μ_1 |v_i| cos θ_i + μ_2 |v_j| cos θ_j  (velocity-angle term)
    - ξ_2 = -σ m_i^{-1} r_{ij}  (distance decay term)
    - θ_i, θ_j are angles between velocity vectors and position unit vector
    - r_{ij} is the pseudo-distance (anisotropic)
    """
    
    def __init__(self, config: AsymmetricConfig):
        self.config = config
    
    def compute_aggressiveness(
        self,
        aggressor: VehicleState,
        sufferer: VehicleState,
        position_label: str = 'unknown'
    ) -> AggressivenessResult:
        """
        Compute asymmetric aggressiveness from aggressor (AGV) to sufferer (SFV).
        
        Note: Ω_{i→j} ≠ Ω_{j→i} due to mass asymmetry and position effects.
        """
        
        # Position vector from aggressor to sufferer
        dx = sufferer.x - aggressor.x
        dy = sufferer.y - aggressor.y
        euclidean_dist = np.sqrt(dx**2 + dy**2)
        
        if euclidean_dist < 0.1:  # Avoid division by zero
            return AggressivenessResult(
                aggressor_id=aggressor.track_id,
                sufferer_id=sufferer.track_id,
                aggressiveness=0.0,
                position_label=position_label,
                distance=euclidean_dist,
                relative_velocity=0.0
            )
        
        # Unit position vector
        pos_unit_x = dx / euclidean_dist
        pos_unit_y = dy / euclidean_dist
        
        # Velocities
        v_i = aggressor.speed
        v_j = sufferer.speed
        
        # Angles between velocity vectors and position vector
        # θ_i: angle for aggressor's velocity relative to position vector
        if v_i > 0.1:
            cos_theta_i = (aggressor.x_velocity * pos_unit_x + 
                         aggressor.y_velocity * pos_unit_y) / v_i
        else:
            cos_theta_i = 0.0
        
        # θ_j: angle for sufferer's velocity relative to REVERSE position vector
        if v_j > 0.1:
            cos_theta_j = -(sufferer.x_velocity * pos_unit_x + 
                          sufferer.y_velocity * pos_unit_y) / v_j
        else:
            cos_theta_j = 0.0
        
        # Clamp cosines
        cos_theta_i = np.clip(cos_theta_i, -1, 1)
        cos_theta_j = np.clip(cos_theta_j, -1, 1)
        
        # Pseudo-distance with ellipse equation (Eq. 11)
        # Accounts for vehicle heading - larger in longitudinal direction
        r_ij = self._compute_pseudo_distance(
            aggressor, sufferer, euclidean_dist
        )
        
        # ξ_1: Velocity-angle term (Eq. 10)
        xi_1 = (self.config.MU_1 * v_i * cos_theta_i + 
                self.config.MU_2 * v_j * cos_theta_j)
        
        # ξ_2: Distance decay term
        xi_2 = -self.config.SIGMA * (aggressor.mass ** -1) * r_ij
        
        # Aggressiveness (Eq. 12)
        mass_term = (aggressor.mass * v_i) / (2 * self.config.DELTA * sufferer.mass)
        omega = mass_term * np.exp(xi_1 + xi_2)
        
        # Normalize to reasonable range [0, 1000]
        omega = np.clip(omega, 0, 1000)
        
        # Relative velocity (approaching is positive)
        rel_vel = -(aggressor.x_velocity - sufferer.x_velocity) * pos_unit_x - \
                   (aggressor.y_velocity - sufferer.y_velocity) * pos_unit_y
        
        return AggressivenessResult(
            aggressor_id=aggressor.track_id,
            sufferer_id=sufferer.track_id,
            aggressiveness=omega,
            position_label=position_label,
            distance=euclidean_dist,
            relative_velocity=rel_vel
        )
    
    def _compute_pseudo_distance(
        self,
        aggressor: VehicleState,
        sufferer: VehicleState,
        euclidean_dist: float
    ) -> float:
        """
        Compute anisotropic pseudo-distance using ellipse equation (Eq. 11).
        
        The pseudo-distance is larger in the longitudinal direction (heading)
        to account for the elongated shape of vehicles.
        """
        
        dx = sufferer.x - aggressor.x
        dy = sufferer.y - aggressor.y
        
        phi = aggressor.heading  # Aggressor's heading angle
        
        # Rotate to aggressor's local frame
        cos_phi = np.cos(phi)
        sin_phi = np.sin(phi)
        
        # Local coordinates (x_local is longitudinal, y_local is lateral)
        x_local = dx * cos_phi + dy * sin_phi
        y_local = -dx * sin_phi + dy * cos_phi
        
        # Acceleration term for dynamic expansion
        accel = np.sqrt(aggressor.x_acceleration**2 + aggressor.y_acceleration**2)
        t0 = 1.0  # Acceleration influence coefficient
        
        # Dynamic expansion factor based on velocity
        v_i = aggressor.speed
        expansion = np.exp(2 * self.config.BETA * (v_i + accel * t0))
        
        # Ellipse semi-axes (larger in longitudinal direction)
        tau_1_scaled = self.config.TAU_1 * expansion
        tau_2_scaled = self.config.TAU_2
        
        # Pseudo-distance using ellipse equation
        if tau_1_scaled > 0 and tau_2_scaled > 0:
            r_pseudo = np.sqrt(
                (x_local**2) / (tau_1_scaled**2) + 
                (y_local**2) / (tau_2_scaled**2)
            )
        else:
            r_pseudo = euclidean_dist
        
        # Scale back to physical units
        r_pseudo = r_pseudo * min(tau_1_scaled, tau_2_scaled)
        
        return max(r_pseudo, 0.1)  # Minimum distance


class MultiAgentSVOCalculator:
    """
    Calculates SVO considering interactions with all 6 surrounding vehicles.
    
    SVO is derived from the balance of:
    - Aggressiveness the ego vehicle EXERTS on others
    - Aggressiveness the ego vehicle SUFFERS from others
    
    Higher exerted → lower SVO (more competitive/egoistic)
    Higher suffered → higher SVO (more cooperative/yielding)
    """
    
    def __init__(self, config: AsymmetricConfig):
        self.config = config
        self.aggr_calc = AsymmetricAggressivenessCalculator(config)
        
        # History for smoothing
        self.svo_history: List[float] = []
    
    def compute_multi_agent_svo(
        self,
        ego: VehicleState,
        surrounding: Dict[str, Optional[VehicleState]]
    ) -> MultiAgentSVOState:
        """
        Compute SVO state considering all surrounding vehicles.
        
        Args:
            ego: Ego vehicle state
            surrounding: Dict mapping position labels to vehicle states
        """
        
        state = MultiAgentSVOState(ego_id=ego.track_id, frame=ego.frame)
        
        for pos_label, other in surrounding.items():
            if other is None:
                continue
            
            # Aggressiveness from ego TO other (ego as aggressor)
            ego_to_other = self.aggr_calc.compute_aggressiveness(
                ego, other, pos_label
            )
            state.ego_to_others.append(ego_to_other)
            state.total_exerted += ego_to_other.aggressiveness
            
            # Aggressiveness from other TO ego (ego as sufferer)
            other_to_ego = self.aggr_calc.compute_aggressiveness(
                other, ego, pos_label
            )
            state.others_to_ego.append(other_to_ego)
            state.total_suffered += other_to_ego.aggressiveness
        
        # Compute SVO from aggressiveness balance
        state.svo_angle = self._compute_svo_from_aggressiveness(state)
        state.cooperativeness = (state.svo_angle + 45) / 135  # Map [-45, 90] to [0, 1]
        
        return state
    
    def _compute_svo_from_aggressiveness(self, state: MultiAgentSVOState) -> float:
        """
        Derive SVO angle from the balance of exerted vs suffered aggressiveness.
        
        Logic:
        - If ego exerts MORE aggressiveness than suffers → competitive (lower SVO)
        - If ego suffers MORE aggressiveness than exerts → cooperative (higher SVO)
        - Balance → neutral (45°)
        """
        
        total = state.total_exerted + state.total_suffered
        
        if total < 1e-6:
            return self.config.SVO_BASE  # No interaction → neutral
        
        # Ratio of suffered to total
        # High ratio (suffer more) → cooperative → high SVO
        # Low ratio (exert more) → competitive → low SVO
        suffer_ratio = state.total_suffered / total
        
        # Map [0, 1] to [-45, 90] degrees
        # 0.5 → 45° (neutral)
        # 1.0 → 90° (altruistic)
        # 0.0 → -45° (competitive)
        raw_svo = -45 + 135 * suffer_ratio
        
        # Apply smoothing to prevent abrupt jumps
        smoothed_svo = self._smooth_svo(raw_svo)
        
        return smoothed_svo
    
    def _smooth_svo(self, raw_svo: float) -> float:
        """Apply temporal smoothing to prevent abrupt SVO jumps."""
        
        self.svo_history.append(raw_svo)
        
        # Keep limited history
        max_history = self.config.SMOOTH_WINDOW * 2
        if len(self.svo_history) > max_history:
            self.svo_history = self.svo_history[-max_history:]
        
        if len(self.svo_history) < 3:
            return raw_svo
        
        # Apply Savitzky-Golay filter for smooth derivative
        window = min(len(self.svo_history), self.config.SMOOTH_WINDOW)
        if window % 2 == 0:
            window -= 1
        if window < 3:
            window = 3
        
        try:
            smoothed = savgol_filter(
                self.svo_history, 
                window, 
                self.config.SMOOTH_POLY_ORDER
            )
            result = smoothed[-1]
        except:
            result = np.mean(self.svo_history[-5:])
        
        # Rate limiting - prevent jumps larger than MAX_SVO_DELTA
        if len(self.svo_history) > 1:
            prev_svo = self.svo_history[-2]
            delta = result - prev_svo
            if abs(delta) > self.config.MAX_SVO_DELTA:
                result = prev_svo + np.sign(delta) * self.config.MAX_SVO_DELTA
        
        return np.clip(result, -45, 90)
    
    def reset(self):
        """Reset history for new interaction."""
        self.svo_history = []


# =============================================================================
# Potential Field Generator
# =============================================================================

class AsymmetricPotentialFieldGenerator:
    """
    Generates potential fields based on asymmetric aggressiveness.
    
    Unlike standard APF which uses symmetric Gaussian fields,
    this creates anisotropic fields that reflect the asymmetric
    aggressiveness relationship.
    """
    
    def __init__(self, config: AsymmetricConfig):
        self.config = config
        self.aggr_calc = AsymmetricAggressivenessCalculator(config)
    
    def compute_field(
        self,
        x_grid: np.ndarray,
        y_grid: np.ndarray,
        ego: VehicleState,
        surrounding: Dict[str, Optional[VehicleState]]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute two potential fields:
        1. Field experienced BY ego (from surrounding vehicles)
        2. Field exerted BY ego (on surrounding vehicles)
        """
        
        field_suffered = np.zeros_like(x_grid)
        field_exerted = np.zeros_like(x_grid)
        
        # For each grid point, compute the aggressiveness
        # This is computationally expensive, so we use a simplified approach
        
        # Field suffered by ego: sum of repulsive potentials from others
        for pos_label, other in surrounding.items():
            if other is None:
                continue
            
            # Simplified: use Gaussian centered on other vehicle
            # with anisotropic spread based on their velocity
            other_field = self._compute_vehicle_field(
                x_grid, y_grid, other, is_aggressor=True
            )
            field_suffered += other_field
        
        # Field exerted by ego
        ego_field = self._compute_vehicle_field(
            x_grid, y_grid, ego, is_aggressor=True
        )
        field_exerted = ego_field
        
        return field_suffered, field_exerted
    
    def _compute_vehicle_field(
        self,
        x_grid: np.ndarray,
        y_grid: np.ndarray,
        vehicle: VehicleState,
        is_aggressor: bool = True
    ) -> np.ndarray:
        """
        Compute anisotropic potential field for a single vehicle.
        
        Based on Eq. 11 - uses ellipse equation with dynamic expansion.
        """
        
        dx = x_grid - vehicle.x
        dy = y_grid - vehicle.y
        
        # Rotate to vehicle's local frame
        cos_h = np.cos(vehicle.heading)
        sin_h = np.sin(vehicle.heading)
        
        dx_local = dx * cos_h + dy * sin_h   # Longitudinal
        dy_local = -dx * sin_h + dy * cos_h  # Lateral
        
        # Dynamic sigma based on velocity and mass
        v = vehicle.speed
        m = vehicle.mass
        
        # Forward extension is larger (no-zone effect)
        sigma_x_fwd = max(vehicle.length, 5.0) + v * 1.5
        sigma_x_bwd = max(vehicle.length, 5.0) * 0.5
        sigma_y = max(vehicle.width, 2.0) * 1.5
        
        # Asymmetric sigma in x direction
        sigma_x = np.where(dx_local > 0, sigma_x_fwd, sigma_x_bwd)
        
        # Mass-scaled amplitude
        if is_aggressor:
            amplitude = m / self.config.MASS_PC * 10.0
        else:
            amplitude = 5.0
        
        # Anisotropic Gaussian
        exponent = -((dx_local**2) / (2 * sigma_x**2) + 
                     (dy_local**2) / (2 * sigma_y**2))
        
        field = amplitude * np.exp(exponent)
        
        return field


# =============================================================================
# Data Loader
# =============================================================================

class ExiDLoader:
    """Loads exiD dataset with surrounding vehicle information."""
    
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
        self.tracks_df: Optional[pd.DataFrame] = None
        self.tracks_meta_df: Optional[pd.DataFrame] = None
        self.recording_meta: Optional[pd.Series] = None
        self.background_image: Optional[np.ndarray] = None
        self.ortho_px_to_meter: float = 0.1
        self.current_recording: Optional[int] = None
        self.config = AsymmetricConfig()
    
    def load_recording(self, recording_id: int) -> bool:
        """Load a specific recording."""
        self.current_recording = recording_id
        prefix = f"{recording_id:02d}_"
        
        tracks_path = self.data_dir / f"{prefix}tracks.csv"
        meta_path = self.data_dir / f"{prefix}tracksMeta.csv"
        rec_meta_path = self.data_dir / f"{prefix}recordingMeta.csv"
        background_path = self.data_dir / f"{prefix}background.png"
        
        try:
            logger.info(f"Loading recording {recording_id}...")
            
            self.tracks_df = pd.read_csv(tracks_path)
            self.tracks_meta_df = pd.read_csv(meta_path)
            rec_meta_df = pd.read_csv(rec_meta_path)
            self.recording_meta = rec_meta_df.iloc[0]
            
            self.ortho_px_to_meter = self.recording_meta.get('orthoPxToMeter', 0.1)
            
            if background_path.exists():
                self.background_image = plt.imread(str(background_path))
            
            # Merge metadata
            self.tracks_df = self.tracks_df.merge(
                self.tracks_meta_df[['trackId', 'class', 'width', 'length']],
                on='trackId', how='left', suffixes=('', '_meta')
            )
            
            if 'width_meta' in self.tracks_df.columns:
                self.tracks_df['width'] = self.tracks_df['width'].fillna(self.tracks_df['width_meta'])
                self.tracks_df['length'] = self.tracks_df['length'].fillna(self.tracks_df['length_meta'])
            
            # Log summary
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
    
    def get_vehicle_state(self, track_id: int, frame: int) -> Optional[VehicleState]:
        """Get vehicle state at a specific frame."""
        row = self.tracks_df[
            (self.tracks_df['trackId'] == track_id) & 
            (self.tracks_df['frame'] == frame)
        ]
        
        if row.empty:
            return None
        
        row = row.iloc[0]
        vehicle_class = str(row.get('class', 'car')).lower()
        
        # Assign mass based on vehicle class
        if vehicle_class in self.config.HEAVY_VEHICLE_CLASSES:
            mass = self.config.MASS_HV
        else:
            mass = self.config.MASS_PC
        
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
    
    def get_surrounding_vehicles(
        self, track_id: int, frame: int
    ) -> Dict[str, Optional[VehicleState]]:
        """Get all surrounding vehicles for a given ego vehicle."""
        
        row = self.tracks_df[
            (self.tracks_df['trackId'] == track_id) & 
            (self.tracks_df['frame'] == frame)
        ]
        
        if row.empty:
            return {}
        
        row = row.iloc[0]
        surrounding = {}
        
        # Map exiD column names to position labels
        slot_mapping = {
            'leadId': 'lead',
            'rearId': 'rear',
            'leftLeadId': 'leftLead',
            'leftAlongsideId': 'leftAlongside',
            'leftRearId': 'leftRear',
            'rightLeadId': 'rightLead',
            'rightAlongsideId': 'rightAlongside',
            'rightRearId': 'rightRear',
        }
        
        for col, label in slot_mapping.items():
            if col in row.index:
                other_id = row[col]
                # Handle both string and numeric ID formats
                try:
                    if pd.notna(other_id):
                        # Convert to numeric, handling strings like "123" or empty strings
                        other_id_num = pd.to_numeric(other_id, errors='coerce')
                        if pd.notna(other_id_num) and other_id_num > 0:
                            other_state = self.get_vehicle_state(int(other_id_num), frame)
                            surrounding[label] = other_state
                        else:
                            surrounding[label] = None
                    else:
                        surrounding[label] = None
                except (ValueError, TypeError):
                    surrounding[label] = None
            else:
                surrounding[label] = None
        
        return surrounding
    
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

class MultiAgentInteractionDetector:
    """Detects interactions involving heavy vehicles with multiple surrounding agents."""
    
    def __init__(self, loader: ExiDLoader):
        self.loader = loader
        self.config = AsymmetricConfig()
    
    def find_interactions(self) -> List[Dict]:
        """Find all heavy vehicle interactions."""
        
        heavy_vehicles = self.loader.get_heavy_vehicles()
        interactions = []
        
        for _, hv_row in heavy_vehicles.iterrows():
            hv_id = hv_row['trackId']
            hv_track = self.loader.get_track_trajectory(hv_id)
            
            if len(hv_track) < self.config.MIN_INTERACTION_FRAMES:
                continue
            
            # Find frames with multiple surrounding vehicles
            interaction_segments = self._find_dense_segments(hv_id, hv_track)
            
            for segment in interaction_segments:
                interactions.append({
                    'ego_id': hv_id,
                    'ego_class': hv_row['class'],
                    'frames': segment,
                    'start_frame': segment[0],
                    'end_frame': segment[-1],
                })
        
        # Sort by interaction density
        interactions.sort(key=lambda x: len(x['frames']), reverse=True)
        
        return interactions
    
    def _find_dense_segments(
        self, ego_id: int, ego_track: pd.DataFrame
    ) -> List[List[int]]:
        """Find segments with high interaction density (multiple surrounding vehicles)."""
        
        frames = ego_track['frame'].values
        dense_frames = []
        
        for frame in frames:
            surrounding = self.loader.get_surrounding_vehicles(ego_id, frame)
            n_vehicles = sum(1 for v in surrounding.values() if v is not None)
            
            if n_vehicles >= 2:  # At least 2 surrounding vehicles
                dense_frames.append(frame)
        
        if len(dense_frames) < self.config.MIN_INTERACTION_FRAMES:
            return []
        
        # Group into contiguous segments
        segments = []
        current = [dense_frames[0]]
        
        for i in range(1, len(dense_frames)):
            if dense_frames[i] - dense_frames[i-1] <= 10:
                current.append(dense_frames[i])
            else:
                if len(current) >= self.config.MIN_INTERACTION_FRAMES:
                    segments.append(current)
                current = [dense_frames[i]]
        
        if len(current) >= self.config.MIN_INTERACTION_FRAMES:
            segments.append(current)
        
        return segments


# =============================================================================
# Visualizer
# =============================================================================

class AsymmetricAggressivenessVisualizer:
    """
    Multi-panel dashboard visualizing asymmetric aggressiveness and SVO.
    
    Panels:
    1. Bird's Eye View with surrounding vehicles and APF
    2. SVO Chronograph (time-series)
    3. Aggressiveness Balance (exerted vs suffered)
    4. Per-Agent Aggressiveness breakdown
    """
    
    def __init__(self, loader: ExiDLoader):
        self.loader = loader
        self.config = AsymmetricConfig()
        self.svo_calc = MultiAgentSVOCalculator(self.config)
        self.field_gen = AsymmetricPotentialFieldGenerator(self.config)
        
        self.fig = None
        self.axes = {}
        self.elements = {}
    
    def animate_interaction(
        self,
        interaction: Dict,
        save_path: Optional[str] = None,
        show_field: bool = True
    ) -> Optional[animation.FuncAnimation]:
        """Create multi-panel dashboard animation."""
        
        logger.info(f"Preparing visualization for ego vehicle {interaction['ego_id']}...")
        
        # Reset SVO calculator
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
        
        # Panel layout
        self.axes['main'] = self.fig.add_axes([0.03, 0.35, 0.50, 0.60])
        self.axes['svo_chrono'] = self.fig.add_axes([0.56, 0.55, 0.42, 0.40])
        self.axes['aggr_balance'] = self.fig.add_axes([0.56, 0.08, 0.20, 0.40])
        self.axes['per_agent'] = self.fig.add_axes([0.78, 0.08, 0.20, 0.40])
        self.axes['info'] = self.fig.add_axes([0.03, 0.03, 0.50, 0.28])
        
        # Setup panels
        self._setup_main_panel(bounds, interaction, show_field)
        self._setup_svo_chronograph(frames_data)
        self._setup_aggressiveness_balance()
        self._setup_per_agent_panel()
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
            logger.info(f"Saving to {save_path}...")
            writer = animation.PillowWriter(fps=min(self.config.FPS, 12))
            ani.save(save_path, writer=writer, dpi=100)
            logger.info("Saved!")
        
        return ani
    
    def _prepare_frames(self, interaction: Dict) -> List[Dict]:
        """Prepare data for each frame."""
        
        frames_data = []
        ego_id = interaction['ego_id']
        
        # Accumulators for time-series
        all_times = []
        all_svo = []
        all_exerted = []
        all_suffered = []
        
        for i, frame in enumerate(interaction['frames']):
            ego_state = self.loader.get_vehicle_state(ego_id, frame)
            if ego_state is None:
                continue
            
            surrounding = self.loader.get_surrounding_vehicles(ego_id, frame)
            
            # Compute multi-agent SVO
            svo_state = self.svo_calc.compute_multi_agent_svo(ego_state, surrounding)
            
            time_s = i / self.config.FPS
            
            all_times.append(time_s)
            all_svo.append(svo_state.svo_angle)
            all_exerted.append(svo_state.total_exerted)
            all_suffered.append(svo_state.total_suffered)
            
            # Get ego trajectory history
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
                'surrounding': surrounding,
                'svo_state': svo_state,
                'ego_history': ego_history,
                # Time-series data
                'all_times': all_times.copy(),
                'all_svo': all_svo.copy(),
                'all_exerted': all_exerted.copy(),
                'all_suffered': all_suffered.copy(),
            })
        
        return frames_data
    
    def _calculate_bounds(self, frames_data: List[Dict]) -> Tuple:
        all_x, all_y = [], []
        
        for fd in frames_data:
            all_x.append(fd['ego'].x)
            all_y.append(fd['ego'].y)
            
            for v in fd['surrounding'].values():
                if v is not None:
                    all_x.append(v.x)
                    all_y.append(v.y)
        
        padding = 50
        return (min(all_x) - padding, max(all_x) + padding,
                min(all_y) - padding, max(all_y) + padding)
    
    def _setup_main_panel(self, bounds: Tuple, interaction: Dict, show_field: bool):
        """Setup bird's eye view panel."""
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
            f'Multi-Agent Interaction: {interaction["ego_class"].title()} (ID: {interaction["ego_id"]})',
            fontsize=12, fontweight='bold', color='white'
        )
        
        for spine in ax.spines.values():
            spine.set_color('#4A4A6A')
    
    def _setup_svo_chronograph(self, frames_data: List[Dict]):
        """Setup SVO time-series panel."""
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
        
        ax.axhline(45, color='white', linestyle='--', alpha=0.5, linewidth=1)
        
        ax.set_xlabel('Time (s)', fontsize=10, color='white')
        ax.set_ylabel('SVO Angle (°)', fontsize=10, color='white')
        ax.set_title('SVO Chronograph (Smoothed)', fontsize=11, fontweight='bold', color='white')
        ax.tick_params(colors='white')
        ax.grid(True, alpha=0.2, color='white')
        
        for spine in ax.spines.values():
            spine.set_color('#4A4A6A')
    
    def _setup_aggressiveness_balance(self):
        """Setup aggressiveness balance panel."""
        ax = self.axes['aggr_balance']
        ax.set_facecolor('#1A1A2E')
        
        ax.set_xlabel('Time (s)', fontsize=9, color='white')
        ax.set_ylabel('Aggressiveness', fontsize=9, color='white')
        ax.set_title('Exerted vs Suffered', fontsize=10, fontweight='bold', color='white')
        ax.tick_params(colors='white', labelsize=8)
        ax.grid(True, alpha=0.2, color='white')
        
        for spine in ax.spines.values():
            spine.set_color('#4A4A6A')
    
    def _setup_per_agent_panel(self):
        """Setup per-agent aggressiveness panel."""
        ax = self.axes['per_agent']
        ax.set_facecolor('#1A1A2E')
        
        ax.set_title('Per-Agent Aggr.', fontsize=10, fontweight='bold', color='white')
        ax.tick_params(colors='white', labelsize=8)
        
        for spine in ax.spines.values():
            spine.set_color('#4A4A6A')
    
    def _setup_info_panel(self, interaction: Dict):
        """Setup info panel."""
        ax = self.axes['info']
        ax.set_facecolor('#1A1A2E')
        ax.axis('off')
        
        for spine in ax.spines.values():
            spine.set_color('#4A4A6A')
    
    def _init_elements(self, show_field: bool) -> Dict:
        """Initialize plot elements."""
        elements = {}
        
        ax = self.axes['main']
        
        # Field contour
        if show_field:
            elements['field_contour'] = None
        
        # Ego vehicle
        elements['ego'] = patches.FancyBboxPatch(
            (0, 0), 12, 2.5, boxstyle="round,pad=0.02",
            facecolor='#1A1A1A', edgecolor='white', linewidth=2,
            alpha=0.95, zorder=20
        )
        ax.add_patch(elements['ego'])
        
        # Surrounding vehicles (8 slots)
        elements['surrounding'] = {}
        for pos_label, color in self.config.POSITION_COLORS.items():
            patch = patches.FancyBboxPatch(
                (0, 0), 4.5, 1.8, boxstyle="round,pad=0.02",
                facecolor=color, edgecolor='white', linewidth=1.5,
                alpha=0.8, zorder=15, visible=False
            )
            ax.add_patch(patch)
            elements['surrounding'][pos_label] = patch
        
        # Ego trail
        elements['ego_trail'], = ax.plot([], [], color='#E74C3C', alpha=0.6, linewidth=3, zorder=10)
        
        # Ego label
        elements['ego_label'] = ax.text(0, 0, '', fontsize=9, fontweight='bold',
            ha='center', va='bottom', color='white',
            bbox=dict(boxstyle='round', facecolor='#1A1A1A', alpha=0.8), zorder=25)
        
        # Aggressiveness arrows (from others to ego)
        elements['aggr_arrows'] = {}
        
        # SVO chronograph line
        ax_svo = self.axes['svo_chrono']
        elements['svo_line'], = ax_svo.plot([], [], color='#8E44AD', linewidth=2.5)
        elements['svo_marker'], = ax_svo.plot([], [], 'o', color='white', markersize=10,
            markeredgecolor='#8E44AD', markeredgewidth=2)
        
        # Aggressiveness balance lines
        ax_bal = self.axes['aggr_balance']
        elements['exerted_line'], = ax_bal.plot([], [], color='#E74C3C', linewidth=2, label='Exerted')
        elements['suffered_line'], = ax_bal.plot([], [], color='#3498DB', linewidth=2, label='Suffered')
        ax_bal.legend(loc='upper right', fontsize=8)
        
        # Info text
        elements['info_text'] = self.axes['info'].text(0.02, 0.95, '', ha='left', va='top',
            fontsize=10, color='white', family='monospace',
            transform=self.axes['info'].transAxes)
        
        return elements
    
    def _update_frame(self, frame_idx: int, frames_data: List[Dict], show_field: bool):
        """Update all panels for a single frame."""
        
        fd = frames_data[frame_idx]
        ego = fd['ego']
        surrounding = fd['surrounding']
        svo_state = fd['svo_state']
        
        # =====================================================================
        # Panel 1: Main View
        # =====================================================================
        
        ax = self.axes['main']
        
        # Update field
        if show_field and hasattr(self, 'field_grid_x'):
            if self.elements.get('field_contour') is not None:
                for coll in self.elements['field_contour'].collections:
                    coll.remove()
            
            field_suffered, field_exerted = self.field_gen.compute_field(
                self.field_grid_x, self.field_grid_y, ego, surrounding
            )
            
            self.elements['field_contour'] = ax.contourf(
                self.field_grid_x, self.field_grid_y, field_suffered,
                levels=15, cmap='Reds', alpha=0.3, zorder=1
            )
        
        # Update ego vehicle
        self._update_vehicle_patch(self.elements['ego'], ego)
        
        # Update surrounding vehicles
        for pos_label, patch in self.elements['surrounding'].items():
            other = surrounding.get(pos_label)
            if other is not None:
                self._update_vehicle_patch(patch, other)
                patch.set_visible(True)
            else:
                patch.set_visible(False)
        
        # Update ego trail
        if len(fd['ego_history']) > 1:
            self.elements['ego_trail'].set_data(fd['ego_history'][:, 0], fd['ego_history'][:, 1])
        
        # Update ego label
        self.elements['ego_label'].set_position((ego.x, ego.y + ego.width/2 + 3))
        self.elements['ego_label'].set_text(f'{ego.vehicle_class.title()}\n{ego.speed*3.6:.0f} km/h')
        
        # =====================================================================
        # Panel 2: SVO Chronograph
        # =====================================================================
        
        self.elements['svo_line'].set_data(fd['all_times'], fd['all_svo'])
        self.elements['svo_marker'].set_data([fd['time']], [svo_state.svo_angle])
        
        # =====================================================================
        # Panel 3: Aggressiveness Balance
        # =====================================================================
        
        ax_bal = self.axes['aggr_balance']
        ax_bal.set_xlim(0, max(fd['all_times']) + 0.1)
        
        max_aggr = max(max(fd['all_exerted'] + [1]), max(fd['all_suffered'] + [1]))
        ax_bal.set_ylim(0, max_aggr * 1.1)
        
        self.elements['exerted_line'].set_data(fd['all_times'], fd['all_exerted'])
        self.elements['suffered_line'].set_data(fd['all_times'], fd['all_suffered'])
        
        # =====================================================================
        # Panel 4: Per-Agent Breakdown
        # =====================================================================
        
        ax_agent = self.axes['per_agent']
        ax_agent.clear()
        ax_agent.set_facecolor('#1A1A2E')
        
        # Bar chart of aggressiveness from each agent
        labels = []
        values_to_ego = []
        values_from_ego = []
        colors = []
        
        for result in svo_state.others_to_ego:
            labels.append(result.position_label[:6])
            values_to_ego.append(result.aggressiveness)
            colors.append(self.config.POSITION_COLORS.get(result.position_label, '#7F8C8D'))
        
        for result in svo_state.ego_to_others:
            # Find matching position
            for i, lbl in enumerate(labels):
                if result.position_label.startswith(lbl[:4]):
                    values_from_ego.append(result.aggressiveness)
                    break
            else:
                values_from_ego.append(0)
        
        if labels:
            x = np.arange(len(labels))
            width = 0.35
            
            ax_agent.bar(x - width/2, values_to_ego, width, label='To Ego', color='#3498DB', alpha=0.8)
            ax_agent.bar(x + width/2, values_from_ego[:len(labels)], width, label='From Ego', color='#E74C3C', alpha=0.8)
            
            ax_agent.set_xticks(x)
            ax_agent.set_xticklabels(labels, rotation=45, ha='right', fontsize=7, color='white')
            ax_agent.tick_params(colors='white', labelsize=7)
            ax_agent.legend(loc='upper right', fontsize=7)
        
        ax_agent.set_title('Per-Agent Aggr.', fontsize=10, fontweight='bold', color='white')
        
        # =====================================================================
        # Panel 5: Info
        # =====================================================================
        
        n_surrounding = sum(1 for v in surrounding.values() if v is not None)
        
        info_text = (
            f"━━━ Frame {fd['frame']} | Time: {fd['time']:.2f}s ━━━\n\n"
            f"Ego Vehicle: {ego.vehicle_class.title()} (ID: {ego.track_id})\n"
            f"  Speed: {ego.speed*3.6:.1f} km/h\n"
            f"  Mass: {ego.mass/1000:.1f} tons\n\n"
            f"Surrounding Vehicles: {n_surrounding}\n\n"
            f"━━━ Aggressiveness ━━━\n"
            f"  Total Exerted: {svo_state.total_exerted:.1f}\n"
            f"  Total Suffered: {svo_state.total_suffered:.1f}\n"
            f"  Asymmetry Ratio: {svo_state.asymmetry_ratio:.2f}\n\n"
            f"━━━ SVO Analysis ━━━\n"
            f"  SVO Angle: {svo_state.svo_angle:.1f}°\n"
            f"  Cooperativeness: {svo_state.cooperativeness:.0%}\n"
        )
        
        # Add zone interpretation
        if svo_state.svo_angle > 60:
            zone = "ALTRUISTIC (Yielding)"
            zone_color = '#27AE60'
        elif svo_state.svo_angle > 30:
            zone = "COOPERATIVE"
            zone_color = '#3498DB'
        elif svo_state.svo_angle > 0:
            zone = "INDIVIDUALISTIC"
            zone_color = '#F39C12'
        else:
            zone = "COMPETITIVE"
            zone_color = '#E74C3C'
        
        info_text += f"  Zone: {zone}\n"
        
        self.elements['info_text'].set_text(info_text)
        
        return list(self.elements.values())
    
    def _update_vehicle_patch(self, patch, vehicle: VehicleState):
        """Update vehicle patch position and rotation."""
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
        
        self.svo_calc.reset()
        frames_data = self._prepare_frames(interaction)
        
        if not frames_data:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.patch.set_facecolor('#0D1117')
        
        times = [fd['time'] for fd in frames_data]
        svo_angles = [fd['svo_state'].svo_angle for fd in frames_data]
        exerted = [fd['svo_state'].total_exerted for fd in frames_data]
        suffered = [fd['svo_state'].total_suffered for fd in frames_data]
        
        # Plot 1: SVO over time
        ax1 = axes[0, 0]
        ax1.set_facecolor('#1A1A2E')
        ax1.plot(times, svo_angles, color='#8E44AD', linewidth=2)
        ax1.axhline(45, color='white', linestyle='--', alpha=0.5)
        ax1.fill_between(times, 60, 100, alpha=0.2, color='#27AE60')
        ax1.fill_between(times, 30, 60, alpha=0.2, color='#3498DB')
        ax1.fill_between(times, 0, 30, alpha=0.2, color='#F39C12')
        ax1.fill_between(times, -50, 0, alpha=0.2, color='#E74C3C')
        ax1.set_xlabel('Time (s)', color='white')
        ax1.set_ylabel('SVO Angle (°)', color='white')
        ax1.set_title('SVO Evolution (Smoothed - No Abrupt Jumps)', fontsize=12, 
                     fontweight='bold', color='white')
        ax1.tick_params(colors='white')
        ax1.set_ylim(-50, 100)
        ax1.grid(True, alpha=0.2)
        
        # Plot 2: Aggressiveness balance
        ax2 = axes[0, 1]
        ax2.set_facecolor('#1A1A2E')
        ax2.plot(times, exerted, color='#E74C3C', linewidth=2, label='Exerted (ego→others)')
        ax2.plot(times, suffered, color='#3498DB', linewidth=2, label='Suffered (others→ego)')
        ax2.fill_between(times, exerted, alpha=0.3, color='#E74C3C')
        ax2.fill_between(times, suffered, alpha=0.3, color='#3498DB')
        ax2.set_xlabel('Time (s)', color='white')
        ax2.set_ylabel('Aggressiveness', color='white')
        ax2.set_title('Asymmetric Aggressiveness', fontsize=12, fontweight='bold', color='white')
        ax2.tick_params(colors='white')
        ax2.legend(loc='best')
        ax2.grid(True, alpha=0.2)
        
        # Plot 3: Asymmetry ratio
        ax3 = axes[1, 0]
        ax3.set_facecolor('#1A1A2E')
        ratios = [fd['svo_state'].asymmetry_ratio for fd in frames_data]
        ax3.plot(times, ratios, color='#F39C12', linewidth=2)
        ax3.axhline(1.0, color='white', linestyle='--', alpha=0.5, label='Balance')
        ax3.set_xlabel('Time (s)', color='white')
        ax3.set_ylabel('Exerted/Suffered Ratio', color='white')
        ax3.set_title('Aggressiveness Asymmetry Ratio', fontsize=12, fontweight='bold', color='white')
        ax3.tick_params(colors='white')
        ax3.legend(loc='best')
        ax3.grid(True, alpha=0.2)
        
        # Plot 4: Number of surrounding vehicles
        ax4 = axes[1, 1]
        ax4.set_facecolor('#1A1A2E')
        n_surrounding = [sum(1 for v in fd['surrounding'].values() if v is not None) 
                        for fd in frames_data]
        ax4.plot(times, n_surrounding, color='#2ECC71', linewidth=2)
        ax4.set_xlabel('Time (s)', color='white')
        ax4.set_ylabel('Count', color='white')
        ax4.set_title('Number of Surrounding Vehicles', fontsize=12, fontweight='bold', color='white')
        ax4.tick_params(colors='white')
        ax4.set_ylim(0, 8)
        ax4.grid(True, alpha=0.2)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
            logger.info(f"Saved analysis to {save_path}")
            plt.close(fig)
        else:
            plt.show()


# =============================================================================
# Main
# =============================================================================

def main(data_dir: str, recording_id: int, output_dir: str = './output'):
    """Main execution."""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 70)
    logger.info("exiD Asymmetric Aggressiveness & Multi-Agent SVO Visualization")
    logger.info("=" * 70)
    
    # Load data
    loader = ExiDLoader(data_dir)
    if not loader.load_recording(recording_id):
        return None
    
    # Find interactions
    detector = MultiAgentInteractionDetector(loader)
    interactions = detector.find_interactions()
    
    if not interactions:
        logger.warning("No suitable interactions found.")
        return None
    
    logger.info(f"\nFound {len(interactions)} interactions")
    for i, inter in enumerate(interactions[:5]):
        logger.info(f"  {i+1}. {inter['ego_class']} (ID: {inter['ego_id']}) - "
                   f"{len(inter['frames'])} frames")
    
    # Visualize best interaction
    best = interactions[0]
    logger.info(f"\nVisualizing: {best['ego_class']} (ID: {best['ego_id']})")
    
    visualizer = AsymmetricAggressivenessVisualizer(loader)
    
    # Animation
    anim_path = output_path / f'recording_{recording_id}_asymmetric_aggr.gif'
    ani = visualizer.animate_interaction(best, save_path=str(anim_path), show_field=True)
    
    # Static analysis
    analysis_path = output_path / f'recording_{recording_id}_svo_analysis.png'
    visualizer.create_static_analysis(best, save_path=str(analysis_path))
    
    logger.info(f"\nOutputs saved to: {output_path}")
    
    return interactions


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='exiD Asymmetric Aggressiveness Visualization')
    parser.add_argument('--data_dir', type=str, default='C:\\exiD-tools\\data')
    parser.add_argument('--recording', type=int, default=25)
    parser.add_argument('--output_dir', type=str, default='./output')
    
    args = parser.parse_args()
    main(args.data_dir, args.recording, args.output_dir)