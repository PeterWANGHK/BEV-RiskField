"""
exiD Dataset: Mechanical Wave-Based Vehicle Aggressiveness Visualization
=========================================================================
Implements the vehicle aggressiveness model from:
Hu et al. (2023) "Formulating Vehicle Aggressiveness Towards Social 
Cognitive Autonomous Driving" IEEE Transactions on Intelligent Vehicles

Key Features:
1. Asymmetric aggressiveness based on mass, motion states, and position
2. Mechanical wave analogy for threat propagation
3. Elliptical pseudo-distance accounting for acceleration
4. Doppler-like effect for relative motion

Mathematical Model (Simplified Formulation - Equation 23):
    Ω_{i→j} = (m_i |v_i|) / (2δ m_j) * exp(ξ₁ + ξ₂)
    
    where:
    - ξ₁ = μ₁|v_i|cos(θ_i) + μ₂|v_j|cos(θ_j)  (motion asymmetry)
    - ξ₂ = -σ * m_i^(-1) * R_ij               (distance decay)
    - R_ij is the elliptical pseudo-distance (Equation 17)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Set
from collections import defaultdict
import warnings
import argparse
import logging

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


# =============================================================================
# Configuration - Parameters from Paper Table I
# =============================================================================

@dataclass
class Config:
    """Configuration for Mechanical Wave Aggressiveness model.
    
    Parameters are from Table I in Hu et al. (2023).
    """
    
    HEAVY_VEHICLE_CLASSES: Set[str] = field(default_factory=lambda: {'truck', 'bus', 'van', 'trailer'})
    CAR_CLASSES: Set[str] = field(default_factory=lambda: {'car'})
    
    # Observation range (meters, relative to ego)
    OBS_RANGE_AHEAD: float = 60.0
    OBS_RANGE_BEHIND: float = 30.0
    OBS_RANGE_LEFT: float = 15.0
    OBS_RANGE_RIGHT: float = 15.0
    
    # Grid resolution for field visualization
    FIELD_GRID_X: int = 80
    FIELD_GRID_Y: int = 40
    
    # ========================================================================
    # Mechanical Wave Model Parameters (Table I from paper)
    # ========================================================================
    
    # Frequency coefficients (Equation 3 & 11 - used in general formulation)
    GAMMA_1: float = 1200.0  # AGV frequency coefficient
    GAMMA_2: float = 100.0   # SFV frequency coefficient
    EPSILON_1: float = 0.65  # AGV frequency mass exponent
    EPSILON_2: float = 0.5   # SFV frequency mass exponent
    
    # Distance and decay parameters
    TAU: float = 0.2         # Critical safe distance threshold
    BETA: float = 0.05       # Shape coefficient for ellipse
    SIGMA: float = 600.0     # Decay coefficient (Equation 5)
    
    # Damping ratio coefficient (Equation 9)
    DELTA: float = 5e-4      # δ in simplified formula
    
    # Velocity coefficients for Doppler effect (Equation 19)
    MU_1: float = 0.15       # AGV velocity coefficient
    MU_2: float = 0.16       # SFV velocity coefficient
    
    # Acceleration influence coefficient (Equation 18)
    T_0: float = 2.0         # Acceleration time horizon
    
    # ========================================================================
    # Vehicle mass parameters
    # ========================================================================
    MASS_HV: float = 15000.0   # Heavy vehicle mass (kg) - 15 tons
    MASS_PC: float = 3000.0    # Passenger car mass (kg) - 3 tons
    
    # Reference values for normalization
    V_REF: float = 25.0        # Reference velocity (m/s)
    DIST_REF: float = 25.0     # Reference distance (m)
    
    # ========================================================================
    # SVO computation weights (retained from original)
    # ========================================================================
    WEIGHT_AGGR: float = 0.45
    WEIGHT_DECEL: float = 0.30
    WEIGHT_YIELD: float = 0.25
    
    # Vehicle dimensions (for visualization)
    TRUCK_LENGTH: float = 12.0
    TRUCK_WIDTH: float = 2.5
    CAR_LENGTH: float = 4.5
    CAR_WIDTH: float = 1.8
    
    # Visualization
    FPS: int = 25
    COLORS: Dict[str, str] = field(default_factory=lambda: {
        'truck': '#E74C3C',
        'car': '#3498DB',
        'bus': '#F39C12',
        'van': '#9B59B6',
    })


# =============================================================================
# Mechanical Wave Aggressiveness Model (Based on Paper Equations)
# =============================================================================

def compute_pseudo_distance(aggressor: Dict, sufferer: Dict, config: Config) -> float:
    """
    Compute elliptical pseudo-distance R_ij (Equation 17).
    
    The pseudo-distance uses an ellipse aligned with AGV's heading,
    with semi-major axis in driving direction affected by velocity and acceleration.
    
    R_ij = sqrt( ((x_j-x_i)cos(φ_i) - (y_j-y_i)sin(φ_i))² / ρ² 
               + ((x_j-x_i)sin(φ_i) + (y_j-y_i)cos(φ_i))² / η² )
    
    where ρ = τ * exp(β * (|v_i| + cos(θ_i) * a_i * t_0)) and η = τ
    """
    
    # Position difference
    dx = sufferer['x'] - aggressor['x']
    dy = sufferer['y'] - aggressor['y']
    
    # AGV heading angle φ_i
    phi_i = aggressor['heading']
    cos_phi = np.cos(phi_i)
    sin_phi = np.sin(phi_i)
    
    # Transform to AGV local coordinates
    x_local = dx * cos_phi + dy * sin_phi   # longitudinal
    y_local = -dx * sin_phi + dy * cos_phi  # lateral
    
    # Compute velocity direction angle θ_i for acceleration effect
    v_i = aggressor['speed']
    if v_i > 0.1:
        # cos(θ_i) based on velocity alignment with position vector
        dist = np.sqrt(dx**2 + dy**2)
        if dist > 0.1:
            cos_theta_i = (aggressor['vx'] * dx + aggressor['vy'] * dy) / (v_i * dist)
            cos_theta_i = np.clip(cos_theta_i, -1, 1)
        else:
            cos_theta_i = 1.0
    else:
        cos_theta_i = 0.0
    
    # Compute acceleration magnitude in heading direction
    a_i = aggressor['ax'] * cos_phi + aggressor['ay'] * sin_phi
    
    # Compute semi-major axis ρ (Equation 18)
    # ρ = τ * exp(β * (|v_i| + cos(θ_i) * a_i * t_0))
    accel_term = max(0, cos_theta_i * a_i * config.T_0)  # Only consider positive contribution
    rho = config.TAU * np.exp(config.BETA * (v_i + accel_term))
    
    # Semi-minor axis η = τ
    eta = config.TAU
    
    # Ensure minimum values
    rho = max(rho, 0.1)
    eta = max(eta, 0.1)
    
    # Compute pseudo-distance (Equation 17)
    R_ij = np.sqrt((x_local**2) / (rho**2) + (y_local**2) / (eta**2))
    
    # Scale by minimum axis to get meaningful distance
    R_ij = R_ij * min(rho, eta)
    
    return max(R_ij, 0.1)  # Avoid division by zero


def compute_doppler_factor(aggressor: Dict, sufferer: Dict, config: Config) -> Tuple[float, float]:
    """
    Compute the Doppler-like frequency modification factors (Equation 19-20).
    
    ξ₁ = μ₁|v_i|cos(θ_i) + μ₂|v_j|cos(θ_j)
    
    where:
    - θ_i is angle between AGV velocity and AGV→SFV position vector
    - θ_j is angle between SFV velocity and SFV→AGV position vector
    
    Returns:
        (cos_theta_i, cos_theta_j) the direction cosines
    """
    
    # Position vectors
    dx = sufferer['x'] - aggressor['x']
    dy = sufferer['y'] - aggressor['y']
    dist = np.sqrt(dx**2 + dy**2)
    
    if dist < 0.1:
        return 0.0, 0.0
    
    # Unit vector from AGV to SFV
    ux_ij = dx / dist
    uy_ij = dy / dist
    
    # AGV velocity
    v_i = aggressor['speed']
    if v_i > 0.1:
        # cos(θ_i) = v_i · d_ij / |v_i|
        cos_theta_i = (aggressor['vx'] * ux_ij + aggressor['vy'] * uy_ij) / v_i
    else:
        cos_theta_i = 0.0
    
    # SFV velocity
    v_j = sufferer['speed']
    if v_j > 0.1:
        # cos(θ_j) = v_j · d_ji / |v_j|  (note: d_ji = -d_ij)
        cos_theta_j = -(sufferer['vx'] * ux_ij + sufferer['vy'] * uy_ij) / v_j
    else:
        cos_theta_j = 0.0
    
    # Clip to valid range
    cos_theta_i = np.clip(cos_theta_i, -1, 1)
    cos_theta_j = np.clip(cos_theta_j, -1, 1)
    
    return cos_theta_i, cos_theta_j


def compute_aggressiveness(aggressor: Dict, sufferer: Dict, config: Config) -> float:
    """
    Compute directional aggressiveness Ω_{i→j} using simplified formulation (Equation 23).
    
    Ω_{i→j} = (m_i |v_i|) / (2δ m_j) * exp(ξ₁ + ξ₂)
    
    where:
    - m_i: mass of AGV (aggressor)
    - m_j: mass of SFV (sufferer)
    - |v_i|: speed of AGV
    - δ: damping coefficient
    - ξ₁ = μ₁|v_i|cos(θ_i) + μ₂|v_j|cos(θ_j)  (motion term)
    - ξ₂ = -σ * m_i^(-1) * R_ij                (distance decay term)
    
    Args:
        aggressor: AGV vehicle state dict
        sufferer: SFV vehicle state dict
        config: Model configuration
    
    Returns:
        Aggressiveness value Ω_{i→j}
    """
    
    m_i = aggressor['mass']
    m_j = sufferer['mass']
    v_i = aggressor['speed']
    v_j = sufferer['speed']
    
    # Compute pseudo-distance R_ij (Equation 17)
    R_ij = compute_pseudo_distance(aggressor, sufferer, config)
    
    # Compute Doppler factors (Equation 20)
    cos_theta_i, cos_theta_j = compute_doppler_factor(aggressor, sufferer, config)
    
    # Motion asymmetry term ξ₁ (part of Equation 23)
    xi_1 = config.MU_1 * v_i * cos_theta_i + config.MU_2 * v_j * cos_theta_j
    
    # Distance decay term ξ₂ (part of Equation 23)
    xi_2 = -config.SIGMA * (m_i ** -1) * R_ij
    
    # Momentum ratio term (Equation 23)
    if m_j > 0:
        mass_term = (m_i * v_i) / (2 * config.DELTA * m_j)
    else:
        mass_term = 0.0
    
    # Final aggressiveness (Equation 23)
    omega = mass_term * np.exp(xi_1 + xi_2)
    
    return np.clip(omega, 0, 5000)


def compute_total_aggressiveness(sufferer: Dict, aggressors: List[Dict], config: Config) -> float:
    """
    Compute total aggressiveness on SFV from multiple AGVs (Equation 24).
    
    Ω_j = Σ_{i=1}^{K} (m_i |v_i|) / (2δ m_j) * exp(ξ₁ + ξ₂)
    
    Args:
        sufferer: SFV vehicle state dict
        aggressors: List of AGV vehicle state dicts
        config: Model configuration
    
    Returns:
        Total aggressiveness Ω_j
    """
    
    total_aggr = 0.0
    for aggressor in aggressors:
        total_aggr += compute_aggressiveness(aggressor, sufferer, config)
    
    return total_aggr


def compute_aggressiveness_field(ego: Dict, surrounding: List[Dict], 
                                  x_range: Tuple[float, float], 
                                  y_range: Tuple[float, float],
                                  grid_size: Tuple[int, int],
                                  config: Config,
                                  ego_as_sfv: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute aggressiveness field over a spatial grid.
    
    If ego_as_sfv=True: Compute aggressiveness FROM surrounding vehicles TO positions
                        (where ego might be located)
    If ego_as_sfv=False: Compute aggressiveness FROM ego TO surrounding positions
                         (threat field generated by ego)
    
    Returns:
        (X_mesh, Y_mesh, field) - meshgrid and aggressiveness values
    """
    
    nx, ny = grid_size
    X = np.linspace(x_range[0], x_range[1], nx)
    Y = np.linspace(y_range[0], y_range[1], ny)
    X_mesh, Y_mesh = np.meshgrid(X, Y)
    
    field = np.zeros_like(X_mesh)
    
    # Transform grid to global coordinates
    cos_h = np.cos(ego['heading'])
    sin_h = np.sin(ego['heading'])
    
    for i in range(ny):
        for j in range(nx):
            # Local coordinates (relative to ego)
            x_local = X_mesh[i, j]
            y_local = Y_mesh[i, j]
            
            # Transform to global coordinates
            x_global = ego['x'] + x_local * cos_h - y_local * sin_h
            y_global = ego['y'] + x_local * sin_h + y_local * cos_h
            
            # Create a virtual SFV at this grid point
            virtual_sfv = {
                'x': x_global,
                'y': y_global,
                'heading': ego['heading'],  # Assume same heading as ego
                'vx': ego['vx'],
                'vy': ego['vy'],
                'speed': ego['speed'],
                'ax': 0.0,
                'ay': 0.0,
                'mass': ego['mass']
            }
            
            if ego_as_sfv:
                # Compute aggressiveness from surrounding vehicles to this point
                field[i, j] = compute_total_aggressiveness(virtual_sfv, surrounding, config)
            else:
                # Compute aggressiveness from ego to this point
                field[i, j] = compute_aggressiveness(ego, virtual_sfv, config)
    
    return X_mesh, Y_mesh, field


# =============================================================================
# SVO Computation (Modified to use Mechanical Wave Aggressiveness)
# =============================================================================

def compute_svo_from_aggressiveness(ego: Dict, other: Dict, config: Config) -> Dict:
    """
    Compute bidirectional SVO based on mechanical wave aggressiveness.
    
    Returns:
        Dictionary with ego_svo, other_svo, and aggressiveness values
    """
    
    dx = other['x'] - ego['x']
    dy = other['y'] - ego['y']
    dist = np.sqrt(dx**2 + dy**2)
    
    if dist < 1.0:
        return {'ego_svo': 45.0, 'other_svo': 45.0, 'distance': dist,
                'ego_aggr': 0.0, 'other_aggr': 0.0}
    
    # Bidirectional aggressiveness using mechanical wave model
    aggr_ego_to_other = compute_aggressiveness(ego, other, config)
    aggr_other_to_ego = compute_aggressiveness(other, ego, config)
    
    # Convert aggressiveness to SVO
    ego_svo = _aggr_to_svo(ego, other, aggr_ego_to_other, dist, config)
    other_svo = _aggr_to_svo(other, ego, aggr_other_to_ego, dist, config)
    
    return {
        'ego_svo': ego_svo,
        'other_svo': other_svo,
        'ego_aggr': aggr_ego_to_other,
        'other_aggr': aggr_other_to_ego,
        'distance': dist,
        'dx': dx,
        'dy': dy,
    }


def _aggr_to_svo(veh: Dict, other: Dict, aggr: float, dist: float, config: Config) -> float:
    """Convert aggressiveness to SVO angle."""
    
    # Normalize aggressiveness based on context
    v_ego = max(veh['speed'], 1.0)
    v_other = max(other['speed'], 1.0)
    speed_factor = (v_ego + v_other) / (2 * config.V_REF)
    dist_factor = config.DIST_REF / max(dist, 5.0)
    mass_factor = veh['mass'] / config.MASS_PC
    context = max(100.0 * speed_factor * dist_factor * mass_factor, 10.0)
    
    norm_aggr = np.clip(aggr / context, 0, 1)
    
    # Deceleration component
    accel = (veh['vx'] * veh['ax'] + veh['vy'] * veh['ay']) / max(veh['speed'], 0.1)
    decel = max(0, -accel)
    norm_decel = np.tanh(decel / 2.0)
    
    # Yielding component
    speed_ratio = veh['speed'] / max(config.V_REF, 1)
    norm_yield = np.clip(1 - speed_ratio, 0, 1)
    
    # Weighted SVO
    svo = (config.WEIGHT_AGGR * (90 - 135 * norm_aggr) +
           config.WEIGHT_DECEL * (45 * norm_decel) +
           config.WEIGHT_YIELD * (-22.5 + 67.5 * norm_yield))
    
    return np.clip(svo, -45, 90)


# =============================================================================
# Data Loader (Same as original with minor modifications)
# =============================================================================

class ExiDLoader:
    """Load exiD data for visualization."""
    
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.config = Config()
        self.tracks_df = None
        self.tracks_meta_df = None
        self.merge_bounds: Dict[int, Tuple[float, float]] = {}
        self.recording_meta = None
        self.background_image = None
        self.ortho_px_to_meter = 0.1
    
    def load_recording(self, recording_id: int) -> bool:
        prefix = f"{recording_id:02d}_"
        
        try:
            self.tracks_df = pd.read_csv(self.data_dir / f"{prefix}tracks.csv")
            self.tracks_meta_df = pd.read_csv(self.data_dir / f"{prefix}tracksMeta.csv")
            
            # Optional recording metadata
            rec_meta_path = self.data_dir / f"{prefix}recordingMeta.csv"
            if rec_meta_path.exists():
                rec_meta_df = pd.read_csv(rec_meta_path)
                if not rec_meta_df.empty:
                    self.recording_meta = rec_meta_df.iloc[0]
                    self.ortho_px_to_meter = float(self.recording_meta.get('orthoPxToMeter', self.ortho_px_to_meter))
            
            # Optional background
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
            
            logger.info(f"Loaded recording {recording_id}")
            return True
        except Exception as e:
            logger.error(f"Error: {e}")
            return False
    
    def get_snapshot(self, ego_id: int, frame: int, heading_tol_deg: float = 60.0) -> Optional[Dict]:
        """Get a snapshot of ego and surrounding vehicles at a frame."""
        
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
            'width': float(ego_row.get('width', 2.0)),
            'length': float(ego_row.get('length', 5.0)),
            'class': vclass,
            'mass': self.config.MASS_HV if vclass in self.config.HEAVY_VEHICLE_CLASSES else self.config.MASS_PC,
        }
        
        # Get surrounding vehicles
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
                'width': float(row.get('width', 2.0)),
                'length': float(row.get('length', 5.0)),
                'class': other_class,
                'mass': self.config.MASS_HV if other_class in self.config.HEAVY_VEHICLE_CLASSES else self.config.MASS_PC,
            }
            
            # Check if within observation range
            dx = other['x'] - ego['x']
            dy = other['y'] - ego['y']
            
            if (-self.config.OBS_RANGE_BEHIND <= dx <= self.config.OBS_RANGE_AHEAD and
                -self.config.OBS_RANGE_RIGHT <= dy <= self.config.OBS_RANGE_LEFT and
                self._is_same_direction(ego, other, heading_tol_deg)):
                surrounding.append(other)
        
        return {'ego': ego, 'surrounding': surrounding, 'frame': frame}
    
    def _is_same_direction(self, ego: Dict, other: Dict, heading_tol_deg: float = 60.0) -> bool:
        """Check if another vehicle travels roughly the same direction as ego."""
        
        d_heading = np.abs(np.arctan2(
            np.sin(other['heading'] - ego['heading']),
            np.cos(other['heading'] - ego['heading'])
        ))
        if d_heading > np.radians(heading_tol_deg):
            return False
        
        ego_v = np.array([ego['vx'], ego['vy']])
        other_v = np.array([other['vx'], other['vy']])
        if np.linalg.norm(ego_v) > 0.5 and np.linalg.norm(other_v) > 0.5:
            cos_sim = np.dot(ego_v, other_v) / (np.linalg.norm(ego_v) * np.linalg.norm(other_v))
            return cos_sim > np.cos(np.radians(heading_tol_deg))
        
        return True
    
    def find_best_interaction_frame(self, ego_id: int) -> Optional[int]:
        """Find frame with most surrounding vehicles for ego."""
        
        ego_data = self.tracks_df[self.tracks_df['trackId'] == ego_id]
        if ego_data.empty:
            return None
        
        frames = ego_data['frame'].values
        
        # Focus on merge portion if available
        merge_bounds = self._get_merge_bounds(ego_id)
        if merge_bounds is not None:
            s_series = None
            if 'lonLaneletPos' in ego_data.columns:
                s_series = ego_data['lonLaneletPos']
            if (s_series is None or s_series.isna().all()) and 'traveledDistance' in ego_data.columns:
                s_series = ego_data['traveledDistance']
            if s_series is not None and not s_series.isna().all():
                s_vals = np.array(s_series, dtype=float)
                mask_merge = (s_vals >= merge_bounds[0] - 5.0) & (s_vals <= merge_bounds[1] + 5.0)
                if mask_merge.any():
                    frames = ego_data['frame'].values[mask_merge]
        
        best_frame = None
        best_count = -1
        
        for tol in (60.0, 120.0, 179.0):
            for frame in frames[::10]:
                snapshot = self.get_snapshot(ego_id, frame, heading_tol_deg=tol)
                if snapshot is None:
                    continue
                count = len(snapshot['surrounding'])
                if count > best_count:
                    best_count = count
                    best_frame = frame
            if best_count > 0:
                break
        
        if best_frame is None and len(frames) > 0:
            best_frame = int(np.median(frames))
        
        return best_frame
    
    def _get_merge_bounds(self, ego_id: int) -> Optional[Tuple[float, float]]:
        """Estimate merge start/end along s."""
        if ego_id in self.merge_bounds:
            return self.merge_bounds[ego_id]
        
        ego_data = self.tracks_df[self.tracks_df['trackId'] == ego_id]
        if ego_data.empty:
            return None
        
        s_series = None
        if 'lonLaneletPos' in ego_data.columns:
            s_series = ego_data['lonLaneletPos']
        if (s_series is None or s_series.isna().all()) and 'traveledDistance' in ego_data.columns:
            s_series = ego_data['traveledDistance']
        if s_series is None or s_series.isna().all():
            return None
        
        s_clean = s_series.replace([np.inf, -np.inf], np.nan).dropna()
        if s_clean.empty:
            return None
        
        start = float(np.nanpercentile(s_clean, 5))
        end = float(np.nanpercentile(s_clean, 95))
        if end <= start:
            end = start + 1.0
        
        self.merge_bounds[ego_id] = (start, end)
        return self.merge_bounds[ego_id]
    
    def get_heavy_vehicles(self) -> List[int]:
        """Get list of heavy vehicle IDs."""
        mask = self.tracks_meta_df['class'].str.lower().isin(self.config.HEAVY_VEHICLE_CLASSES)
        return self.tracks_meta_df[mask]['trackId'].tolist()
    
    def get_background_extent(self) -> List[float]:
        """Extent for plotting background image in meters."""
        if self.background_image is None:
            return [0, 0, 0, 0]
        h, w = self.background_image.shape[:2]
        return [0, w * self.ortho_px_to_meter, -h * self.ortho_px_to_meter, 0]


# =============================================================================
# Visualization
# =============================================================================

class MechanicalWaveVisualizer:
    """Creates visualizations based on mechanical wave aggressiveness model."""
    
    def __init__(self, config: Config = None, loader: ExiDLoader = None):
        self.config = config or Config()
        self.loader = loader
    
    def create_combined_figure(self, snapshot: Dict, output_path: str = None):
        """
        Create a combined figure with:
        - Aggressiveness field from truck's perspective (received threats)
        - Aggressiveness field generated by truck (emitted threats)
        - Asymmetric aggressiveness comparison
        - Traffic snapshot with SVO annotations
        - SVO analysis summary
        """
        
        ego = snapshot['ego']
        surrounding = snapshot['surrounding']
        
        if not surrounding:
            logger.warning("No surrounding vehicles")
            return
        
        # Setup figure
        fig = plt.figure(figsize=(22, 14))
        fig.patch.set_facecolor('#0D1117')
        
        # Create subplots
        gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.25)
        
        ax1 = fig.add_subplot(gs[0, 0])  # Received aggressiveness field
        ax2 = fig.add_subplot(gs[0, 1])  # Emitted aggressiveness field
        ax3 = fig.add_subplot(gs[0, 2])  # Combined asymmetric field
        ax4 = fig.add_subplot(gs[1, 0])  # Traffic snapshot
        ax5 = fig.add_subplot(gs[1, 1])  # Aggressiveness comparison
        ax6 = fig.add_subplot(gs[1, 2])  # Summary panel
        
        # Compute SVOs for all pairs
        svo_results = []
        for other in surrounding:
            svo = compute_svo_from_aggressiveness(ego, other, self.config)
            svo['other_id'] = other['id']
            svo['other_class'] = other['class']
            svo_results.append(svo)
        
        # 1. Aggressiveness received by ego from surrounding vehicles
        self._plot_received_aggressiveness(ax1, ego, surrounding, 
                                           f"Threat Received by {ego['class'].title()}")
        
        # 2. Aggressiveness emitted by ego
        self._plot_emitted_aggressiveness(ax2, ego, surrounding,
                                          f"Threat Emitted by {ego['class'].title()}")
        
        # 3. Combined asymmetric field
        self._plot_asymmetric_field(ax3, ego, surrounding, svo_results)
        
        # 4. Traffic snapshot
        self._plot_traffic_snapshot(ax4, ego, surrounding, svo_results)
        
        # 5. Aggressiveness comparison bar chart
        self._plot_aggressiveness_comparison(ax5, ego, svo_results)
        
        # 6. Summary panel
        self._plot_summary(ax6, ego, svo_results)
        
        # Title
        fig.suptitle(
            f"Mechanical Wave Aggressiveness Analysis: {ego['class'].title()} (ID: {ego['id']}) | "
            f"Frame: {snapshot['frame']} | Surrounding: {len(surrounding)} vehicles\n"
            f"Based on Hu et al. (2023) IEEE T-IV",
            fontsize=14, fontweight='bold', color='white', y=0.98
        )
        
        plt.tight_layout(rect=[0, 0, 1, 0.94])
        
        if output_path:
            fig.savefig(output_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
            logger.info(f"Saved: {output_path}")
            plt.close(fig)
        else:
            plt.show()
    
    def _plot_received_aggressiveness(self, ax, ego: Dict, surrounding: List[Dict], title: str):
        """Plot aggressiveness field received by ego from surrounding vehicles."""
        
        ax.set_facecolor('#1A1A2E')
        
        # Dynamic bounds
        rel_positions = []
        cos_h = np.cos(-ego['heading'])
        sin_h = np.sin(-ego['heading'])
        for other in surrounding:
            dx = other['x'] - ego['x']
            dy = other['y'] - ego['y']
            dx_rel = dx * cos_h - dy * sin_h
            dy_rel = dx * sin_h + dy * cos_h
            rel_positions.append([dx_rel, dy_rel])
        rel_positions = np.array(rel_positions) if rel_positions else np.array([[0.0, 0.0]])
        
        margin = 10.0
        ahead = max(self.config.OBS_RANGE_AHEAD, rel_positions[:, 0].max() + margin)
        behind = max(self.config.OBS_RANGE_BEHIND, -rel_positions[:, 0].min() + margin)
        lat_span = max(self.config.OBS_RANGE_LEFT, np.abs(rel_positions[:, 1]).max() + margin/2)
        
        x_range = (-behind, ahead)
        y_range = (-lat_span, lat_span)
        
        # Compute field
        X, Y, field = compute_aggressiveness_field(
            ego, surrounding, x_range, y_range,
            (self.config.FIELD_GRID_X, self.config.FIELD_GRID_Y),
            self.config, ego_as_sfv=True
        )
        
        # Normalize for visualization
        field_norm = np.log1p(field)  # Log scale for better visualization
        
        # Plot
        cmap = LinearSegmentedColormap.from_list('threat', 
            ['#1A1A2E', '#2E4057', '#048A81', '#54C6EB', '#F8E16C', '#FFC145', '#FF6B6B'])
        pcm = ax.pcolormesh(X, Y, field_norm, cmap=cmap, shading='gouraud', alpha=0.9)
        
        # Contour lines
        levels = np.percentile(field_norm[field_norm > 0], [25, 50, 75, 90]) if np.any(field_norm > 0) else [0]
        if len(levels) > 0 and np.max(levels) > np.min(levels):
            ax.contour(X, Y, field_norm, levels=levels, colors='white', alpha=0.3, linewidths=0.5)
        
        # Draw ego vehicle at origin
        self._draw_vehicle_local(ax, ego, 0, 0, is_ego=True)
        
        # Draw surrounding vehicles
        for other in surrounding:
            dx = other['x'] - ego['x']
            dy = other['y'] - ego['y']
            dx_rel = dx * cos_h - dy * sin_h
            dy_rel = dx * sin_h + dy * cos_h
            self._draw_vehicle_local(ax, other, dx_rel, dy_rel, is_ego=False)
        
        # Lane markings
        for y_lane in [-7, -3.5, 3.5, 7]:
            ls = '-' if abs(y_lane) == 7 else '--'
            ax.axhline(y_lane, color='white', linestyle=ls, alpha=0.3)
        
        ax.set_xlim(x_range)
        ax.set_ylim(y_range)
        ax.set_xlabel('Longitudinal (m)', color='white')
        ax.set_ylabel('Lateral (m)', color='white')
        ax.set_title(title, fontsize=11, fontweight='bold', color='white')
        ax.tick_params(colors='white')
        ax.set_aspect('equal')
        
        cbar = plt.colorbar(pcm, ax=ax, shrink=0.8)
        cbar.set_label('log(1 + Aggressiveness)', color='white')
        cbar.ax.yaxis.set_tick_params(color='white')
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
        
        for spine in ax.spines.values():
            spine.set_color('#4A4A6A')
    
    def _plot_emitted_aggressiveness(self, ax, ego: Dict, surrounding: List[Dict], title: str):
        """Plot aggressiveness field emitted by ego."""
        
        ax.set_facecolor('#1A1A2E')
        
        # Dynamic bounds
        rel_positions = []
        cos_h = np.cos(-ego['heading'])
        sin_h = np.sin(-ego['heading'])
        for other in surrounding:
            dx = other['x'] - ego['x']
            dy = other['y'] - ego['y']
            dx_rel = dx * cos_h - dy * sin_h
            dy_rel = dx * sin_h + dy * cos_h
            rel_positions.append([dx_rel, dy_rel])
        rel_positions = np.array(rel_positions) if rel_positions else np.array([[0.0, 0.0]])
        
        margin = 10.0
        ahead = max(self.config.OBS_RANGE_AHEAD, rel_positions[:, 0].max() + margin)
        behind = max(self.config.OBS_RANGE_BEHIND, -rel_positions[:, 0].min() + margin)
        lat_span = max(self.config.OBS_RANGE_LEFT, np.abs(rel_positions[:, 1]).max() + margin/2)
        
        x_range = (-behind, ahead)
        y_range = (-lat_span, lat_span)
        
        # Compute field (ego as AGV)
        X, Y, field = compute_aggressiveness_field(
            ego, surrounding, x_range, y_range,
            (self.config.FIELD_GRID_X, self.config.FIELD_GRID_Y),
            self.config, ego_as_sfv=False
        )
        
        # Normalize
        field_norm = np.log1p(field)
        
        # Plot with different colormap to distinguish
        cmap = LinearSegmentedColormap.from_list('emit', 
            ['#1A1A2E', '#2D132C', '#801336', '#C72C41', '#EE4540', '#FF8C42', '#FFD93D'])
        pcm = ax.pcolormesh(X, Y, field_norm, cmap=cmap, shading='gouraud', alpha=0.9)
        
        # Contour lines
        levels = np.percentile(field_norm[field_norm > 0], [25, 50, 75, 90]) if np.any(field_norm > 0) else [0]
        if len(levels) > 0 and np.max(levels) > np.min(levels):
            ax.contour(X, Y, field_norm, levels=levels, colors='white', alpha=0.3, linewidths=0.5)
        
        # Draw vehicles
        self._draw_vehicle_local(ax, ego, 0, 0, is_ego=True)
        for other in surrounding:
            dx = other['x'] - ego['x']
            dy = other['y'] - ego['y']
            dx_rel = dx * cos_h - dy * sin_h
            dy_rel = dx * sin_h + dy * cos_h
            self._draw_vehicle_local(ax, other, dx_rel, dy_rel, is_ego=False)
        
        # Lane markings
        for y_lane in [-7, -3.5, 3.5, 7]:
            ls = '-' if abs(y_lane) == 7 else '--'
            ax.axhline(y_lane, color='white', linestyle=ls, alpha=0.3)
        
        ax.set_xlim(x_range)
        ax.set_ylim(y_range)
        ax.set_xlabel('Longitudinal (m)', color='white')
        ax.set_ylabel('Lateral (m)', color='white')
        ax.set_title(title, fontsize=11, fontweight='bold', color='white')
        ax.tick_params(colors='white')
        ax.set_aspect('equal')
        
        cbar = plt.colorbar(pcm, ax=ax, shrink=0.8)
        cbar.set_label('log(1 + Aggressiveness)', color='white')
        cbar.ax.yaxis.set_tick_params(color='white')
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
        
        for spine in ax.spines.values():
            spine.set_color('#4A4A6A')
    
    def _plot_asymmetric_field(self, ax, ego: Dict, surrounding: List[Dict], svo_results: List[Dict]):
        """Plot combined field showing asymmetric interactions with SVO annotations."""
        
        ax.set_facecolor('#1A1A2E')
        
        # Dynamic bounds
        rel_positions = []
        cos_h = np.cos(-ego['heading'])
        sin_h = np.sin(-ego['heading'])
        for other in surrounding:
            dx = other['x'] - ego['x']
            dy = other['y'] - ego['y']
            dx_rel = dx * cos_h - dy * sin_h
            dy_rel = dx * sin_h + dy * cos_h
            rel_positions.append([dx_rel, dy_rel])
        rel_positions = np.array(rel_positions) if rel_positions else np.array([[0.0, 0.0]])
        
        margin = 8.0
        ahead = max(self.config.OBS_RANGE_AHEAD, rel_positions[:, 0].max() + margin)
        behind = max(self.config.OBS_RANGE_BEHIND, -rel_positions[:, 0].min() + margin)
        lat_span = max(self.config.OBS_RANGE_LEFT, np.abs(rel_positions[:, 1]).max() + margin/2)
        
        x_range = (-behind, ahead)
        y_range = (-lat_span, lat_span)
        
        nx, ny = 60, 30
        X = np.linspace(x_range[0], x_range[1], nx)
        Y = np.linspace(y_range[0], y_range[1], ny)
        X_mesh, Y_mesh = np.meshgrid(X, Y)
        
        # Create asymmetry field: difference between emitted and received
        field = np.zeros_like(X_mesh)
        
        for other, svo in zip(surrounding, svo_results):
            dx = other['x'] - ego['x']
            dy = other['y'] - ego['y']
            dx_rel = dx * cos_h - dy * sin_h
            dy_rel = dx * sin_h + dy * cos_h
            
            # Use asymmetry (other's aggr to ego - ego's aggr to other)
            asymmetry = svo['other_aggr'] - svo['ego_aggr']
            
            # Gaussian influence
            sigma_x = 12 + other['speed'] * 0.3
            sigma_y = 4
            
            field += asymmetry * np.exp(-((X_mesh - dx_rel)**2 / (2*sigma_x**2) + 
                                          (Y_mesh - dy_rel)**2 / (2*sigma_y**2)))
        
        # Diverging colormap for asymmetry
        cmap = LinearSegmentedColormap.from_list('asymm', 
            ['#3498DB', '#2E4057', '#1A1A2E', '#5D4037', '#E74C3C'])
        
        # Normalize to symmetric range
        vmax = max(abs(field.min()), abs(field.max()), 1)
        pcm = ax.pcolormesh(X_mesh, Y_mesh, field, cmap=cmap, shading='gouraud', 
                           vmin=-vmax, vmax=vmax, alpha=0.9)
        
        # Draw ego
        self._draw_vehicle_local(ax, ego, 0, 0, is_ego=True)
        
        # Draw others with asymmetry annotations
        for other, svo in zip(surrounding, svo_results):
            dx = other['x'] - ego['x']
            dy = other['y'] - ego['y']
            dx_rel = dx * cos_h - dy * sin_h
            dy_rel = dx * sin_h + dy * cos_h
            
            asymmetry = svo['other_aggr'] - svo['ego_aggr']
            
            # Color by asymmetry direction
            if asymmetry > 0:
                color = '#E74C3C'  # Other is more aggressive
            else:
                color = '#3498DB'  # Ego is more aggressive
            
            rect = mpatches.FancyBboxPatch(
                (dx_rel - other['length']/2, dy_rel - other['width']/2),
                other['length'], other['width'],
                boxstyle="round,pad=0.02",
                facecolor=color,
                edgecolor='white', linewidth=1.5, alpha=0.9
            )
            ax.add_patch(rect)
            
            # Annotation
            ax.text(dx_rel, dy_rel + other['width']/2 + 2, 
                   f"ID:{other['id']}\nΔΩ:{asymmetry:.0f}",
                   ha='center', va='bottom', fontsize=7, color='white',
                   bbox=dict(boxstyle='round', facecolor='black', alpha=0.6))
        
        ax.set_xlim(x_range)
        ax.set_ylim(y_range)
        ax.set_xlabel('Longitudinal (m)', color='white')
        ax.set_ylabel('Lateral (m)', color='white')
        ax.set_title('Asymmetric Aggressiveness (ΔΩ = Ω_other→ego - Ω_ego→other)', 
                    fontsize=10, fontweight='bold', color='white')
        ax.tick_params(colors='white')
        ax.set_aspect('equal')
        
        cbar = plt.colorbar(pcm, ax=ax, shrink=0.8)
        cbar.set_label('Asymmetry (+ = Other more aggressive)', color='white')
        cbar.ax.yaxis.set_tick_params(color='white')
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
        
        for spine in ax.spines.values():
            spine.set_color('#4A4A6A')
    
    def _plot_traffic_snapshot(self, ax, ego: Dict, surrounding: List[Dict], svo_results: List[Dict]):
        """Plot traffic snapshot in global coordinates."""
        
        ax.set_facecolor('#1A1A2E')
        
        bg_extent = None
        if self.loader and self.loader.background_image is not None:
            bg_extent = self.loader.get_background_extent()
            ax.imshow(self.loader.background_image, extent=bg_extent, alpha=0.6, aspect='equal', zorder=0)
        
        # Compute bounds
        all_x = [ego['x']] + [s['x'] for s in surrounding]
        all_y = [ego['y']] + [s['y'] for s in surrounding]
        
        x_center = ego['x']
        y_center = ego['y']
        span_x = max(50, max(all_x) - min(all_x) + 20)
        span_y = max(30, max(all_y) - min(all_y) + 15)
        
        if bg_extent:
            x_min = max(bg_extent[0], x_center - span_x/2)
            x_max = min(bg_extent[1], x_center + span_x/2)
            y_min = max(bg_extent[2], y_center - span_y/2)
            y_max = min(bg_extent[3], y_center + span_y/2)
        else:
            x_min, x_max = x_center - span_x/2, x_center + span_x/2
            y_min, y_max = y_center - span_y/2, y_center + span_y/2
        
        # Draw ego
        self._draw_vehicle_global(ax, ego, is_ego=True)
        ax.arrow(ego['x'], ego['y'], ego['vx']*0.5, ego['vy']*0.5,
                head_width=1, head_length=0.5, fc='yellow', ec='yellow')
        
        # Draw surrounding
        for other, svo in zip(surrounding, svo_results):
            self._draw_vehicle_global(ax, other, is_ego=False, svo=svo['other_svo'])
            ax.arrow(other['x'], other['y'], other['vx']*0.5, other['vy']*0.5,
                    head_width=0.8, head_length=0.4, fc='cyan', ec='cyan', alpha=0.7)
        
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_xlabel('X (m)', color='white')
        ax.set_ylabel('Y (m)', color='white')
        ax.set_title('Traffic Snapshot (Global)', fontsize=11, fontweight='bold', color='white')
        ax.tick_params(colors='white')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.2)
        
        for spine in ax.spines.values():
            spine.set_color('#4A4A6A')
    
    def _plot_aggressiveness_comparison(self, ax, ego: Dict, svo_results: List[Dict]):
        """Plot bidirectional aggressiveness comparison."""
        
        ax.set_facecolor('#1A1A2E')
        
        if not svo_results:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', color='white')
            return
        
        # Sort by distance
        svo_results = sorted(svo_results, key=lambda x: x['distance'])
        
        n = len(svo_results)
        x = np.arange(n)
        width = 0.35
        
        ego_aggr = [s['ego_aggr'] for s in svo_results]
        other_aggr = [s['other_aggr'] for s in svo_results]
        labels = [f"Car {s['other_id']}" for s in svo_results]
        
        # Use log scale for better visualization
        ego_aggr_log = np.log1p(ego_aggr)
        other_aggr_log = np.log1p(other_aggr)
        
        bars1 = ax.bar(x - width/2, ego_aggr_log, width,
                      label=f'{ego["class"].title()} → Cars',
                      color='#E74C3C', edgecolor='white', alpha=0.8)
        bars2 = ax.bar(x + width/2, other_aggr_log, width,
                      label=f'Cars → {ego["class"].title()}',
                      color='#3498DB', edgecolor='white', alpha=0.8)
        
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right', color='white', fontsize=8)
        ax.set_ylabel('log(1 + Aggressiveness)', color='white')
        ax.set_title('Bidirectional Aggressiveness (Ω)', fontsize=11, fontweight='bold', color='white')
        ax.tick_params(colors='white')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.2, axis='y')
        
        # Add asymmetry indicators
        for i, (ea, oa) in enumerate(zip(ego_aggr, other_aggr)):
            asymm = oa - ea
            color = '#E74C3C' if asymm > 0 else '#3498DB'
            ax.annotate(f'Δ={asymm:.0f}', (i, max(ego_aggr_log[i], other_aggr_log[i]) + 0.3),
                       ha='center', fontsize=7, color=color)
        
        for spine in ax.spines.values():
            spine.set_color('#4A4A6A')
    
    def _plot_summary(self, ax, ego: Dict, svo_results: List[Dict]):
        """Plot analysis summary panel."""
        
        ax.set_facecolor('#1A1A2E')
        ax.axis('off')
        
        if not svo_results:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', color='white')
            return
        
        ego_aggr = [s['ego_aggr'] for s in svo_results]
        other_aggr = [s['other_aggr'] for s in svo_results]
        ego_svos = [s['ego_svo'] for s in svo_results]
        other_svos = [s['other_svo'] for s in svo_results]
        distances = [s['distance'] for s in svo_results]
        
        mean_ego_aggr = np.mean(ego_aggr)
        mean_other_aggr = np.mean(other_aggr)
        asymmetry = mean_other_aggr - mean_ego_aggr
        
        def interpret_asymmetry(asym, ego_class):
            if abs(asym) < 10:
                return "Symmetric interaction"
            elif asym > 0:
                return f"Cars perceive {ego_class} as more threatening"
            else:
                return f"{ego_class.title()} perceives cars as more threatening"
        
        summary_lines = [
            "═" * 40,
            "MECHANICAL WAVE AGGRESSIVENESS ANALYSIS",
            "═" * 40,
            "",
            f"Ego Vehicle: {ego['class'].title()} (ID: {ego['id']})",
            f"  Mass: {ego['mass']/1000:.1f} tons",
            f"  Speed: {ego['speed']*3.6:.1f} km/h ({ego['speed']:.1f} m/s)",
            f"  Acceleration: {np.sqrt(ego['ax']**2 + ego['ay']**2):.2f} m/s²",
            "",
            "─" * 40,
            f"AGGRESSIVENESS (Ω) - {ego['class'].title()} → Cars",
            "─" * 40,
            f"  Mean: {mean_ego_aggr:.1f}",
            f"  Range: [{min(ego_aggr):.1f}, {max(ego_aggr):.1f}]",
            f"  This represents threat emitted by {ego['class']}",
            "",
            "─" * 40,
            f"AGGRESSIVENESS (Ω) - Cars → {ego['class'].title()}",
            "─" * 40,
            f"  Mean: {mean_other_aggr:.1f}",
            f"  Range: [{min(other_aggr):.1f}, {max(other_aggr):.1f}]",
            f"  This represents threat received by {ego['class']}",
            "",
            "─" * 40,
            "ASYMMETRY ANALYSIS",
            "─" * 40,
            f"  ΔΩ (mean): {asymmetry:.1f}",
            f"  Interpretation: {interpret_asymmetry(asymmetry, ego['class'])}",
            "",
            "─" * 40,
            "SVO ANALYSIS",
            "─" * 40,
            f"  {ego['class'].title()} SVO: {np.mean(ego_svos):.1f}°",
            f"  Cars SVO: {np.mean(other_svos):.1f}°",
            "",
            "─" * 40,
            "SPATIAL STATISTICS",
            "─" * 40,
            f"  Avg distance: {np.mean(distances):.1f} m",
            f"  N vehicles: {len(svo_results)}",
            "",
            "═" * 40,
            "Model: Hu et al. (2023) IEEE T-IV",
            "Eq. 23: Ω = (m_i|v_i|)/(2δm_j) exp(ξ₁+ξ₂)",
            "═" * 40,
        ]
        
        ax.text(0.02, 0.98, "\n".join(summary_lines), transform=ax.transAxes,
               fontsize=8, color='white', family='monospace',
               verticalalignment='top')
        
        for spine in ax.spines.values():
            spine.set_color('#4A4A6A')
    
    def _draw_vehicle_local(self, ax, veh: Dict, x_rel: float, y_rel: float, is_ego: bool = False):
        """Draw vehicle in local (ego-relative) coordinates."""
        
        if is_ego:
            color = self.config.COLORS.get(veh['class'], '#E74C3C')
            lw = 2
        else:
            color = self.config.COLORS.get(veh['class'], '#3498DB')
            lw = 1
        
        rect = mpatches.FancyBboxPatch(
            (x_rel - veh['length']/2, y_rel - veh['width']/2),
            veh['length'], veh['width'],
            boxstyle="round,pad=0.02",
            facecolor=color,
            edgecolor='white', linewidth=lw, alpha=0.9
        )
        ax.add_patch(rect)
        
        label = 'EGO' if is_ego else str(veh['id'])
        ax.text(x_rel, y_rel + veh['width']/2 + 1.5, label,
               ha='center', va='bottom', fontsize=8, 
               color='white' if is_ego else 'yellow',
               fontweight='bold' if is_ego else 'normal')
    
    def _draw_vehicle_global(self, ax, veh: Dict, is_ego: bool = False, svo: float = None):
        """Draw vehicle in global coordinates."""
        
        if is_ego:
            color = self.config.COLORS.get(veh['class'], '#E74C3C')
            lw = 2
            alpha = 1.0
        else:
            if svo is not None:
                if svo > 60:
                    color = '#27AE60'
                elif svo > 30:
                    color = '#3498DB'
                elif svo > 0:
                    color = '#F39C12'
                else:
                    color = '#E74C3C'
            else:
                color = self.config.COLORS.get(veh['class'], '#3498DB')
            lw = 1
            alpha = 0.8
        
        # Rotated rectangle
        corners = self._get_rotated_rect(veh['x'], veh['y'], veh['length'], veh['width'], veh['heading'])
        rect = plt.Polygon(corners, closed=True, facecolor=color, 
                          edgecolor='white', linewidth=lw, alpha=alpha)
        ax.add_patch(rect)
        
        label = 'EGO' if is_ego else str(veh['id'])
        if svo is not None and not is_ego:
            label += f"\n{svo:.0f}°"
        ax.text(veh['x'], veh['y'] + veh['width']/2 + 2, label,
               ha='center', va='bottom', fontsize=8, 
               color='white' if is_ego else 'yellow',
               fontweight='bold' if is_ego else 'normal')
    
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


# =============================================================================
# Main
# =============================================================================

def main(data_dir: str, recording_id: int, ego_id: Optional[int] = None, 
         frame: Optional[int] = None, output_dir: str = './output_mw'):
    """Main entry point."""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 60)
    logger.info("Mechanical Wave Aggressiveness Visualization")
    logger.info("Based on Hu et al. (2023) IEEE T-IV")
    logger.info("=" * 60)
    
    # Load data
    loader = ExiDLoader(data_dir)
    if not loader.load_recording(recording_id):
        return
    
    # Find ego vehicle
    if ego_id is None:
        heavy_ids = loader.get_heavy_vehicles()
        if not heavy_ids:
            logger.error("No heavy vehicles found")
            return
        ego_id = heavy_ids[0]
        logger.info(f"Auto-selected ego: {ego_id}")
    
    # Find best frame
    if frame is None:
        frame = loader.find_best_interaction_frame(ego_id)
        if frame is None:
            logger.error("Could not find suitable frame")
            return
        logger.info(f"Auto-selected frame: {frame}")
    
    # Get snapshot
    snapshot = loader.get_snapshot(ego_id, frame)
    if snapshot is None:
        ego_frames = loader.tracks_df[loader.tracks_df['trackId'] == ego_id]['frame'].values
        if len(ego_frames) > 0:
            nearest = int(ego_frames[np.abs(ego_frames - frame).argmin()])
            if nearest != frame:
                logger.warning(f"Frame {frame} not found; using nearest: {nearest}")
                snapshot = loader.get_snapshot(ego_id, nearest)
        if snapshot is None:
            logger.error("Could not get snapshot")
            return
    
    logger.info(f"Ego: {snapshot['ego']['class']} (ID: {ego_id})")
    logger.info(f"Surrounding: {len(snapshot['surrounding'])} vehicles")
    
    # Create visualization
    viz = MechanicalWaveVisualizer(loader=loader)
    
    output_file = output_path / f'mw_aggr_recording{recording_id}_ego{ego_id}_frame{snapshot["frame"]}.png'
    viz.create_combined_figure(snapshot, str(output_file))
    
    logger.info(f"Output saved to: {output_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Mechanical Wave Aggressiveness Visualization (Hu et al. 2023)')
    parser.add_argument('--data_dir', type=str, default='./data',
                       help='Path to exiD dataset directory')
    parser.add_argument('--recording', type=int, default=25,
                       help='Recording ID to process')
    parser.add_argument('--ego_id', type=int, default=None,
                       help='Ego vehicle ID (auto-select if not specified)')
    parser.add_argument('--frame', type=int, default=None,
                       help='Frame number (auto-select if not specified)')
    parser.add_argument('--output_dir', type=str, default='./output_mw',
                       help='Output directory for figures')
    
    args = parser.parse_args()
    main(args.data_dir, args.recording, args.ego_id, args.frame, args.output_dir)