"""
exiD Dataset: Static GVF and SVO Visualization
===============================================
Creates publication-quality static figures showing:
1. Gaussian Velocity Field (GVF) from truck's perspective
2. Gaussian Velocity Field (GVF) from cars' perspective  
3. Combined interaction field with SVO annotations
4. SVO radar/polar plots

Reference: Zhang et al. (2021) "Spatiotemporal learning of multivehicle 
interaction patterns in lane-change scenarios" IEEE T-ITS
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

# For GVF computation
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
    """Configuration for GVF and SVO visualization."""
    
    HEAVY_VEHICLE_CLASSES: Set[str] = field(default_factory=lambda: {'truck', 'bus', 'van', 'trailer'})
    CAR_CLASSES: Set[str] = field(default_factory=lambda: {'car'})
    
    # GVF observation range (meters, relative to ego)
    OBS_RANGE_AHEAD: float = 60.0
    OBS_RANGE_BEHIND: float = 30.0
    OBS_RANGE_LEFT: float = 15.0
    OBS_RANGE_RIGHT: float = 15.0
    
    # GVF grid resolution
    GVF_GRID_X: int = 50
    GVF_GRID_Y: int = 25
    
    # GVF RBF kernel parameters
    SIGMA_X: float = 15.0  # Longitudinal length scale
    SIGMA_Y: float = 2.5   # Lateral length scale
    RBF_AMPLITUDE: float = 1.0
    
    # SVO parameters
    MU_1: float = 0.25
    MU_2: float = 0.25
    SIGMA_SVO: float = 0.08
    DELTA: float = 0.001
    TAU_1: float = 0.25
    TAU_2: float = 0.12
    BETA: float = 0.04
    
    MASS_HV: float = 15000.0
    MASS_PC: float = 3000.0
    V_REF: float = 25.0
    DIST_REF: float = 25.0
    
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
# GVF Computation (based on reference implementation)
# =============================================================================

def rbf_kernel_2d(XA: np.ndarray, XB: np.ndarray, 
                  length_scale: Tuple[float, float] = (15, 1.5), 
                  amplitude: float = 1.0) -> np.ndarray:
    """
    2D RBF kernel with anisotropic length scales.
    
    Args:
        XA: Points array (N, 2)
        XB: Points array (M, 2)
        length_scale: (sigma_x, sigma_y) length scales
        amplitude: Kernel amplitude
    
    Returns:
        Kernel matrix (N, M)
    """
    sigma_x, sigma_y = length_scale
    
    # Compute separable RBF kernel
    K_x = amplitude * rbf_kernel(XA[:, [0]], XB[:, [0]], gamma=0.5 / sigma_x**2)
    K_y = amplitude * rbf_kernel(XA[:, [1]], XB[:, [1]], gamma=0.5 / sigma_y**2)
    
    return K_x * K_y


def gp_posterior(X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray,
                 length_scale: Tuple[float, float] = (15, 1.5),
                 noise: float = 1e-6) -> Tuple[np.ndarray, np.ndarray]:
    """
    Gaussian Process posterior mean and covariance.
    
    Args:
        X_train: Training positions (N, 2)
        X_test: Test positions (M, 2)
        y_train: Training values (N,)
        length_scale: RBF length scales
        noise: Observation noise for numerical stability
    
    Returns:
        (mean, covariance) at test points
    """
    K = rbf_kernel_2d(X_train, X_train, length_scale)
    K_s = rbf_kernel_2d(X_test, X_train, length_scale)
    K_ss = rbf_kernel_2d(X_test, X_test, length_scale)
    
    # Add noise for numerical stability
    K_inv = inv(K + noise * np.eye(len(K)))
    
    mu = K_s @ K_inv @ y_train
    sigma = K_ss - K_s @ K_inv @ K_s.T
    
    return mu, sigma


def construct_gvf(positions: np.ndarray, velocities: np.ndarray,
                  x_range: Tuple[float, float], y_range: Tuple[float, float],
                  grid_size: Tuple[int, int] = (50, 25),
                  length_scale: Tuple[float, float] = (15, 1.5)) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Construct Gaussian Velocity Field.
    
    Args:
        positions: Vehicle positions relative to ego (N, 2)
        velocities: Vehicle velocities relative to ego (N, 2)
        x_range: (x_min, x_max) for grid
        y_range: (y_min, y_max) for grid
        grid_size: (nx, ny) grid points
        length_scale: RBF kernel length scales
    
    Returns:
        (X_mesh, Y_mesh, VX_field, VY_field)
    """
    nx, ny = grid_size
    
    X = np.linspace(x_range[0], x_range[1], nx)
    Y = np.linspace(y_range[0], y_range[1], ny)
    X_mesh, Y_mesh = np.meshgrid(X, Y)
    
    # Test points
    P_test = np.column_stack([X_mesh.ravel(), Y_mesh.ravel()])
    
    if len(positions) == 0:
        return X_mesh, Y_mesh, np.zeros_like(X_mesh), np.zeros_like(Y_mesh)
    
    # GP regression for velocity field
    VX_field, _ = gp_posterior(positions, P_test, velocities[:, 0], length_scale)
    VY_field, _ = gp_posterior(positions, P_test, velocities[:, 1], length_scale)
    
    return X_mesh, Y_mesh, VX_field.reshape(X_mesh.shape), VY_field.reshape(Y_mesh.shape)


# =============================================================================
# SVO Computation
# =============================================================================

def compute_svo(ego: Dict, other: Dict, config: Config) -> Dict:
    """Compute bidirectional SVO between ego and other vehicle."""
    
    dx = other['x'] - ego['x']
    dy = other['y'] - ego['y']
    dist = np.sqrt(dx**2 + dy**2)
    
    if dist < 1.0:
        return {'ego_svo': 45.0, 'other_svo': 45.0, 'distance': dist}
    
    # Aggressiveness ego -> other
    aggr_ego = _compute_aggressiveness(ego, other, dist, dx, dy, config)
    aggr_other = _compute_aggressiveness(other, ego, dist, -dx, -dy, config)
    
    # SVO computation
    ego_svo = _compute_single_svo(ego, other, aggr_ego, dist, config)
    other_svo = _compute_single_svo(other, ego, aggr_other, dist, config)
    
    return {
        'ego_svo': ego_svo,
        'other_svo': other_svo,
        'ego_aggr': aggr_ego,
        'other_aggr': aggr_other,
        'distance': dist,
        'dx': dx,
        'dy': dy,
    }


def _compute_aggressiveness(aggressor: Dict, sufferer: Dict, 
                           dist: float, dx: float, dy: float,
                           config: Config) -> float:
    """Compute directional aggressiveness."""
    
    ux, uy = dx / dist, dy / dist
    
    vi = aggressor['speed']
    vj = sufferer['speed']
    
    if vi > 0.1:
        cos_i = (aggressor['vx'] * ux + aggressor['vy'] * uy) / vi
    else:
        cos_i = 0.0
    
    if vj > 0.1:
        cos_j = -(sufferer['vx'] * ux + sufferer['vy'] * uy) / vj
    else:
        cos_j = 0.0
    
    cos_i = np.clip(cos_i, -1, 1)
    cos_j = np.clip(cos_j, -1, 1)
    
    phi = aggressor['heading']
    x_loc = dx * np.cos(phi) + dy * np.sin(phi)
    y_loc = -dx * np.sin(phi) + dy * np.cos(phi)
    
    exp_factor = np.exp(2 * config.BETA * vi)
    tau1 = config.TAU_1 * exp_factor
    tau2 = config.TAU_2
    
    r_pseudo = np.sqrt((x_loc**2)/(tau1**2) + (y_loc**2)/(tau2**2)) * min(tau1, tau2)
    r_pseudo = max(r_pseudo, 0.1)
    
    xi1 = config.MU_1 * vi * cos_i + config.MU_2 * vj * cos_j
    xi2 = -config.SIGMA_SVO * (aggressor['mass'] ** -1) * r_pseudo
    
    mass_term = (aggressor['mass'] * vi) / (2 * config.DELTA * sufferer['mass'])
    omega = mass_term * np.exp(xi1 + xi2)
    
    return np.clip(omega, 0, 2000)


def _compute_single_svo(veh: Dict, other: Dict, aggr: float, 
                        dist: float, config: Config) -> float:
    """Compute single SVO value."""
    
    v_ego = max(veh['speed'], 1.0)
    v_other = max(other['speed'], 1.0)
    speed_factor = (v_ego + v_other) / (2 * config.V_REF)
    dist_factor = config.DIST_REF / max(dist, 5.0)
    mass_factor = veh['mass'] / config.MASS_PC
    context = max(100.0 * speed_factor * dist_factor * mass_factor, 10.0)
    
    norm_aggr = np.clip(aggr / context, 0, 1)
    
    # Deceleration
    accel = (veh['vx'] * veh['ax'] + veh['vy'] * veh['ay']) / max(veh['speed'], 0.1)
    decel = max(0, -accel)
    norm_decel = np.tanh(decel / 2.0)
    
    # Yielding
    speed_ratio = veh['speed'] / max(config.V_REF, 1)
    norm_yield = np.clip(1 - speed_ratio, 0, 1)
    
    svo = (config.WEIGHT_AGGR * (90 - 135 * norm_aggr) +
           config.WEIGHT_DECEL * (45 * norm_decel) +
           config.WEIGHT_YIELD * (-22.5 + 67.5 * norm_yield))
    
    return np.clip(svo, -45, 90)


# =============================================================================
# Data Loader
# =============================================================================

class ExiDLoader:
    """Load exiD data for static visualization."""
    
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
            
            # Optional recording metadata (for background scaling)
            rec_meta_path = self.data_dir / f"{prefix}recordingMeta.csv"
            if rec_meta_path.exists():
                rec_meta_df = pd.read_csv(rec_meta_path)
                if not rec_meta_df.empty:
                    self.recording_meta = rec_meta_df.iloc[0]
                    self.ortho_px_to_meter = float(self.recording_meta.get('orthoPxToMeter', self.ortho_px_to_meter))
            
            # Optional lane layout background
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
        
        # Heading-based check (robust to wrap-around)
        d_heading = np.abs(np.arctan2(
            np.sin(other['heading'] - ego['heading']),
            np.cos(other['heading'] - ego['heading'])
        ))
        if d_heading > np.radians(heading_tol_deg):
            return False
        
        # Velocity alignment when both move
        ego_v = np.array([ego['vx'], ego['vy']])
        other_v = np.array([other['vx'], other['vy']])
        if np.linalg.norm(ego_v) > 0.5 and np.linalg.norm(other_v) > 0.5:
            cos_sim = np.dot(ego_v, other_v) / (np.linalg.norm(ego_v) * np.linalg.norm(other_v))
            return cos_sim > np.cos(np.radians(heading_tol_deg))
        
        return True
    
    def find_best_interaction_frame(self, ego_id: int) -> Optional[int]:
        """Find frame with most surrounding vehicles for ego (with relaxed fallbacks)."""
        
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
        
        # Try stricter to looser direction tolerances
        for tol in (60.0, 120.0, 179.0):
            for frame in frames[::10]:  # Sample every 10 frames
                snapshot = self.get_snapshot(ego_id, frame, heading_tol_deg=tol)
                if snapshot is None:
                    continue
                count = len(snapshot['surrounding'])
                if count > best_count:
                    best_count = count
                    best_frame = frame
            if best_count > 0:
                break  # Found at least one with neighbors
        
        # Fallback to ego mid-frame if nothing found
        if best_frame is None and len(frames) > 0:
            best_frame = int(np.median(frames))
        
        return best_frame
    
    def _contiguous_spans(self, frames: List[int]) -> List[List[int]]:
        """Group frames into contiguous spans."""
        if not frames:
            return []
        spans = []
        start = frames[0]
        prev = frames[0]
        for f in frames[1:]:
            if f == prev + 1:
                prev = f
                continue
            spans.append(list(range(start, prev + 1)))
            start = f
            prev = f
        spans.append(list(range(start, prev + 1)))
        return spans
    
    def get_merging_heavy_vehicles(self, min_frames: int = 5,
                                   margin_back: float = 50.0,
                                   margin_front: float = 100.0,
                                   lateral_factor: float = 1.5) -> List[int]:
        """Return heavy vehicles that exhibit merge-area interactions (mirrors heavy_vehicle_interactions logic)."""
        
        heavy_ids = self.get_heavy_vehicles()
        candidates: List[int] = []
        car_df = self.tracks_df[self.tracks_df['class'].str.lower().isin(self.config.CAR_CLASSES)]
        cars_by_frame: Dict[int, pd.DataFrame] = {f: df for f, df in car_df.groupby('frame')}
        
        for hv_id in heavy_ids:
            hv_track = self.tracks_df[self.tracks_df['trackId'] == hv_id].sort_values('frame')
            merge_bounds = self._get_merge_bounds(hv_id)
            if hv_track.empty or merge_bounds is None:
                continue
            
            # Pull series with fallbacks
            s_series = self._get_series(hv_track, primary='lonLaneletPos', fallback='traveledDistance')
            d_series = self._get_series(hv_track, primary='latLaneCenterOffset', fallback='yCenter')
            lane_w_series = self._get_series(hv_track, primary='laneWidth', default_val=3.6)
            
            if s_series is None or d_series is None:
                continue
            
            interaction_frames: Dict[int, List[int]] = {}
            
            for row_idx, row in hv_track.iterrows():
                frame = int(row['frame'])
                s_val = s_series.loc[row_idx]
                d_val = d_series.loc[row_idx]
                lw_val = lane_w_series.loc[row_idx] if lane_w_series is not None else 3.6
                
                if pd.isna(s_val) or pd.isna(d_val):
                    continue
                
                roi_min = s_val - margin_back
                roi_max = s_val + margin_front
                lat_thresh = lw_val * lateral_factor if not pd.isna(lw_val) else 3.6 * lateral_factor
                
                cars_frame = cars_by_frame.get(frame)
                if cars_frame is None:
                    continue
                
                c_s = self._get_series(cars_frame, primary='lonLaneletPos', fallback='traveledDistance')
                c_d = self._get_series(cars_frame, primary='latLaneCenterOffset', fallback='yCenter')
                if c_s is None or c_d is None:
                    continue
                
                for car_idx, car_row in cars_frame.iterrows():
                    s_car = c_s.loc[car_idx]
                    d_car = c_d.loc[car_idx]
                    if pd.isna(s_car) or pd.isna(d_car):
                        continue
                    if roi_min <= s_car <= roi_max and abs(d_car - d_val) < lat_thresh:
                        car_id = int(car_row['trackId'])
                        frames_list = interaction_frames.setdefault(car_id, [])
                        frames_list.append(frame)
            
            has_valid_span = False
            for frames_list in interaction_frames.values():
                frames_sorted = sorted(set(frames_list))
                for span in self._contiguous_spans(frames_sorted):
                    if len(span) >= min_frames:
                        has_valid_span = True
                        break
                if has_valid_span:
                    break
            
            if has_valid_span:
                candidates.append(hv_id)
        
        return candidates
    
    def _get_series(self, df: pd.DataFrame, primary: str, fallback: str = None, default_val: float = None) -> Optional[pd.Series]:
        """Return a column with optional fallback/default handling."""
        if primary in df.columns:
            series = df[primary]
            if not series.isna().all():
                return series
        if fallback and fallback in df.columns:
            series_fb = df[fallback]
            if not series_fb.isna().all():
                return series_fb
        if default_val is not None:
            return pd.Series(default_val, index=df.index)
        return None

    def _get_merge_bounds(self, ego_id: int) -> Optional[Tuple[float, float]]:
        """Estimate merge start/end along s using 5th/95th percentiles (similar to heavy_vehicle_interactions)."""
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

class GVFSVOVisualizer:
    """Creates static GVF and SVO visualizations."""
    
    def __init__(self, config: Config = None, loader: ExiDLoader = None):
        self.config = config or Config()
        self.loader = loader
    
    def create_combined_figure(self, snapshot: Dict, output_path: str = None):
        """
        Create a combined figure with:
        - GVF from truck's perspective
        - GVF from cars' perspective (average)
        - Combined interaction field with SVO
        - SVO summary
        """
        
        ego = snapshot['ego']
        surrounding = snapshot['surrounding']
        
        if not surrounding:
            logger.warning("No surrounding vehicles")
            return
        
        # Setup figure
        fig = plt.figure(figsize=(20, 12))
        fig.patch.set_facecolor('#0D1117')
        
        # Create subplots
        gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.25)
        
        ax1 = fig.add_subplot(gs[0, 0])  # GVF from truck
        ax2 = fig.add_subplot(gs[0, 1])  # GVF from car (representative)
        ax3 = fig.add_subplot(gs[0, 2])  # Combined field
        ax4 = fig.add_subplot(gs[1, 0])  # Traffic snapshot
        ax5 = fig.add_subplot(gs[1, 1])  # SVO bar chart
        ax6 = fig.add_subplot(gs[1, 2])  # SVO summary
        
        # Compute SVOs for all pairs
        svo_results = []
        for other in surrounding:
            svo = compute_svo(ego, other, self.config)
            svo['other_id'] = other['id']
            svo['other_class'] = other['class']
            svo_results.append(svo)
        
        # 1. GVF from Truck's perspective
        self._plot_gvf_from_ego(ax1, ego, surrounding, "Truck's Velocity Field")
        
        # 2. GVF from a representative car's perspective
        if surrounding:
            # Use the car closest to truck
            closest_car = min(surrounding, key=lambda c: np.sqrt((c['x']-ego['x'])**2 + (c['y']-ego['y'])**2))
            self._plot_gvf_from_ego(ax2, closest_car, [ego] + [s for s in surrounding if s['id'] != closest_car['id']], 
                                   f"Car {closest_car['id']}'s Velocity Field")
        
        # 3. Combined interaction field
        self._plot_combined_field(ax3, ego, surrounding, svo_results)
        
        # 4. Traffic snapshot
        self._plot_traffic_snapshot(ax4, ego, surrounding, svo_results)
        
        # 5. SVO bar chart
        self._plot_svo_bars(ax5, ego, svo_results)
        
        # 6. SVO summary panel
        self._plot_svo_summary(ax6, ego, svo_results)
        
        # Title
        fig.suptitle(
            f"GVF & SVO Analysis: {ego['class'].title()} (ID: {ego['id']}) | Frame: {snapshot['frame']} | "
            f"Surrounding: {len(surrounding)} vehicles",
            fontsize=14, fontweight='bold', color='white', y=0.98
        )
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        if output_path:
            fig.savefig(output_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
            logger.info(f"Saved: {output_path}")
            plt.close(fig)
        else:
            plt.show()
    
    def _plot_gvf_from_ego(self, ax, ego: Dict, others: List[Dict], title: str):
        """Plot GVF from ego's perspective."""
        
        ax.set_facecolor('#1A1A2E')
        
        # Compute relative positions and velocities
        positions = []
        velocities = []
        
        for other in others:
            # Position relative to ego
            dx = other['x'] - ego['x']
            dy = other['y'] - ego['y']
            
            # Rotate to ego's reference frame
            cos_h = np.cos(-ego['heading'])
            sin_h = np.sin(-ego['heading'])
            dx_rel = dx * cos_h - dy * sin_h
            dy_rel = dx * sin_h + dy * cos_h
            
            # Velocity relative to ego
            dvx = other['vx'] - ego['vx']
            dvy = other['vy'] - ego['vy']
            dvx_rel = dvx * cos_h - dvy * sin_h
            dvy_rel = dvx * sin_h + dvy * cos_h
            
            positions.append([dx_rel, dy_rel])
            velocities.append([dvx_rel, dvy_rel])
        
        positions = np.array(positions) if positions else np.array([]).reshape(0, 2)
        velocities = np.array(velocities) if velocities else np.array([]).reshape(0, 2)
        
        if positions.size == 0:
            positions = np.array([[0.0, 0.0]])
            velocities = np.array([[0.0, 0.0]])
        else:
            positions = np.vstack([positions, [0.0, 0.0]])
            velocities = np.vstack([velocities, [0.0, 0.0]])
        
        # Dynamic bounds so IDs and field stay within view
        margin_x = 5.0
        margin_y = 2.0
        ahead = max(self.config.OBS_RANGE_AHEAD, positions[:, 0].max() + margin_x)
        behind = max(self.config.OBS_RANGE_BEHIND, -positions[:, 0].min() + margin_x)
        left = max(self.config.OBS_RANGE_LEFT, positions[:, 1].max() + margin_y)
        right = max(self.config.OBS_RANGE_RIGHT, -positions[:, 1].min() + margin_y)
        lat_span = max(left, right, (ahead + behind) / 3.0)  # avoid overly skinny plot
        
        x_range = (-behind, ahead)
        y_range = (-lat_span, lat_span)
        
        # Construct GVF
        X, Y, VX, VY = construct_gvf(
            positions, velocities, x_range, y_range,
            grid_size=(self.config.GVF_GRID_X, self.config.GVF_GRID_Y),
            length_scale=(self.config.SIGMA_X, self.config.SIGMA_Y)
        )
        
        # Magnitude
        V_mag = np.sqrt(VX**2 + VY**2)
        
        # Plot
        cmap = 'jet'
        pcm = ax.pcolormesh(X, Y, V_mag, cmap=cmap, shading='gouraud', alpha=0.8)
        
        # Quiver plot
        skip = 3
        ax.quiver(X[::skip, ::skip], Y[::skip, ::skip], 
                 VX[::skip, ::skip], VY[::skip, ::skip],
                 color='white', alpha=0.7, scale=200)
        
        # Draw ego vehicle at origin
        ego_rect = mpatches.FancyBboxPatch(
            (-ego['length']/2, -ego['width']/2), 
            ego['length'], ego['width'],
            boxstyle="round,pad=0.02",
            facecolor=self.config.COLORS.get(ego['class'], '#E74C3C'),
            edgecolor='white', linewidth=2
        )
        ax.add_patch(ego_rect)
        
        # Draw other vehicles
        for i, other in enumerate(others):
            dx_rel, dy_rel = positions[i] if i < len(positions) else (0, 0)
            
            rect = mpatches.FancyBboxPatch(
                (dx_rel - other['length']/2, dy_rel - other['width']/2),
                other['length'], other['width'],
                boxstyle="round,pad=0.02",
                facecolor=self.config.COLORS.get(other['class'], '#3498DB'),
                edgecolor='white', linewidth=1, alpha=0.8
            )
            ax.add_patch(rect)
            ax.text(dx_rel, dy_rel + other['width']/2 + 2, str(other['id']),
                   ha='center', va='bottom', fontsize=8, color='yellow')
        
        # Lane markings
        ax.axhline(3.5, color='white', linestyle='--', alpha=0.5)
        ax.axhline(-3.5, color='white', linestyle='--', alpha=0.5)
        ax.axhline(7, color='white', linestyle='-', alpha=0.5)
        ax.axhline(-7, color='white', linestyle='-', alpha=0.5)
        
        ax.set_xlim(x_range)
        ax.set_ylim(y_range)
        ax.set_xlabel('Longitudinal (m)', color='white')
        ax.set_ylabel('Lateral (m)', color='white')
        ax.set_title(title, fontsize=11, fontweight='bold', color='white')
        ax.tick_params(colors='white')
        ax.set_aspect('equal')
        
        # Colorbar
        cbar = plt.colorbar(pcm, ax=ax, shrink=0.8)
        cbar.set_label('Relative Velocity (m/s)', color='white')
        cbar.ax.yaxis.set_tick_params(color='white')
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
        
        for spine in ax.spines.values():
            spine.set_color('#4A4A6A')
    
    def _plot_combined_field(self, ax, ego: Dict, surrounding: List[Dict], svo_results: List[Dict]):
        """Plot combined interaction field with SVO annotations."""
        
        ax.set_facecolor('#1A1A2E')
        
        # Compute relative bounds so annotations stay visible
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
        
        margin_x = 8.0
        margin_y = 3.0
        ahead = max(self.config.OBS_RANGE_AHEAD, rel_positions[:, 0].max() + margin_x)
        behind = max(self.config.OBS_RANGE_BEHIND, -rel_positions[:, 0].min() + margin_x)
        lat_span = max(self.config.OBS_RANGE_LEFT, self.config.OBS_RANGE_RIGHT, rel_positions[:, 1].ptp()/2 + margin_y)
        lat_span = max(lat_span, (ahead + behind) / 4.0)
        
        x_range = (-behind, ahead)
        y_range = (-lat_span, lat_span)
        
        nx, ny = 60, 30
        X = np.linspace(x_range[0], x_range[1], nx)
        Y = np.linspace(y_range[0], y_range[1], ny)
        X_mesh, Y_mesh = np.meshgrid(X, Y)
        
        # Combined potential field
        field = np.zeros_like(X_mesh)
        
        for other, svo in zip(surrounding, svo_results):
            dx = other['x'] - ego['x']
            dy = other['y'] - ego['y']
            
            # Rotate to ego frame
            cos_h = np.cos(-ego['heading'])
            sin_h = np.sin(-ego['heading'])
            dx_rel = dx * cos_h - dy * sin_h
            dy_rel = dx * sin_h + dy * cos_h
            
            # Gaussian influence scaled by aggressiveness
            aggr = svo['other_aggr']
            sigma_x = 10 + other['speed'] * 0.5
            sigma_y = 3
            
            field += aggr * np.exp(-((X_mesh - dx_rel)**2 / (2*sigma_x**2) + 
                                    (Y_mesh - dy_rel)**2 / (2*sigma_y**2)))
        
        # Plot field
        cmap = LinearSegmentedColormap.from_list('aggr', ['#1A1A2E', '#F39C12', '#E74C3C'])
        pcm = ax.pcolormesh(X_mesh, Y_mesh, field, cmap=cmap, shading='gouraud', alpha=0.8)
        
        # Draw ego
        ego_rect = mpatches.FancyBboxPatch(
            (-ego['length']/2, -ego['width']/2),
            ego['length'], ego['width'],
            boxstyle="round,pad=0.02",
            facecolor=self.config.COLORS.get(ego['class'], '#E74C3C'),
            edgecolor='white', linewidth=2
        )
        ax.add_patch(ego_rect)
        ax.text(0, -ego['width']/2 - 3, f"SVO: {np.mean([s['ego_svo'] for s in svo_results]):.1f} deg",
               ha='center', color='white', fontsize=10, fontweight='bold')
        
        # Draw others with SVO labels
        for other, svo in zip(surrounding, svo_results):
            dx = other['x'] - ego['x']
            dy = other['y'] - ego['y']
            
            cos_h = np.cos(-ego['heading'])
            sin_h = np.sin(-ego['heading'])
            dx_rel = dx * cos_h - dy * sin_h
            dy_rel = dx * sin_h + dy * cos_h
            
            # Color based on SVO
            if svo['other_svo'] > 60:
                color = '#27AE60'  # Altruistic
            elif svo['other_svo'] > 30:
                color = '#3498DB'  # Cooperative
            elif svo['other_svo'] > 0:
                color = '#F39C12'  # Individualistic
            else:
                color = '#E74C3C'  # Competitive
            
            rect = mpatches.FancyBboxPatch(
                (dx_rel - other['length']/2, dy_rel - other['width']/2),
                other['length'], other['width'],
                boxstyle="round,pad=0.02",
                facecolor=color,
                edgecolor='white', linewidth=1.5, alpha=0.9
            )
            ax.add_patch(rect)
            
            # SVO annotation
            ax.text(dx_rel, dy_rel + other['width']/2 + 2, 
                   f"{other['id']}\n{svo['other_svo']:.0f} deg",
                   ha='center', va='bottom', fontsize=8, color='white',
                   bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))
        
        ax.set_xlim(x_range)
        ax.set_ylim(y_range)
        ax.set_xlabel('Longitudinal (m)', color='white')
        ax.set_ylabel('Lateral (m)', color='white')
        ax.set_title('Combined Interaction Field with SVO', fontsize=11, fontweight='bold', color='white')
        ax.tick_params(colors='white')
        ax.set_aspect('equal')
        
        cbar = plt.colorbar(pcm, ax=ax, shrink=0.8)
        cbar.set_label('Interaction Intensity', color='white')
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
        
        # Compute bounds (tighter around ego so icons/background appear larger)
        all_x = [ego['x']] + [s['x'] for s in surrounding]
        all_y = [ego['y']] + [s['y'] for s in surrounding]
        
        x_center = ego['x']
        y_center = ego['y']
        span_x = max(40, max(all_x) - min(all_x) + 10, self.config.OBS_RANGE_AHEAD + self.config.OBS_RANGE_BEHIND + 20)
        span_y = max(30, max(all_y) - min(all_y) + 8, (self.config.OBS_RANGE_LEFT + self.config.OBS_RANGE_RIGHT) + 12)
        half_x = span_x / 2
        half_y = span_y / 2
        
        if bg_extent:
            x_min = max(bg_extent[0], x_center - half_x)
            x_max = min(bg_extent[1], x_center + half_x)
            y_min = max(bg_extent[2], y_center - half_y)
            y_max = min(bg_extent[3], y_center + half_y)
        else:
            x_min = x_center - half_x
            x_max = x_center + half_x
            y_min = y_center - half_y
            y_max = y_center + half_y
        
        # Draw ego with velocity arrow
        self._draw_vehicle(ax, ego, is_ego=True)
        ax.arrow(ego['x'], ego['y'], ego['vx']*0.5, ego['vy']*0.5,
                head_width=1, head_length=0.5, fc='yellow', ec='yellow')
        
        # Draw surrounding with velocity arrows
        for other, svo in zip(surrounding, svo_results):
            self._draw_vehicle(ax, other, is_ego=False, svo=svo['other_svo'])
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
    
    def _draw_vehicle(self, ax, veh: Dict, is_ego: bool = False, svo: float = None):
        """Draw a vehicle."""
        
        if is_ego:
            color = self.config.COLORS.get(veh['class'], '#E74C3C')
            alpha = 1.0
            lw = 2
        else:
            # Color by SVO
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
            alpha = 0.8
            lw = 1
        
        # Create rotated rectangle
        corners = self._get_rotated_rect(
            veh['x'], veh['y'], veh['length'], veh['width'], veh['heading']
        )
        
        rect = plt.Polygon(corners, closed=True, facecolor=color, 
                          edgecolor='white', linewidth=lw, alpha=alpha)
        ax.add_patch(rect)
        
        # Label
        label = f"{'EGO' if is_ego else veh['id']}"
        if svo is not None and not is_ego:
            label += f"\n{svo:.0f} deg"
        ax.text(veh['x'], veh['y'] + veh['width']/2 + 1.5, label,
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
    
    def _plot_svo_bars(self, ax, ego: Dict, svo_results: List[Dict]):
        """Plot SVO bar chart."""
        
        ax.set_facecolor('#1A1A2E')
        
        if not svo_results:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', color='white')
            return
        
        # Sort by distance
        svo_results = sorted(svo_results, key=lambda x: x['distance'])
        
        n = len(svo_results)
        x = np.arange(n)
        width = 0.35
        
        ego_svos = [s['ego_svo'] for s in svo_results]
        other_svos = [s['other_svo'] for s in svo_results]
        labels = [f"Car {s['other_id']}" for s in svo_results]
        
        # Bar colors based on SVO value
        def svo_color(svo):
            if svo > 60: return '#27AE60'
            elif svo > 30: return '#3498DB'
            elif svo > 0: return '#F39C12'
            else: return '#E74C3C'
        
        bars1 = ax.bar(
            x - width/2, ego_svos, width,
            label=f'{ego["class"].title()} -> Cars',
            color=[svo_color(s) for s in ego_svos],
            edgecolor='white', alpha=0.8
        )
        bars2 = ax.bar(
            x + width/2, other_svos, width,
            label=f'Cars -> {ego["class"].title()}',
            color=[svo_color(s) for s in other_svos],
            edgecolor='white', alpha=0.8
        )
        
        ax.axhline(45, color='white', linestyle='--', alpha=0.5)
        ax.axhline(0, color='white', linestyle='-', alpha=0.3)
        
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right', color='white', fontsize=8)
        ax.set_ylabel('SVO Angle (deg)', color='white')
        ax.set_title('Bidirectional SVO by Vehicle', fontsize=11, fontweight='bold', color='white')
        ax.tick_params(colors='white')
        ax.legend(loc='upper right', fontsize=8)
        ax.set_ylim(-50, 100)
        ax.grid(True, alpha=0.2, axis='y')
        
        for spine in ax.spines.values():
            spine.set_color('#4A4A6A')
    
    def _plot_svo_summary(self, ax, ego: Dict, svo_results: List[Dict]):
        """Plot SVO summary panel."""
        
        ax.set_facecolor('#1A1A2E')
        ax.axis('off')
        
        if not svo_results:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', color='white')
            return
        
        ego_svos = [s['ego_svo'] for s in svo_results]
        other_svos = [s['other_svo'] for s in svo_results]
        distances = [s['distance'] for s in svo_results]
        
        mean_ego = np.mean(ego_svos)
        mean_other = np.mean(other_svos)
        asymmetry = mean_other - mean_ego
        
        def interpret_svo(svo):
            if svo > 60: return 'Altruistic'
            elif svo > 30: return 'Cooperative'
            elif svo > 0: return 'Individualistic'
            else: return 'Competitive'
        
        summary_lines = [
            "SVO ANALYSIS SUMMARY",
            f"Ego Vehicle: {ego['class'].title()} (ID: {ego['id']})",
            f"Speed: {ego['speed']*3.6:.1f} km/h",
            f"Mass: {ego['mass']/1000:.1f} tons",
            "",
            f"{ego['class'].title()} -> Cars",
            f"Mean SVO: {mean_ego:.1f} deg",
            f"Range: [{min(ego_svos):.1f} deg, {max(ego_svos):.1f} deg]",
            f"Behavior: {interpret_svo(mean_ego)}",
            "",
            f"Cars -> {ego['class'].title()}",
            f"Mean SVO: {mean_other:.1f} deg",
            f"Range: [{min(other_svos):.1f} deg, {max(other_svos):.1f} deg]",
            f"Behavior: {interpret_svo(mean_other)}",
            "",
            "Asymmetry (Cars - Ego)",
            f"Delta: {asymmetry:.1f} deg",
            f"Cars are {abs(asymmetry):.1f} deg more {'cooperative' if asymmetry > 0 else 'competitive'}",
            "",
            "Spatial",
            f"Avg distance: {np.mean(distances):.1f} m",
            f"N vehicles: {len(svo_results)}"
        ]
        summary = "\n".join(summary_lines)
        
        ax.text(0.05, 0.95, summary, transform=ax.transAxes,
               fontsize=10, color='white', family='monospace',
               verticalalignment='top')
        
        for spine in ax.spines.values():
            spine.set_color('#4A4A6A')


# =============================================================================
# Main
# =============================================================================

def main(data_dir: str, recording_id: int, ego_id: Optional[int] = None, 
         frame: Optional[int] = None, output_dir: str = './output_gvf'):
    """Main entry point."""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 60)
    logger.info("GVF & SVO Static Visualization")
    logger.info("=" * 60)
    
    # Load data
    loader = ExiDLoader(data_dir)
    if not loader.load_recording(recording_id):
        return
    
    # Find ego vehicle (prefer merge-capable heavy vehicles)
    if ego_id is None:
        merging_ids = loader.get_merging_heavy_vehicles()
        if merging_ids:
            ego_id = merging_ids[0]
            logger.info(f"Auto-selected ego (merge-capable): {ego_id}")
        else:
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
        # Try nearest ego frame as fallback
        ego_frames = loader.tracks_df[loader.tracks_df['trackId'] == ego_id]['frame'].values
        if len(ego_frames) > 0:
            nearest = int(ego_frames[np.abs(ego_frames - frame).argmin()])
            if nearest != frame:
                logger.warning(f"Requested frame {frame} not found; falling back to nearest ego frame {nearest}")
                snapshot = loader.get_snapshot(ego_id, nearest)
        if snapshot is None:
            logger.error("Could not get snapshot")
            return
    
    logger.info(f"Ego: {snapshot['ego']['class']} (ID: {ego_id})")
    logger.info(f"Surrounding: {len(snapshot['surrounding'])} vehicles")
    
    # Create visualization
    viz = GVFSVOVisualizer(loader=loader)
    
    output_file = output_path / f'gvf_svo_recording{recording_id}_ego{ego_id}_frame{frame}.png'
    viz.create_combined_figure(snapshot, str(output_file))
    
    logger.info(f"Output saved to: {output_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GVF & SVO Static Visualization')
    parser.add_argument('--data_dir', type=str, default='C:\\exiD-tools\\data')
    parser.add_argument('--recording', type=int, default=25)
    parser.add_argument('--ego_id', type=int, default=None)
    parser.add_argument('--frame', type=int, default=None)
    parser.add_argument('--output_dir', type=str, default='./output_gvf')
    
    args = parser.parse_args()
    main(args.data_dir, args.recording, args.ego_id, args.frame, args.output_dir)
