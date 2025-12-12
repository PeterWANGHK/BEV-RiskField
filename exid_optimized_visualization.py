"""
exiD Dataset: Optimized Interactive SVO Visualization
=====================================================
OPTIMIZATIONS:
1. Pre-compute all frame data at startup (like intersection_visualizer.py)
2. Extract interaction scenario only (frames with surrounding vehicles)
3. Efficient patch updates without recreation
4. Proper vehicle lifecycle tracking

Based on intersection_visualizer.py logic.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import Button, Slider
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
# Configuration
# =============================================================================

@dataclass
class Config:
    HEAVY_VEHICLE_CLASSES: Set[str] = field(default_factory=lambda: {'truck', 'bus', 'van', 'trailer'})
    CAR_CLASSES: Set[str] = field(default_factory=lambda: {'car'})
    
    # SVO parameters
    MU_1: float = 0.25
    MU_2: float = 0.25
    SIGMA: float = 0.08
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
    
    EMA_ALPHA: float = 0.15
    
    # Visualization
    FPS: int = 25
    SKIP_N_FRAMES: int = 10
    MIN_INTERACTION_FRAMES: int = 100
    MIN_SURROUNDING_VEHICLES: int = 2
    
    COLORS: Dict[str, str] = field(default_factory=lambda: {
        'car': '#3498DB', 'truck': '#E74C3C', 'bus': '#F39C12',
        'van': '#9B59B6', 'trailer': '#E67E22', 'motorcycle': '#1ABC9C',
        'default': '#7F8C8D'
    })


# =============================================================================
# Data Loader with Pre-computation
# =============================================================================

class ExiDLoader:
    """Loads and pre-processes exiD data."""
    
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.config = Config()
        self.tracks_df = None
        self.tracks_meta_df = None
        self.recording_meta = None
        self.background_image = None
        self.ortho_px_to_meter = 0.1
        
        # Pre-computed data structures
        self.vehicle_data: Dict[int, Dict] = {}  # track_id -> all frame data
        self.ids_for_frame: Dict[int, List[int]] = {}
        self.tracks_meta: Dict[int, Dict] = {}
        self.maximum_frames = 0
    
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
            
            # Merge class info
            self.tracks_df = self.tracks_df.merge(
                self.tracks_meta_df[['trackId', 'class', 'width', 'length']],
                on='trackId', how='left', suffixes=('', '_meta')
            )
            
            if 'width_meta' in self.tracks_df.columns:
                self.tracks_df['width'] = self.tracks_df['width'].fillna(self.tracks_df['width_meta'])
                self.tracks_df['length'] = self.tracks_df['length'].fillna(self.tracks_df['length_meta'])
            
            # Pre-compute all data
            logger.info("Pre-computing vehicle data...")
            self._precompute_all_data()
            
            logger.info(f"  Total frames: {self.maximum_frames}")
            logger.info(f"  Total vehicles: {len(self.vehicle_data)}")
            
            return True
        except Exception as e:
            logger.error(f"Error: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _precompute_all_data(self):
        """Pre-compute all vehicle data indexed by frame (like intersection_visualizer)."""
        
        self.maximum_frames = int(self.tracks_df['frame'].max()) + 1
        
        # Build tracks_meta
        for _, row in self.tracks_meta_df.iterrows():
            tid = row['trackId']
            track_data = self.tracks_df[self.tracks_df['trackId'] == tid]
            
            self.tracks_meta[tid] = {
                'trackId': tid,
                'class': str(row.get('class', 'car')).lower(),
                'width': float(row.get('width', 2.0)),
                'length': float(row.get('length', 5.0)),
                'initialFrame': int(track_data['frame'].min()),
                'finalFrame': int(track_data['frame'].max()),
            }
        
        # Build ids_for_frame
        for frame in range(self.maximum_frames):
            frame_data = self.tracks_df[self.tracks_df['frame'] == frame]
            self.ids_for_frame[frame] = frame_data['trackId'].tolist()
        
        # Pre-compute vehicle data for each track
        for tid in self.tracks_meta.keys():
            track_data = self.tracks_df[self.tracks_df['trackId'] == tid].sort_values('frame')
            
            frames = track_data['frame'].values
            n_frames = len(frames)
            
            # Pre-compute arrays
            x = track_data['xCenter'].values
            y = track_data['yCenter'].values
            heading = np.radians(track_data['heading'].values) if 'heading' in track_data.columns else np.zeros(n_frames)
            vx = track_data['xVelocity'].values if 'xVelocity' in track_data.columns else np.zeros(n_frames)
            vy = track_data['yVelocity'].values if 'yVelocity' in track_data.columns else np.zeros(n_frames)
            ax = track_data['xAcceleration'].values if 'xAcceleration' in track_data.columns else np.zeros(n_frames)
            ay = track_data['yAcceleration'].values if 'yAcceleration' in track_data.columns else np.zeros(n_frames)
            
            width = self.tracks_meta[tid]['width']
            length = self.tracks_meta[tid]['length']
            vclass = self.tracks_meta[tid]['class']
            mass = self.config.MASS_HV if vclass in self.config.HEAVY_VEHICLE_CLASSES else self.config.MASS_PC
            
            # Pre-compute bounding boxes and triangles for all frames
            bboxes = np.zeros((n_frames, 4, 2))
            triangles = np.zeros((n_frames, 3, 2))
            centers = np.column_stack([x, y])
            speeds = np.sqrt(vx**2 + vy**2)
            
            half_l = length / 2
            half_w = width / 2
            
            for i in range(n_frames):
                cos_h = np.cos(heading[i])
                sin_h = np.sin(heading[i])
                R = np.array([[cos_h, -sin_h], [sin_h, cos_h]])
                
                # Bounding box corners (local frame)
                corners_local = np.array([
                    [-half_l, -half_w],
                    [half_l, -half_w],
                    [half_l, half_w],
                    [-half_l, half_w]
                ])
                bboxes[i] = corners_local @ R.T + np.array([x[i], y[i]])
                
                # Direction triangle
                tri_local = np.array([
                    [half_l, 0],
                    [half_l - half_w * 0.6, half_w * 0.4],
                    [half_l - half_w * 0.6, -half_w * 0.4]
                ])
                triangles[i] = tri_local @ R.T + np.array([x[i], y[i]])
            
            # Get surrounding vehicle IDs
            surrounding_cols = ['leadId', 'rearId', 'leftLeadId', 'leftAlongsideId', 
                              'leftRearId', 'rightLeadId', 'rightAlongsideId', 'rightRearId']
            surrounding_ids = {}
            for col in surrounding_cols:
                if col in track_data.columns:
                    surrounding_ids[col] = track_data[col].values
            
            self.vehicle_data[tid] = {
                'frames': frames,
                'frame_to_idx': {f: i for i, f in enumerate(frames)},
                'x': x,
                'y': y,
                'heading': heading,
                'vx': vx,
                'vy': vy,
                'ax': ax,
                'ay': ay,
                'speed': speeds,
                'bbox': bboxes,
                'triangle': triangles,
                'center': centers,
                'width': width,
                'length': length,
                'class': vclass,
                'mass': mass,
                'surrounding_ids': surrounding_ids,
            }
    
    def get_vehicle_at_frame(self, track_id: int, frame: int) -> Optional[Dict]:
        """Get pre-computed vehicle data at a specific frame."""
        if track_id not in self.vehicle_data:
            return None
        
        vdata = self.vehicle_data[track_id]
        if frame not in vdata['frame_to_idx']:
            return None
        
        idx = vdata['frame_to_idx'][frame]
        
        return {
            'track_id': track_id,
            'frame': frame,
            'idx': idx,
            'x': vdata['x'][idx],
            'y': vdata['y'][idx],
            'heading': vdata['heading'][idx],
            'vx': vdata['vx'][idx],
            'vy': vdata['vy'][idx],
            'ax': vdata['ax'][idx],
            'ay': vdata['ay'][idx],
            'speed': vdata['speed'][idx],
            'bbox': vdata['bbox'][idx],
            'triangle': vdata['triangle'][idx],
            'width': vdata['width'],
            'length': vdata['length'],
            'class': vdata['class'],
            'mass': vdata['mass'],
        }
    
    def get_surrounding_ids_at_frame(self, track_id: int, frame: int) -> Dict[str, Optional[int]]:
        """Get surrounding vehicle IDs at a specific frame."""
        if track_id not in self.vehicle_data:
            return {}
        
        vdata = self.vehicle_data[track_id]
        if frame not in vdata['frame_to_idx']:
            return {}
        
        idx = vdata['frame_to_idx'][frame]
        result = {}
        
        slot_mapping = {
            'leadId': 'lead', 'rearId': 'rear',
            'leftLeadId': 'leftLead', 'leftAlongsideId': 'leftAlongside',
            'leftRearId': 'leftRear', 'rightLeadId': 'rightLead',
            'rightAlongsideId': 'rightAlongside', 'rightRearId': 'rightRear',
        }
        
        for col, label in slot_mapping.items():
            if col in vdata['surrounding_ids']:
                val = vdata['surrounding_ids'][col][idx]
                try:
                    # Handle various types: string, float, int, nan
                    if pd.notna(val):
                        # Convert to numeric first
                        numeric_val = pd.to_numeric(val, errors='coerce')
                        if pd.notna(numeric_val) and numeric_val > 0:
                            result[label] = int(numeric_val)
                        else:
                            result[label] = None
                    else:
                        result[label] = None
                except (ValueError, TypeError):
                    result[label] = None
            else:
                result[label] = None
        
        return result
    
    def get_heavy_vehicles(self) -> List[int]:
        """Get list of heavy vehicle track IDs."""
        return [tid for tid, meta in self.tracks_meta.items() 
                if meta['class'] in self.config.HEAVY_VEHICLE_CLASSES]
    
    def get_background_extent(self) -> List[float]:
        if self.background_image is None:
            return [0, 500, -400, 0]
        h, w = self.background_image.shape[:2]
        return [0, w * self.ortho_px_to_meter, -h * self.ortho_px_to_meter, 0]


# =============================================================================
# Interaction Extractor
# =============================================================================

class InteractionExtractor:
    """Extract interaction scenarios from the dataset."""
    
    def __init__(self, loader: ExiDLoader):
        self.loader = loader
        self.config = Config()
    
    def find_interaction_scenario(self, ego_id: int) -> Dict:
        """Find frames where ego has meaningful interactions."""
        
        if ego_id not in self.loader.vehicle_data:
            return None
        
        vdata = self.loader.vehicle_data[ego_id]
        frames = vdata['frames']
        
        # Find frames with at least MIN_SURROUNDING_VEHICLES
        interaction_frames = []
        
        for frame in frames:
            surr = self.loader.get_surrounding_ids_at_frame(ego_id, frame)
            n_surrounding = sum(1 for v in surr.values() if v is not None)
            
            if n_surrounding >= self.config.MIN_SURROUNDING_VEHICLES:
                interaction_frames.append(frame)
        
        if len(interaction_frames) < self.config.MIN_INTERACTION_FRAMES:
            logger.warning(f"Ego {ego_id}: Only {len(interaction_frames)} interaction frames (need {self.config.MIN_INTERACTION_FRAMES})")
            # Fall back to using all frames
            interaction_frames = list(frames)
        
        # Find continuous segments
        if len(interaction_frames) > 0:
            # Use the longest continuous segment
            segments = []
            current_segment = [interaction_frames[0]]
            
            for i in range(1, len(interaction_frames)):
                if interaction_frames[i] - interaction_frames[i-1] <= 5:  # Allow small gaps
                    current_segment.append(interaction_frames[i])
                else:
                    if len(current_segment) >= 50:
                        segments.append(current_segment)
                    current_segment = [interaction_frames[i]]
            
            if len(current_segment) >= 50:
                segments.append(current_segment)
            
            if segments:
                # Use longest segment
                best_segment = max(segments, key=len)
                interaction_frames = best_segment
        
        logger.info(f"Ego {ego_id}: Extracted {len(interaction_frames)} interaction frames "
                   f"({interaction_frames[0]} to {interaction_frames[-1]})")
        
        return {
            'ego_id': ego_id,
            'ego_class': self.loader.tracks_meta[ego_id]['class'],
            'frames': interaction_frames,
            'start_frame': interaction_frames[0],
            'end_frame': interaction_frames[-1],
        }
    
    def find_best_ego(self) -> Optional[int]:
        """Find the heavy vehicle with best interaction scenario."""
        
        heavy_ids = self.loader.get_heavy_vehicles()
        
        if not heavy_ids:
            logger.warning("No heavy vehicles found")
            return None
        
        best_id = None
        best_score = 0
        
        for hv_id in heavy_ids:
            vdata = self.loader.vehicle_data[hv_id]
            frames = vdata['frames']
            
            if len(frames) < 100:
                continue
            
            # Score based on frames with surrounding vehicles
            score = 0
            for frame in frames[::5]:  # Sample every 5 frames
                surr = self.loader.get_surrounding_ids_at_frame(hv_id, frame)
                n = sum(1 for v in surr.values() if v is not None)
                if n >= 2:
                    score += n
            
            if score > best_score:
                best_score = score
                best_id = hv_id
        
        return best_id


# =============================================================================
# SVO Calculator (Optimized)
# =============================================================================

class SVOCalculator:
    """Fast SVO computation using pre-computed data."""
    
    def __init__(self, config: Config):
        self.config = config
        self.ema_ego = None
        self.ema_others = None
    
    def compute(self, ego: Dict, surrounding: List[Dict]) -> Dict:
        """Compute SVO for ego and surrounding vehicles."""
        
        if not surrounding:
            ego_svo = self.ema_ego if self.ema_ego else 45.0
            others_svo = self.ema_others if self.ema_others else 45.0
            return {
                'ego_svo': ego_svo,
                'others_svo': others_svo,
                'exerted': 0.0,
                'suffered': 0.0,
                'n_surrounding': 0
            }
        
        ego_svos = []
        others_svos = []
        total_exerted = 0.0
        total_suffered = 0.0
        
        for other in surrounding:
            dx = other['x'] - ego['x']
            dy = other['y'] - ego['y']
            dist = np.sqrt(dx**2 + dy**2)
            
            if dist < 1.0:
                continue
            
            # Aggressiveness
            aggr_ego = self._compute_aggr(ego, other, dist, dx, dy)
            aggr_other = self._compute_aggr(other, ego, dist, -dx, -dy)
            
            total_exerted += aggr_ego
            total_suffered += aggr_other
            
            # SVOs
            ego_svo = self._compute_svo(ego, other, aggr_ego, dist)
            other_svo = self._compute_svo(other, ego, aggr_other, dist)
            
            ego_svos.append(ego_svo)
            others_svos.append(other_svo)
        
        if ego_svos:
            mean_ego = np.mean(ego_svos)
            mean_others = np.mean(others_svos)
        else:
            mean_ego = 45.0
            mean_others = 45.0
        
        # EMA smoothing
        alpha = self.config.EMA_ALPHA
        if self.ema_ego is None:
            self.ema_ego = mean_ego
            self.ema_others = mean_others
        else:
            self.ema_ego = alpha * mean_ego + (1 - alpha) * self.ema_ego
            self.ema_others = alpha * mean_others + (1 - alpha) * self.ema_others
        
        return {
            'ego_svo': self.ema_ego,
            'others_svo': self.ema_others,
            'exerted': total_exerted,
            'suffered': total_suffered,
            'n_surrounding': len(surrounding)
        }
    
    def _compute_svo(self, veh: Dict, other: Dict, aggr: float, dist: float) -> float:
        """Compute single SVO value."""
        
        # Context normalization
        v_ego = max(veh['speed'], 1.0)
        v_other = max(other['speed'], 1.0)
        speed_factor = (v_ego + v_other) / (2 * self.config.V_REF)
        dist_factor = self.config.DIST_REF / max(dist, 5.0)
        mass_factor = veh['mass'] / self.config.MASS_PC
        context = max(100.0 * speed_factor * dist_factor * mass_factor, 10.0)
        
        norm_aggr = np.clip(aggr / context, 0, 1)
        
        # Deceleration
        accel = (veh['vx'] * veh['ax'] + veh['vy'] * veh['ay']) / max(veh['speed'], 0.1)
        decel = max(0, -accel)
        norm_decel = np.tanh(decel / 2.0)
        
        # Yielding
        speed_ratio = veh['speed'] / max(self.config.V_REF, 1)
        norm_yield = np.clip(1 - speed_ratio, 0, 1)
        
        # Combine
        svo = (self.config.WEIGHT_AGGR * (90 - 135 * norm_aggr) +
               self.config.WEIGHT_DECEL * (45 * norm_decel) +
               self.config.WEIGHT_YIELD * (-22.5 + 67.5 * norm_yield))
        
        return np.clip(svo, -45, 90)
    
    def _compute_aggr(self, aggressor: Dict, sufferer: Dict, dist: float, dx: float, dy: float) -> float:
        """Compute aggressiveness."""
        
        if dist < 0.1:
            return 0.0
        
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
        
        exp_factor = np.exp(2 * self.config.BETA * vi)
        tau1 = self.config.TAU_1 * exp_factor
        tau2 = self.config.TAU_2
        
        r_pseudo = np.sqrt((x_loc**2)/(tau1**2) + (y_loc**2)/(tau2**2)) * min(tau1, tau2)
        r_pseudo = max(r_pseudo, 0.1)
        
        xi1 = self.config.MU_1 * vi * cos_i + self.config.MU_2 * vj * cos_j
        xi2 = -self.config.SIGMA * (aggressor['mass'] ** -1) * r_pseudo
        
        mass_term = (aggressor['mass'] * vi) / (2 * self.config.DELTA * sufferer['mass'])
        omega = mass_term * np.exp(xi1 + xi2)
        
        return np.clip(omega, 0, 2000)
    
    def reset(self):
        self.ema_ego = None
        self.ema_others = None


# =============================================================================
# Interactive Visualizer (Optimized)
# =============================================================================

class InteractiveSVOVisualizer:
    """Optimized interactive visualizer."""
    
    def __init__(self, loader: ExiDLoader, interaction: Dict):
        self.loader = loader
        self.config = Config()
        self.svo_calc = SVOCalculator(self.config)
        
        # Interaction data
        self.ego_id = interaction['ego_id']
        self.frames = interaction['frames']
        self.n_frames = len(self.frames)
        self.frame_to_idx = {f: i for i, f in enumerate(self.frames)}
        
        # Current state
        self.current_idx = 0
        self.current_frame = self.frames[0]
        
        # Pre-compute SVO for all frames
        logger.info("Pre-computing SVO for all frames...")
        self._precompute_svo()
        
        # Plot objects
        self.plot_objs = {'vehicles': {}, 'trails': {}}
        self.changed_button = False
        
        # Initialize UI
        self._init_ui()
    
    def _precompute_svo(self):
        """Pre-compute SVO for all interaction frames."""
        
        self.svo_data = []
        self.svo_calc.reset()
        
        for frame in self.frames:
            ego = self.loader.get_vehicle_at_frame(self.ego_id, frame)
            
            # Get surrounding vehicles that actually exist in this frame
            surr_ids = self.loader.get_surrounding_ids_at_frame(self.ego_id, frame)
            surrounding = []
            
            for label, vid in surr_ids.items():
                if vid is not None:
                    vdata = self.loader.get_vehicle_at_frame(vid, frame)
                    if vdata is not None:
                        surrounding.append(vdata)
            
            svo = self.svo_calc.compute(ego, surrounding)
            
            self.svo_data.append({
                'frame': frame,
                'ego': ego,
                'surrounding': surrounding,
                'svo': svo,
            })
        
        logger.info(f"Pre-computed {len(self.svo_data)} frames")
    
    def _init_ui(self):
        """Initialize the UI."""
        
        # Create figure
        self.fig = plt.figure(figsize=(16, 9))
        self.fig.patch.set_facecolor('#1A1A2E')
        
        # Main view
        self.ax_main = self.fig.add_axes([0.03, 0.20, 0.55, 0.75])
        self.ax_main.set_facecolor('#0D1117')
        
        # SVO plot
        self.ax_svo = self.fig.add_axes([0.62, 0.50, 0.35, 0.45])
        self.ax_svo.set_facecolor('#1A1A2E')
        
        # Info panel
        self.ax_info = self.fig.add_axes([0.62, 0.20, 0.35, 0.25])
        self.ax_info.set_facecolor('#1A1A2E')
        self.ax_info.axis('off')
        
        # Controls
        self.ax_slider = self.fig.add_axes([0.15, 0.08, 0.40, 0.03], facecolor='#3A3A5A')
        self.ax_btn_prev = self.fig.add_axes([0.03, 0.08, 0.05, 0.03])
        self.ax_btn_next = self.fig.add_axes([0.56, 0.08, 0.05, 0.03])
        self.ax_btn_play = self.fig.add_axes([0.62, 0.08, 0.05, 0.03])
        self.ax_btn_stop = self.fig.add_axes([0.68, 0.08, 0.05, 0.03])
        self.ax_btn_skip_prev = self.fig.add_axes([0.09, 0.08, 0.05, 0.03])
        self.ax_btn_skip_next = self.fig.add_axes([0.74, 0.08, 0.05, 0.03])
        
        # Widgets
        self.slider = Slider(self.ax_slider, 'Frame', 0, self.n_frames - 1, 
                            valinit=0, valstep=1, color='#E74C3C')
        
        self.btn_prev = Button(self.ax_btn_prev, '◀', color='#3A3A5A', hovercolor='#5A5A7A')
        self.btn_next = Button(self.ax_btn_next, '▶', color='#3A3A5A', hovercolor='#5A5A7A')
        self.btn_play = Button(self.ax_btn_play, '▶ Play', color='#27AE60', hovercolor='#2ECC71')
        self.btn_stop = Button(self.ax_btn_stop, '■ Stop', color='#E74C3C', hovercolor='#EC7063')
        self.btn_skip_prev = Button(self.ax_btn_skip_prev, f'◀◀', color='#3A3A5A', hovercolor='#5A5A7A')
        self.btn_skip_next = Button(self.ax_btn_skip_next, f'▶▶', color='#3A3A5A', hovercolor='#5A5A7A')
        
        # Connect callbacks
        self.slider.on_changed(self._on_slider)
        self.btn_prev.on_clicked(lambda e: self._step(-1))
        self.btn_next.on_clicked(lambda e: self._step(1))
        self.btn_skip_prev.on_clicked(lambda e: self._step(-self.config.SKIP_N_FRAMES))
        self.btn_skip_next.on_clicked(lambda e: self._step(self.config.SKIP_N_FRAMES))
        self.btn_play.on_clicked(self._on_play)
        self.btn_stop.on_clicked(self._on_stop)
        
        # Keyboard
        self.fig.canvas.mpl_connect('key_press_event', self._on_key)
        
        # Timer
        self.timer = self.fig.canvas.new_timer(interval=40)  # 25 FPS
        self.timer.add_callback(self._on_timer)
        
        # Setup plots
        self._setup_main()
        self._setup_svo()
        
        # Title
        self.title = self.fig.suptitle('', fontsize=12, fontweight='bold', color='white', y=0.98)
        
        # Info text
        self.info_text = self.ax_info.text(0.05, 0.95, '', transform=self.ax_info.transAxes,
                                           fontsize=10, color='white', family='monospace',
                                           verticalalignment='top')
        
        # Initial draw
        self._update_display()
    
    def _setup_main(self):
        """Setup main traffic view."""
        ax = self.ax_main
        
        # Background
        if self.loader.background_image is not None:
            extent = self.loader.get_background_extent()
            ax.imshow(self.loader.background_image, extent=extent, alpha=0.6, aspect='auto')
        
        # Compute bounds from ego trajectory
        ego_data = self.loader.vehicle_data[self.ego_id]
        x_min, x_max = ego_data['x'].min() - 60, ego_data['x'].max() + 60
        y_min, y_max = ego_data['y'].min() - 40, ego_data['y'].max() + 40
        
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect('equal')
        ax.set_xlabel('X (m)', color='white')
        ax.set_ylabel('Y (m)', color='white')
        ax.tick_params(colors='white')
        
        for spine in ax.spines.values():
            spine.set_color('#4A4A6A')
    
    def _setup_svo(self):
        """Setup SVO plot."""
        ax = self.ax_svo
        
        # Background zones
        ax.axhspan(60, 100, alpha=0.15, color='#27AE60')
        ax.axhspan(30, 60, alpha=0.15, color='#3498DB')
        ax.axhspan(0, 30, alpha=0.15, color='#F39C12')
        ax.axhspan(-50, 0, alpha=0.15, color='#E74C3C')
        ax.axhline(45, color='white', linestyle='--', alpha=0.3)
        
        # Plot full SVO trajectory
        times = np.arange(self.n_frames) / self.config.FPS
        ego_svo = [d['svo']['ego_svo'] for d in self.svo_data]
        others_svo = [d['svo']['others_svo'] for d in self.svo_data]
        
        ax.plot(times, ego_svo, color='#E74C3C', linewidth=1.5, alpha=0.5, label='Truck SVO')
        ax.plot(times, others_svo, color='#3498DB', linewidth=1.5, alpha=0.5, label='Cars SVO')
        
        # Current position markers
        self.svo_marker_ego, = ax.plot([], [], 'o', color='#E74C3C', markersize=10)
        self.svo_marker_others, = ax.plot([], [], 's', color='#3498DB', markersize=10)
        self.svo_vline = ax.axvline(0, color='white', linestyle='-', alpha=0.5)
        
        ax.set_xlim(0, times[-1])
        ax.set_ylim(-50, 100)
        ax.set_xlabel('Time (s)', color='white')
        ax.set_ylabel('SVO (°)', color='white')
        ax.set_title('Bidirectional SVO', fontsize=11, fontweight='bold', color='white')
        ax.tick_params(colors='white')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.2)
        
        for spine in ax.spines.values():
            spine.set_color('#4A4A6A')
    
    def _update_display(self):
        """Update display for current frame."""
        
        data = self.svo_data[self.current_idx]
        frame = data['frame']
        ego = data['ego']
        surrounding = data['surrounding']
        svo = data['svo']
        
        time_s = self.current_idx / self.config.FPS
        
        # Update title
        self.title.set_text(
            f'Frame {frame} | Time: {time_s:.1f}s | '
            f'Ego: {self.loader.tracks_meta[self.ego_id]["class"].title()} (ID: {self.ego_id}) | '
            f'Surrounding: {len(surrounding)}'
        )
        
        # Clear old vehicle patches
        for tid, objs in list(self.plot_objs['vehicles'].items()):
            objs['rect'].remove()
            objs['tri'].remove()
            objs['text'].remove()
        self.plot_objs['vehicles'].clear()
        
        # Draw ego
        self._draw_vehicle(ego, is_ego=True)
        
        # Draw surrounding
        for veh in surrounding:
            self._draw_vehicle(veh, is_ego=False)
        
        # Update SVO markers
        self.svo_marker_ego.set_data([time_s], [svo['ego_svo']])
        self.svo_marker_others.set_data([time_s], [svo['others_svo']])
        self.svo_vline.set_xdata([time_s, time_s])
        
        # Update info
        info = (
            f"═══ EGO ═══\n"
            f"Speed: {ego['speed']*3.6:.1f} km/h\n\n"
            f"═══ SVO ═══\n"
            f"Truck: {svo['ego_svo']:.1f}°\n"
            f"Cars:  {svo['others_svo']:.1f}°\n"
            f"Δ: {svo['others_svo']-svo['ego_svo']:.1f}°\n\n"
            f"═══ AGGR ═══\n"
            f"Exerted: {svo['exerted']:.1f}\n"
            f"Suffered: {svo['suffered']:.1f}"
        )
        self.info_text.set_text(info)
        
        self.fig.canvas.draw_idle()
    
    def _draw_vehicle(self, veh: Dict, is_ego: bool = False):
        """Draw a vehicle."""
        
        ax = self.ax_main
        tid = veh['track_id']
        
        if is_ego:
            color = '#E74C3C'
            alpha = 0.9
            zorder = 25
            lw = 2
        else:
            color = self.config.COLORS.get(veh['class'], self.config.COLORS['default'])
            alpha = 0.8
            zorder = 20
            lw = 1
        
        rect = plt.Polygon(veh['bbox'], closed=True, zorder=zorder,
                          facecolor=color, edgecolor='white', linewidth=lw, alpha=alpha)
        ax.add_patch(rect)
        
        tri = plt.Polygon(veh['triangle'], closed=True, zorder=zorder+1,
                         facecolor='black', alpha=alpha*0.8)
        ax.add_patch(tri)
        
        text = ax.text(veh['x'], veh['y'] + veh['width']/2 + 1.5,
                      str(tid), ha='center', va='bottom', fontsize=8,
                      color='white' if is_ego else 'yellow',
                      fontweight='bold' if is_ego else 'normal', zorder=zorder+2)
        
        self.plot_objs['vehicles'][tid] = {'rect': rect, 'tri': tri, 'text': text}
    
    def _step(self, delta: int):
        """Step by delta frames."""
        new_idx = np.clip(self.current_idx + delta, 0, self.n_frames - 1)
        if new_idx != self.current_idx:
            self.current_idx = new_idx
            self.current_frame = self.frames[new_idx]
            self.changed_button = True
            self._update_slider()
            self._update_display()
    
    def _update_slider(self):
        """Update slider without triggering callback."""
        self.slider.eventson = False
        self.slider.set_val(self.current_idx)
        self.slider.eventson = True
    
    def _on_slider(self, val):
        if not self.changed_button:
            self.current_idx = int(val)
            self.current_frame = self.frames[self.current_idx]
            self._update_display()
        self.changed_button = False
    
    def _on_play(self, event):
        self.timer.start()
    
    def _on_stop(self, event):
        self.timer.stop()
    
    def _on_timer(self):
        if self.current_idx < self.n_frames - 1:
            self._step(1)
        else:
            self._on_stop(None)
    
    def _on_key(self, event):
        if event.key == 'right':
            self._step(self.config.SKIP_N_FRAMES)
        elif event.key == 'left':
            self._step(-self.config.SKIP_N_FRAMES)
        elif event.key in ('up', ' '):
            self._on_play(None)
        elif event.key in ('down', 'escape'):
            self._on_stop(None)
    
    def show(self):
        plt.show()


# =============================================================================
# Main
# =============================================================================

def main(data_dir: str, recording_id: int, ego_id: Optional[int] = None):
    """Main entry point."""
    
    logger.info("=" * 60)
    logger.info("Optimized Interactive exiD SVO Visualization")
    logger.info("=" * 60)
    
    # Load data
    loader = ExiDLoader(data_dir)
    if not loader.load_recording(recording_id):
        return
    
    # Find interaction
    extractor = InteractionExtractor(loader)
    
    if ego_id is None:
        ego_id = extractor.find_best_ego()
        if ego_id is None:
            logger.error("No suitable ego vehicle found")
            return
    
    interaction = extractor.find_interaction_scenario(ego_id)
    if interaction is None:
        logger.error(f"Could not extract interaction for ego {ego_id}")
        return
    
    logger.info(f"\nEgo: {interaction['ego_class']} (ID: {ego_id})")
    logger.info(f"Frames: {interaction['start_frame']} to {interaction['end_frame']} ({len(interaction['frames'])} frames)")
    
    # Create visualizer
    viz = InteractiveSVOVisualizer(loader, interaction)
    
    logger.info("\nControls:")
    logger.info("  ▶/◀ or Arrow keys: Step frames")
    logger.info("  ▶▶/◀◀: Skip 10 frames")
    logger.info("  Play/Stop or Space/Escape: Animation")
    
    viz.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Optimized Interactive exiD SVO Visualization')
    parser.add_argument('--data_dir', type=str, default='C:\\exiD-tools\\data')
    parser.add_argument('--recording', type=int, default=25)
    parser.add_argument('--ego_id', type=int, default=None)
    
    args = parser.parse_args()
    main(args.data_dir, args.recording, args.ego_id)