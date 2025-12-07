"""
exiD Dataset: Heavy Vehicle - Car Interaction Visualization
============================================================
Production script for visualizing truck/bus/van - car interactions
during highway merging scenarios.

Designed for the actual exiD dataset structure.
Implements D-SVO (Dynamic Social Value Orientation) framework.

Usage:
    python exid_real_data_viz.py --data_dir "C:\\exiD-tools\\data" --recording 25
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Set
import warnings
import argparse
warnings.filterwarnings('ignore')


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class Config:
    """Configuration parameters for the visualization."""
    # Vehicle classification
    HEAVY_VEHICLE_CLASSES: Set[str] = field(default_factory=lambda: {'truck', 'bus', 'van', 'trailer'})
    CAR_CLASSES: Set[str] = field(default_factory=lambda: {'car'})
    
    # Interaction detection thresholds
    ROI_BEHIND: float = 50.0        # meters behind heavy vehicle
    ROI_AHEAD: float = 100.0        # meters ahead
    LATERAL_THRESHOLD: float = 8.0  # meters lateral distance
    MIN_INTERACTION_FRAMES: int = 50  # minimum 2 seconds at 25fps
    GAP_THRESHOLD: float = 40.0      # meters for "close" interaction
    
    # D-SVO parameters
    MERGE_ZONE_LENGTH: float = 150.0  # typical merge zone length
    
    # Visualization
    TRAIL_LENGTH: int = 75  # frames of trajectory trail
    FPS: int = 25
    
    # Colors
    COLORS: Dict[str, str] = field(default_factory=lambda: {
        'truck': '#E63946',
        'bus': '#E63946', 
        'van': '#F4A261',
        'car': '#457B9D',
        'bg_vehicle': '#ADB5BD',
        'road': '#343A40',
        'merge_zone': '#2A9D8F',
        'danger_zone': '#F4A261',
        'gap_line': '#FFD166'
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
    lon_velocity: float
    lat_velocity: float
    vehicle_class: str
    
    # Lane information
    lat_lane_offset: float = 0.0
    lane_width: float = 3.5
    lanelet_id: int = -1
    lon_lanelet_pos: float = 0.0
    lanelet_length: float = 0.0
    
    # Surrounding vehicles
    lead_id: int = -1
    rear_id: int = -1
    left_lead_id: int = -1
    left_rear_id: int = -1
    right_lead_id: int = -1
    right_rear_id: int = -1
    
    @property
    def speed(self) -> float:
        return np.sqrt(self.x_velocity**2 + self.y_velocity**2)
    
    @property
    def lon_speed(self) -> float:
        return abs(self.lon_velocity)
    
    @property
    def is_heavy_vehicle(self) -> bool:
        return self.vehicle_class.lower() in Config().HEAVY_VEHICLE_CLASSES


@dataclass
class InteractionEvent:
    """Represents a detected interaction between heavy vehicle and car."""
    heavy_vehicle_id: int
    car_id: int
    heavy_vehicle_class: str
    start_frame: int
    end_frame: int
    recording_id: int
    interaction_type: str
    frames: List[int] = field(default_factory=list)
    
    # Computed metrics
    min_gap: float = float('inf')
    min_ttc: float = float('inf')  # Time to collision
    max_relative_velocity: float = 0.0
    d_svo_profile: List[float] = field(default_factory=list)
    
    @property
    def duration_seconds(self) -> float:
        return len(self.frames) / 25.0
    
    def __repr__(self):
        return (f"Interaction(HV[{self.heavy_vehicle_class}]:{self.heavy_vehicle_id} "
                f"<-> Car:{self.car_id}, {self.duration_seconds:.1f}s, "
                f"type={self.interaction_type}, min_gap={self.min_gap:.1f}m)")


# =============================================================================
# Data Loader
# =============================================================================

class ExiDLoader:
    """Loads exiD dataset files."""
    
    # Location to recording mapping
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
        """
        Initialize loader.
        
        Args:
            data_dir: Path to exiD data directory (e.g., "C:\\exiD-tools\\data")
        """
        self.data_dir = Path(data_dir)
        self.tracks_df: Optional[pd.DataFrame] = None
        self.tracks_meta_df: Optional[pd.DataFrame] = None
        self.recording_meta: Optional[pd.Series] = None
        self.current_recording: Optional[int] = None
        self.config = Config()
        
    def load_recording(self, recording_id: int) -> bool:
        """Load a specific recording."""
        self.current_recording = recording_id
        prefix = f"{recording_id:02d}_"
        
        tracks_path = self.data_dir / f"{prefix}tracks.csv"
        meta_path = self.data_dir / f"{prefix}tracksMeta.csv"
        rec_meta_path = self.data_dir / f"{prefix}recordingMeta.csv"
        
        try:
            print(f"Loading recording {recording_id}...")
            
            # Load tracks (can be large)
            print(f"  Loading tracks from {tracks_path}...")
            self.tracks_df = pd.read_csv(tracks_path)
            
            # Load metadata
            self.tracks_meta_df = pd.read_csv(meta_path)
            rec_meta_df = pd.read_csv(rec_meta_path)
            self.recording_meta = rec_meta_df.iloc[0]
            
            # Merge class info into tracks
            self.tracks_df = self.tracks_df.merge(
                self.tracks_meta_df[['trackId', 'class', 'width', 'length']],
                on='trackId',
                how='left',
                suffixes=('', '_meta')
            )
            
            # Use metadata width/length if not in tracks
            if 'width_meta' in self.tracks_df.columns:
                self.tracks_df['width'] = self.tracks_df['width'].fillna(self.tracks_df['width_meta'])
                self.tracks_df['length'] = self.tracks_df['length'].fillna(self.tracks_df['length_meta'])
            
            # Get location info
            location_id = int(self.recording_meta.get('locationId', -1))
            location_name = self.LOCATION_INFO.get(location_id, {}).get('name', 'Unknown')
            
            # Print summary
            print(f"\n{'='*60}")
            print(f"Recording {recording_id} loaded successfully")
            print(f"{'='*60}")
            print(f"  Location: {location_id} ({location_name})")
            print(f"  Duration: {self.recording_meta.get('duration', 0):.1f} seconds")
            print(f"  Frame rate: {self.recording_meta.get('frameRate', 25)} fps")
            print(f"  Total tracks: {len(self.tracks_meta_df)}")
            print(f"  Total frames: {self.tracks_df['frame'].nunique()}")
            
            # Vehicle breakdown
            class_counts = self.tracks_meta_df['class'].value_counts()
            print(f"\n  Vehicle breakdown:")
            for cls, count in class_counts.items():
                print(f"    - {cls}: {count}")
            
            # Heavy vehicles
            heavy_mask = self.tracks_meta_df['class'].str.lower().isin(self.config.HEAVY_VEHICLE_CLASSES)
            heavy_count = heavy_mask.sum()
            print(f"\n  Heavy vehicles (truck/bus/van): {heavy_count}")
            
            return True
            
        except FileNotFoundError as e:
            print(f"Error: File not found - {e}")
            return False
        except Exception as e:
            print(f"Error loading recording: {e}")
            return False
    
    def get_heavy_vehicles(self) -> pd.DataFrame:
        """Get all heavy vehicle track IDs and metadata."""
        mask = self.tracks_meta_df['class'].str.lower().isin(self.config.HEAVY_VEHICLE_CLASSES)
        return self.tracks_meta_df[mask].copy()
    
    def get_cars(self) -> pd.DataFrame:
        """Get all car track IDs and metadata."""
        mask = self.tracks_meta_df['class'].str.lower().isin(self.config.CAR_CLASSES)
        return self.tracks_meta_df[mask].copy()
    
    def get_vehicle_state(self, track_id: int, frame: int) -> Optional[VehicleState]:
        """Get vehicle state at a specific frame."""
        row = self.tracks_df[
            (self.tracks_df['trackId'] == track_id) & 
            (self.tracks_df['frame'] == frame)
        ]
        
        if row.empty:
            return None
        
        row = row.iloc[0]
        
        return VehicleState(
            track_id=int(row['trackId']),
            frame=int(row['frame']),
            x=row['xCenter'],
            y=row['yCenter'],
            heading=np.radians(row.get('heading', 0)),  # Convert to radians
            width=row.get('width', 2.0),
            length=row.get('length', 5.0),
            x_velocity=row.get('xVelocity', 0.0),
            y_velocity=row.get('yVelocity', 0.0),
            lon_velocity=row.get('lonVelocity', 0.0),
            lat_velocity=row.get('latVelocity', 0.0),
            vehicle_class=row['class'],
            lat_lane_offset=row.get('latLaneCenterOffset', 0.0),
            lane_width=row.get('laneWidth', 3.5),
            lanelet_id=int(row.get('laneletId', -1)),
            lon_lanelet_pos=row.get('lonLaneletPos', 0.0),
            lanelet_length=row.get('laneletLength', 0.0),
            lead_id=int(row.get('leadId', -1)),
            rear_id=int(row.get('rearId', -1)),
            left_lead_id=int(row.get('leftLeadId', -1)) if pd.notna(row.get('leftLeadId')) else -1,
            left_rear_id=int(row.get('leftRearId', -1)) if pd.notna(row.get('leftRearId')) else -1,
            right_lead_id=int(row.get('rightLeadId', -1)) if pd.notna(row.get('rightLeadId')) else -1,
            right_rear_id=int(row.get('rightRearId', -1)) if pd.notna(row.get('rightRearId')) else -1,
        )
    
    def get_track_trajectory(self, track_id: int) -> pd.DataFrame:
        """Get full trajectory for a track."""
        return self.tracks_df[self.tracks_df['trackId'] == track_id].sort_values('frame')


# =============================================================================
# Interaction Detector
# =============================================================================

class InteractionDetector:
    """Detects and classifies heavy vehicle - car interactions."""
    
    def __init__(self, loader: ExiDLoader):
        self.loader = loader
        self.config = Config()
        
    def detect_all_interactions(self) -> List[InteractionEvent]:
        """Detect all interactions in the loaded recording."""
        
        heavy_meta = self.loader.get_heavy_vehicles()
        cars_meta = self.loader.get_cars()
        
        if heavy_meta.empty:
            print("No heavy vehicles found.")
            return []
        
        if cars_meta.empty:
            print("No cars found.")
            return []
        
        print(f"\nDetecting interactions...")
        print(f"  Heavy vehicles: {len(heavy_meta)}")
        print(f"  Cars: {len(cars_meta)}")
        
        interactions = []
        
        for _, hv_row in heavy_meta.iterrows():
            hv_id = hv_row['trackId']
            hv_class = hv_row['class']
            hv_interactions = self._detect_for_heavy_vehicle(hv_id, hv_class, cars_meta)
            interactions.extend(hv_interactions)
        
        # Sort by significance (smallest gap first)
        interactions.sort(key=lambda x: x.min_gap)
        
        print(f"\nDetected {len(interactions)} interaction events")
        
        return interactions
    
    def _detect_for_heavy_vehicle(
        self, 
        hv_id: int, 
        hv_class: str,
        cars_meta: pd.DataFrame
    ) -> List[InteractionEvent]:
        """Detect interactions for a single heavy vehicle."""
        
        hv_track = self.loader.get_track_trajectory(hv_id)
        if hv_track.empty:
            return []
        
        hv_frames = set(hv_track['frame'].values)
        interactions = []
        
        for _, car_row in cars_meta.iterrows():
            car_id = car_row['trackId']
            car_track = self.loader.get_track_trajectory(car_id)
            
            if car_track.empty:
                continue
            
            car_frames = set(car_track['frame'].values)
            common_frames = sorted(hv_frames & car_frames)
            
            if len(common_frames) < self.config.MIN_INTERACTION_FRAMES:
                continue
            
            # Check proximity for each common frame
            interaction_frames = []
            
            for frame in common_frames:
                hv_state = hv_track[hv_track['frame'] == frame].iloc[0]
                car_state = car_track[car_track['frame'] == frame].iloc[0]
                
                # Calculate distance
                dx = car_state['xCenter'] - hv_state['xCenter']
                dy = car_state['yCenter'] - hv_state['yCenter']
                distance = np.sqrt(dx**2 + dy**2)
                
                # Check if within ROI
                if distance < self.config.ROI_AHEAD + self.config.ROI_BEHIND:
                    # Lateral check
                    lateral_dist = abs(dy)
                    
                    if lateral_dist < self.config.LATERAL_THRESHOLD or distance < self.config.GAP_THRESHOLD:
                        interaction_frames.append(frame)
            
            # Group into events
            if len(interaction_frames) >= self.config.MIN_INTERACTION_FRAMES:
                events = self._create_events(
                    hv_id, hv_class, car_id, 
                    interaction_frames, hv_track, car_track
                )
                interactions.extend(events)
        
        return interactions
    
    def _create_events(
        self,
        hv_id: int,
        hv_class: str,
        car_id: int,
        frames: List[int],
        hv_track: pd.DataFrame,
        car_track: pd.DataFrame
    ) -> List[InteractionEvent]:
        """Create interaction events from frame list."""
        
        events = []
        frames = sorted(frames)
        
        # Group contiguous frames
        segments = []
        current_segment = [frames[0]]
        
        for i in range(1, len(frames)):
            if frames[i] - frames[i-1] <= 15:  # Allow small gaps
                current_segment.append(frames[i])
            else:
                if len(current_segment) >= self.config.MIN_INTERACTION_FRAMES:
                    segments.append(current_segment)
                current_segment = [frames[i]]
        
        if len(current_segment) >= self.config.MIN_INTERACTION_FRAMES:
            segments.append(current_segment)
        
        # Create events
        for segment in segments:
            interaction_type = self._classify_interaction(hv_track, car_track, segment)
            
            event = InteractionEvent(
                heavy_vehicle_id=hv_id,
                car_id=car_id,
                heavy_vehicle_class=hv_class,
                start_frame=segment[0],
                end_frame=segment[-1],
                recording_id=self.loader.current_recording,
                interaction_type=interaction_type,
                frames=segment
            )
            
            # Compute metrics
            event = self._compute_metrics(event, hv_track, car_track)
            events.append(event)
        
        return events
    
    def _classify_interaction(
        self,
        hv_track: pd.DataFrame,
        car_track: pd.DataFrame,
        frames: List[int]
    ) -> str:
        """Classify interaction type."""
        
        start, end = frames[0], frames[-1]
        
        hv_start = hv_track[hv_track['frame'] == start].iloc[0]
        hv_end = hv_track[hv_track['frame'] == end].iloc[0]
        car_start = car_track[car_track['frame'] == start].iloc[0]
        car_end = car_track[car_track['frame'] == end].iloc[0]
        
        # Position changes
        car_dx = car_end['xCenter'] - car_start['xCenter']
        car_dy = car_end['yCenter'] - car_start['yCenter']
        hv_dy = hv_end['yCenter'] - hv_start['yCenter']
        
        # Relative position
        car_ahead_start = car_start['xCenter'] > hv_start['xCenter']
        car_ahead_end = car_end['xCenter'] > hv_end['xCenter']
        
        # Classify
        if abs(car_dy) > 3.0:  # Significant lateral movement
            if not car_ahead_start and car_ahead_end:
                return 'merge_cut_in'
            elif car_ahead_start and not car_ahead_end:
                return 'merge_behind'
            else:
                return 'lane_change'
        elif abs(hv_dy) > 3.0:
            return 'truck_lane_change'
        elif not car_ahead_start and car_ahead_end:
            return 'car_overtake'
        elif car_ahead_start and not car_ahead_end:
            return 'truck_overtake'
        else:
            return 'following'
    
    def _compute_metrics(
        self,
        event: InteractionEvent,
        hv_track: pd.DataFrame,
        car_track: pd.DataFrame
    ) -> InteractionEvent:
        """Compute detailed metrics for an event."""
        
        for i, frame in enumerate(event.frames):
            hv_row = hv_track[hv_track['frame'] == frame]
            car_row = car_track[car_track['frame'] == frame]
            
            if hv_row.empty or car_row.empty:
                continue
            
            hv_row = hv_row.iloc[0]
            car_row = car_row.iloc[0]
            
            # Gap
            dx = car_row['xCenter'] - hv_row['xCenter']
            dy = car_row['yCenter'] - hv_row['yCenter']
            gap = np.sqrt(dx**2 + dy**2)
            event.min_gap = min(event.min_gap, gap)
            
            # Relative velocity
            dvx = car_row.get('xVelocity', 0) - hv_row.get('xVelocity', 0)
            dvy = car_row.get('yVelocity', 0) - hv_row.get('yVelocity', 0)
            rel_vel = np.sqrt(dvx**2 + dvy**2)
            event.max_relative_velocity = max(event.max_relative_velocity, rel_vel)
            
            # TTC (simplified)
            if rel_vel > 0.1:
                ttc = gap / rel_vel
                event.min_ttc = min(event.min_ttc, ttc)
            
            # D-SVO
            d_svo = self._compute_d_svo(hv_row, car_row, i, len(event.frames))
            event.d_svo_profile.append(d_svo)
        
        return event
    
    def _compute_d_svo(
        self,
        hv_row: pd.Series,
        car_row: pd.Series,
        progress_idx: int,
        total_frames: int
    ) -> float:
        """Compute Dynamic SVO angle."""
        
        progress = progress_idx / max(1, total_frames - 1)
        
        # Gap factor
        dx = car_row['xCenter'] - hv_row['xCenter']
        dy = car_row['yCenter'] - hv_row['yCenter']
        gap = np.sqrt(dx**2 + dy**2)
        gap_factor = np.clip(gap / 50.0, 0, 1)
        
        # Lanelet progress (if available)
        lon_pos = car_row.get('lonLaneletPos', 0)
        lane_length = car_row.get('laneletLength', 100)
        lane_progress = lon_pos / max(1, lane_length)
        
        # D-SVO calculation
        base_svo = 45.0
        
        # Desperation: progress through lane
        desperation = (1 - gap_factor) * lane_progress * 30
        
        # Intimidation from heavy vehicle
        intimidation = (1 - gap_factor) * 15
        
        # Heavy vehicle size factor (bigger = more intimidating)
        hv_length = hv_row.get('length', 5)
        size_factor = min(1.0, hv_length / 18.0) * 10
        
        svo = base_svo - desperation + intimidation + size_factor * (1 - progress)
        
        return np.clip(svo, 0, 90)


# =============================================================================
# Visualizer
# =============================================================================

class InteractionVisualizer:
    """Creates animated visualizations of interactions."""
    
    def __init__(self, loader: ExiDLoader):
        self.loader = loader
        self.config = Config()
        self.fig = None
        self.ax_main = None
        self.ax_svo = None
        self.ax_gap = None
        
    def animate_interaction(
        self,
        event: InteractionEvent,
        save_path: Optional[str] = None,
        show_surrounding: bool = True,
        show_d_svo: bool = True,
        show_no_zone: bool = True,
        padding: float = 40.0
    ) -> Optional[animation.FuncAnimation]:
        """Create animation for an interaction event."""
        
        print(f"\nPreparing animation for {event}...")
        
        # Prepare frame data
        frames_data = self._prepare_frames(event, show_surrounding)
        
        if not frames_data:
            print("No valid frame data.")
            return None
        
        # Calculate bounds
        bounds = self._calculate_bounds(frames_data, padding)
        
        # Create figure
        self.fig = plt.figure(figsize=(16, 10))
        
        # Main view
        self.ax_main = self.fig.add_axes([0.05, 0.30, 0.9, 0.60])
        self._setup_main_axis(bounds, event)
        
        # D-SVO gauge
        if show_d_svo:
            self.ax_svo = self.fig.add_axes([0.05, 0.05, 0.25, 0.12])
            self._setup_svo_gauge()
        
        # Gap plot
        self.ax_gap = self.fig.add_axes([0.35, 0.05, 0.6, 0.12])
        self._setup_gap_plot(frames_data)
        
        # Initialize elements
        self.elements = self._init_elements(show_no_zone, show_surrounding)
        
        # Create animation
        ani = animation.FuncAnimation(
            self.fig,
            self._update_frame,
            frames=len(frames_data),
            fargs=(frames_data, show_no_zone, show_d_svo),
            interval=1000 // self.config.FPS,
            blit=False,
            repeat=True
        )
        
        plt.tight_layout()
        
        if save_path:
            print(f"Saving to {save_path}...")
            writer = animation.PillowWriter(fps=min(self.config.FPS, 15))
            ani.save(save_path, writer=writer, dpi=100)
            print("Saved!")
        
        return ani
    
    def _prepare_frames(
        self, 
        event: InteractionEvent, 
        show_surrounding: bool
    ) -> List[Dict]:
        """Prepare data for each animation frame."""
        
        hv_track = self.loader.get_track_trajectory(event.heavy_vehicle_id)
        car_track = self.loader.get_track_trajectory(event.car_id)
        
        frames_data = []
        
        for i, frame in enumerate(event.frames):
            hv_state = self.loader.get_vehicle_state(event.heavy_vehicle_id, frame)
            car_state = self.loader.get_vehicle_state(event.car_id, frame)
            
            if hv_state is None or car_state is None:
                continue
            
            # Get trajectory history
            hv_history = hv_track[
                (hv_track['frame'] <= frame) &
                (hv_track['frame'] > frame - self.config.TRAIL_LENGTH)
            ][['xCenter', 'yCenter']].values
            
            car_history = car_track[
                (car_track['frame'] <= frame) &
                (car_track['frame'] > frame - self.config.TRAIL_LENGTH)
            ][['xCenter', 'yCenter']].values
            
            # Get surrounding vehicles
            surrounding = []
            if show_surrounding:
                frame_data = self.loader.tracks_df[self.loader.tracks_df['frame'] == frame]
                for _, row in frame_data.iterrows():
                    if row['trackId'] not in [event.heavy_vehicle_id, event.car_id]:
                        dx = row['xCenter'] - hv_state.x
                        dy = row['yCenter'] - hv_state.y
                        if np.sqrt(dx**2 + dy**2) < 80:  # Within view
                            surrounding.append({
                                'x': row['xCenter'],
                                'y': row['yCenter'],
                                'heading': np.radians(row.get('heading', 0)),
                                'width': row.get('width', 1.8),
                                'length': row.get('length', 4.5),
                                'class': row['class']
                            })
            
            # Calculate metrics
            gap = np.sqrt((car_state.x - hv_state.x)**2 + (car_state.y - hv_state.y)**2)
            rel_vel = np.sqrt(
                (car_state.x_velocity - hv_state.x_velocity)**2 +
                (car_state.y_velocity - hv_state.y_velocity)**2
            )
            
            d_svo = event.d_svo_profile[i] if i < len(event.d_svo_profile) else 45.0
            
            frames_data.append({
                'frame': frame,
                'hv': hv_state,
                'car': car_state,
                'hv_history': hv_history,
                'car_history': car_history,
                'surrounding': surrounding,
                'gap': gap,
                'rel_vel': rel_vel,
                'd_svo': d_svo,
                'progress': i / max(1, len(event.frames) - 1)
            })
        
        return frames_data
    
    def _calculate_bounds(
        self, 
        frames_data: List[Dict], 
        padding: float
    ) -> Tuple[float, float, float, float]:
        """Calculate view bounds."""
        
        all_x, all_y = [], []
        
        for fd in frames_data:
            all_x.extend([fd['hv'].x, fd['car'].x])
            all_y.extend([fd['hv'].y, fd['car'].y])
        
        return (
            min(all_x) - padding,
            max(all_x) + padding,
            min(all_y) - padding,
            max(all_y) + padding
        )
    
    def _setup_main_axis(self, bounds: Tuple, event: InteractionEvent):
        """Setup main visualization axis."""
        
        self.ax_main.set_xlim(bounds[0], bounds[1])
        self.ax_main.set_ylim(bounds[2], bounds[3])
        self.ax_main.set_aspect('equal')
        self.ax_main.set_xlabel('X Position (m)', fontsize=11)
        self.ax_main.set_ylabel('Y Position (m)', fontsize=11)
        
        # Get location info
        location_id = int(self.loader.recording_meta.get('locationId', -1))
        location_name = ExiDLoader.LOCATION_INFO.get(location_id, {}).get('name', 'Unknown')
        
        self.ax_main.set_title(
            f'Heavy Vehicle ({event.heavy_vehicle_class.title()}) - Car Interaction\n'
            f'Recording {event.recording_id} | Location {location_id}: {location_name}\n'
            f'Type: {event.interaction_type.replace("_", " ").title()} | '
            f'Duration: {event.duration_seconds:.1f}s',
            fontsize=12, fontweight='bold'
        )
        
        self.ax_main.set_facecolor('#E8E8E8')
        self.ax_main.grid(True, alpha=0.3)
    
    def _setup_svo_gauge(self):
        """Setup D-SVO gauge."""
        
        self.ax_svo.set_xlim(0, 90)
        self.ax_svo.set_ylim(0, 1)
        
        # Gradient
        gradient = np.linspace(0, 1, 100).reshape(1, -1)
        self.ax_svo.imshow(
            gradient, extent=[0, 90, 0, 1],
            aspect='auto', cmap='RdYlGn', alpha=0.4
        )
        
        self.ax_svo.set_xlabel('D-SVO Angle', fontsize=9)
        self.ax_svo.set_yticks([])
        self.ax_svo.set_xticks([0, 45, 90])
        self.ax_svo.set_xticklabels(['0°\nEgoism', '45°\nNeutral', '90°\nAltruism'], fontsize=8)
        self.ax_svo.set_title("Car Driver's Dynamic SVO", fontsize=10, fontweight='bold')
        self.ax_svo.axvline(45, color='gray', linestyle='--', alpha=0.5)
    
    def _setup_gap_plot(self, frames_data: List[Dict]):
        """Setup gap monitoring plot."""
        
        frames = [fd['frame'] for fd in frames_data]
        gaps = [fd['gap'] for fd in frames_data]
        
        self.ax_gap.set_xlim(frames[0], frames[-1])
        self.ax_gap.set_ylim(0, max(gaps) * 1.2)
        self.ax_gap.set_xlabel('Frame', fontsize=9)
        self.ax_gap.set_ylabel('Gap (m)', fontsize=9)
        self.ax_gap.set_title('Gap Distance', fontsize=10, fontweight='bold')
        
        self.ax_gap.plot(frames, gaps, color='#FFD166', alpha=0.3, linewidth=2)
        self.ax_gap.axhline(15, color='red', linestyle='--', alpha=0.5, label='Danger Zone')
        self.ax_gap.legend(loc='upper right', fontsize=8)
    
    def _init_elements(self, show_no_zone: bool, show_surrounding: bool) -> Dict:
        """Initialize plot elements."""
        
        elements = {}
        
        # Heavy vehicle
        elements['hv'] = patches.FancyBboxPatch(
            (0, 0), 16, 2.5,
            boxstyle="round,pad=0.01",
            facecolor=self.config.COLORS['truck'],
            edgecolor='black', linewidth=2,
            alpha=0.95, zorder=10
        )
        self.ax_main.add_patch(elements['hv'])
        
        # Car
        elements['car'] = patches.FancyBboxPatch(
            (0, 0), 4.5, 1.8,
            boxstyle="round,pad=0.01",
            facecolor=self.config.COLORS['car'],
            edgecolor='black', linewidth=2,
            alpha=0.95, zorder=10
        )
        self.ax_main.add_patch(elements['car'])
        
        # Trails
        elements['hv_trail'], = self.ax_main.plot(
            [], [], color=self.config.COLORS['truck'],
            alpha=0.4, linewidth=3, zorder=5
        )
        elements['car_trail'], = self.ax_main.plot(
            [], [], color=self.config.COLORS['car'],
            alpha=0.4, linewidth=3, zorder=5
        )
        
        # No-zone
        if show_no_zone:
            elements['no_zone'] = patches.Ellipse(
                (0, 0), 50, 10,
                facecolor=self.config.COLORS['danger_zone'],
                alpha=0.15, zorder=2
            )
            self.ax_main.add_patch(elements['no_zone'])
        
        # Gap line
        elements['gap_line'], = self.ax_main.plot(
            [], [], color=self.config.COLORS['gap_line'],
            linewidth=3, linestyle='-', alpha=0.8, zorder=8
        )
        
        # Labels (anchored to the right of the main axis to stay outside the animation area)
        elements['hv_label'] = self.ax_main.text(
            1.02, 0.90, '', transform=self.ax_main.transAxes,
            fontsize=10, fontweight='bold', ha='left', va='center', color='white',
            bbox=dict(boxstyle='round', facecolor=self.config.COLORS['truck'], alpha=0.9),
            clip_on=False, zorder=25
        )
        elements['car_label'] = self.ax_main.text(
            1.02, 0.78, '', transform=self.ax_main.transAxes,
            fontsize=10, fontweight='bold', ha='left', va='center', color='white',
            bbox=dict(boxstyle='round', facecolor=self.config.COLORS['car'], alpha=0.9),
            clip_on=False, zorder=25
        )
        
        # Info panel
        elements['info'] = self.ax_main.text(
            0.02, 0.98, '', transform=self.ax_main.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9),
            zorder=20
        )
        
        # D-SVO marker
        if hasattr(self, 'ax_svo') and self.ax_svo:
            elements['svo_marker'] = self.ax_svo.axvline(45, color='black', linewidth=4)
            elements['svo_text'] = self.ax_svo.text(45, 1.15, '45°', ha='center', fontsize=10, fontweight='bold')
        
        # Gap progress
        elements['gap_progress'], = self.ax_gap.plot([], [], color='#FFD166', linewidth=3)
        elements['gap_marker'], = self.ax_gap.plot([], [], 'o', color='red', markersize=10)
        
        # Surrounding vehicles
        elements['surrounding'] = []
        if show_surrounding:
            for _ in range(20):
                rect = patches.FancyBboxPatch(
                    (0, 0), 4.5, 1.8,
                    boxstyle="round,pad=0.01",
                    facecolor=self.config.COLORS['bg_vehicle'],
                    edgecolor='black', linewidth=1,
                    alpha=0.5, zorder=3, visible=False
                )
                self.ax_main.add_patch(rect)
                elements['surrounding'].append(rect)
        
        return elements
    
    def _update_frame(
        self,
        frame_idx: int,
        frames_data: List[Dict],
        show_no_zone: bool,
        show_d_svo: bool
    ):
        """Update animation frame."""
        
        fd = frames_data[frame_idx]
        hv = fd['hv']
        car = fd['car']
        
        # Update vehicles
        self._update_vehicle_patch(self.elements['hv'], hv.x, hv.y, hv.length, hv.width, hv.heading)
        self._update_vehicle_patch(self.elements['car'], car.x, car.y, car.length, car.width, car.heading)
        
        # Update trails
        if len(fd['hv_history']) > 1:
            self.elements['hv_trail'].set_data(fd['hv_history'][:, 0], fd['hv_history'][:, 1])
        if len(fd['car_history']) > 1:
            self.elements['car_trail'].set_data(fd['car_history'][:, 0], fd['car_history'][:, 1])
        
        # Update no-zone
        if show_no_zone and 'no_zone' in self.elements:
            self.elements['no_zone'].set_center((hv.x + hv.length/2 * np.cos(hv.heading), 
                                                  hv.y + hv.length/2 * np.sin(hv.heading)))
            self.elements['no_zone'].angle = np.degrees(hv.heading)
        
        # Update gap line
        self.elements['gap_line'].set_data([hv.x, car.x], [hv.y, car.y])
        
        # Update labels
        hv_speed_kmh = hv.speed * 3.6
        car_speed_kmh = car.speed * 3.6
        
        
        self.elements['hv_label'].set_position((hv.x, hv.y + hv.width/2 + 2))
        self.elements['hv_label'].set_text(f'{hv.vehicle_class.title()}\n{hv_speed_kmh:.0f} km/h')
        
        self.elements['car_label'].set_position((car.x, car.y + car.width/2 + 2))
        self.elements['car_label'].set_text(f'Car\n{car_speed_kmh:.0f} km/h')
        
        # Update info
        info_text = (
            f"Frame: {fd['frame']}\n"
            f"Gap: {fd['gap']:.1f} m\n"
            f"Rel. Speed: {fd['rel_vel']*3.6:.1f} km/h\n"
            f"D-SVO: {fd['d_svo']:.1f}°"
        )
        self.elements['info'].set_text(info_text)
        
        # Update D-SVO gauge
        if show_d_svo and 'svo_marker' in self.elements:
            self.elements['svo_marker'].set_xdata([fd['d_svo'], fd['d_svo']])
            self.elements['svo_text'].set_position((fd['d_svo'], 1.15))
            self.elements['svo_text'].set_text(f"{fd['d_svo']:.0f}°")
            
            if fd['d_svo'] < 25:
                self.elements['svo_text'].set_color('red')
            elif fd['d_svo'] > 55:
                self.elements['svo_text'].set_color('green')
            else:
                self.elements['svo_text'].set_color('black')
        
        # Update gap plot
        frames_so_far = [frames_data[i]['frame'] for i in range(frame_idx + 1)]
        gaps_so_far = [frames_data[i]['gap'] for i in range(frame_idx + 1)]
        self.elements['gap_progress'].set_data(frames_so_far, gaps_so_far)
        self.elements['gap_marker'].set_data([fd['frame']], [fd['gap']])
        
        # Update surrounding vehicles
        for i, rect in enumerate(self.elements['surrounding']):
            if i < len(fd['surrounding']):
                sv = fd['surrounding'][i]
                rect.set_visible(True)
                self._update_vehicle_patch(rect, sv['x'], sv['y'], sv['length'], sv['width'], sv['heading'])
                
                # Color based on class
                color = self.config.COLORS.get(sv['class'].lower(), self.config.COLORS['bg_vehicle'])
                rect.set_facecolor(color)
            else:
                rect.set_visible(False)
        
        return list(self.elements.values())
    
    def _update_vehicle_patch(
        self,
        patch: patches.FancyBboxPatch,
        x: float, y: float,
        length: float, width: float,
        heading: float
    ):
        """Update vehicle patch position."""
        
        half_l = length / 2
        half_w = width / 2
        
        cos_h = np.cos(heading)
        sin_h = np.sin(heading)
        
        corner_x = x - half_l * cos_h + half_w * sin_h
        corner_y = y - half_l * sin_h - half_w * cos_h
        
        patch.set_bounds(corner_x, corner_y, length, width)
        
        t = plt.matplotlib.transforms.Affine2D().rotate_around(x, y, heading) + self.ax_main.transData
        patch.set_transform(t)
    
    def plot_summary(self, interactions: List[InteractionEvent], save_path: Optional[str] = None):
        """Create summary visualization of all interactions."""
        
        if not interactions:
            print("No interactions to summarize.")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Interaction types
        ax1 = axes[0, 0]
        types = [e.interaction_type for e in interactions]
        unique, counts = np.unique(types, return_counts=True)
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique)))
        ax1.bar(range(len(unique)), counts, color=colors)
        ax1.set_xticks(range(len(unique)))
        ax1.set_xticklabels([t.replace('_', '\n') for t in unique], fontsize=9)
        ax1.set_ylabel('Count')
        ax1.set_title('Interaction Types')
        
        # 2. Min gap distribution
        ax2 = axes[0, 1]
        gaps = [e.min_gap for e in interactions if e.min_gap < float('inf')]
        ax2.hist(gaps, bins=20, color='#FFD166', edgecolor='black')
        ax2.axvline(np.mean(gaps), color='red', linestyle='--', label=f'Mean: {np.mean(gaps):.1f}m')
        ax2.set_xlabel('Minimum Gap (m)')
        ax2.set_ylabel('Count')
        ax2.set_title('Minimum Gap Distribution')
        ax2.legend()
        
        # 3. Duration distribution
        ax3 = axes[1, 0]
        durations = [e.duration_seconds for e in interactions]
        ax3.hist(durations, bins=20, color='#457B9D', edgecolor='black')
        ax3.axvline(np.mean(durations), color='red', linestyle='--', label=f'Mean: {np.mean(durations):.1f}s')
        ax3.set_xlabel('Duration (seconds)')
        ax3.set_ylabel('Count')
        ax3.set_title('Interaction Duration')
        ax3.legend()
        
        # 4. D-SVO profiles
        ax4 = axes[1, 1]
        for i, e in enumerate(interactions[:5]):
            if e.d_svo_profile:
                progress = np.linspace(0, 1, len(e.d_svo_profile))
                ax4.plot(progress, e.d_svo_profile, label=f'{e.heavy_vehicle_class}→Car', alpha=0.7)
        
        ax4.axhline(45, color='gray', linestyle='--', alpha=0.5)
        ax4.set_xlabel('Interaction Progress')
        ax4.set_ylabel('D-SVO Angle (°)')
        ax4.set_title('Sample D-SVO Profiles')
        ax4.set_ylim(0, 90)
        ax4.legend(fontsize=8)
        
        plt.suptitle(
            f'Interaction Analysis Summary - Recording {interactions[0].recording_id}\n'
            f'{len(interactions)} Total Interactions',
            fontsize=14, fontweight='bold'
        )
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Summary saved to {save_path}")
        
        plt.show()


# =============================================================================
# Main
# =============================================================================

def main(data_dir: str, recording_id: int, output_dir: str = './output'):
    """Main execution function."""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("exiD Dataset: Heavy Vehicle - Car Interaction Visualization")
    print("=" * 70)
    
    # Load data
    loader = ExiDLoader(data_dir)
    if not loader.load_recording(recording_id):
        return None, None
    
    # Detect interactions
    detector = InteractionDetector(loader)
    interactions = detector.detect_all_interactions()
    
    if not interactions:
        print("No interactions found.")
        return None, None
    
    # Print top interactions
    print(f"\n{'='*70}")
    print("Top 10 Closest Interactions:")
    print("="*70)
    for i, event in enumerate(interactions[:10]):
        print(f"{i+1}. {event}")
    
    # Create visualizer
    visualizer = InteractionVisualizer(loader)
    
    # Create summary
    summary_path = output_path / f'recording_{recording_id}_summary.png'
    visualizer.plot_summary(interactions, save_path=str(summary_path))
    
    # Animate closest interaction
    closest = interactions[0]
    print(f"\nAnimating closest interaction: {closest}")
    
    anim_path = output_path / f'recording_{recording_id}_closest_interaction.gif'
    ani = visualizer.animate_interaction(
        closest,
        save_path=str(anim_path),
        show_surrounding=True,
        show_d_svo=True,
        show_no_zone=True
    )
    
    return interactions, visualizer


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='exiD Heavy Vehicle - Car Interaction Visualization'
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        default='C:\\exiD-tools\\data',
        help='Path to exiD data directory'
    )
    parser.add_argument(
        '--recording',
        type=int,
        default=25,
        help='Recording ID (0-92)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./output',
        help='Output directory'
    )
    
    args = parser.parse_args()
    main(args.data_dir, args.recording, args.output_dir)