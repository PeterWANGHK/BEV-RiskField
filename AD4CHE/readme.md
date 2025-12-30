# AD4CHE Role & Occlusion Analysis

## Overview

This program framework is adapted from the exiD dataset analysis program for use with the **AD4CHE (Aerial Dataset for China Congested Highway and Expressway)** format. It provides:

1. **Agent Role Classification** - Identifies vehicle behavioral roles (Normal, Merging, Exiting, Yielding)
2. **Occlusion Detection** - Detects when trucks/buses block the view between vehicles
3. **Ego-as-Occluder Analysis** - Shows occlusions caused BY the ego truck
4. **Visualization** - Traffic snapshots, role distributions, and occlusion tables
5. **CSV Logging** - Exports occlusion events for training/analysis

![The example visualization of the occlusion scenarios in AD4CHE highway dataset](02_ego_occlusion.png)

## AD4CHE-Specific Features Used

- **`drivingDirection`**: 1 = upper lanes (left direction), 2 = lower lanes (right direction)
- **`laneId`**: Direct lane assignment (no inference needed)
- **`orientation`**: Vehicle heading already in radians
- **`ego_offset`**: Offset from lane center line
- **`yaw_rate`**: Vehicle yaw rate
- **Surrounding vehicle IDs**: `precedingId`, `followingId`, `leftPrecedingId`, `leftAlongsideId`, `leftFollowingId`, `rightPrecedingId`, `rightAlongsideId`, `rightFollowingId`

## Usage
(the directory of the AD4CHE should be changed from ./data to the specific directory, here ./data is the default name for example)

```bash
# Basic usage
python ad4che_role_occlusion_analysis.py --data_dir ./data --recording 1

# Specify ego vehicle and frame
python ad4che_role_occlusion_analysis.py --data_dir ./data --recording 1 --ego_id 5 --frame 100

# Skip animation generation
python ad4che_role_occlusion_analysis.py --data_dir ./data --recording 1 --no-animation

# Log occlusions for all frames (for training data)
python ad4che_role_occlusion_analysis.py --data_dir ./data --recording 1 --log-all-frames
```

## Command Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--data_dir` | `./data` | Path to AD4CHE data directory |
| `--recording` | `1` | Recording ID to analyze |
| `--ego_id` | None | Ego vehicle ID (auto-selects first truck/bus if None) |
| `--frame` | None | Frame to analyze (auto-selects best interaction frame if None) |
| `--output_dir` | `./output_ad4che` | Output directory |
| `--no-animation` | False | Skip animation generation |
| `--log-all-frames` | False | Log occlusions for all frames to CSV |

## Output Files

The program creates an organized output folder with:

```
output_ad4che/
└── rec01_ego5_frame100/
    ├── 01_traffic_snapshot.png    # Full traffic view with roles & occlusions
    ├── 02_ego_occlusion.png       # Occlusions caused BY the ego truck
    ├── 03_role_distribution.png   # Bar chart of agent roles
    ├── 04_occlusion_table.png     # Table of occlusion events
    ├── 05_summary.png             # Analysis summary statistics
    ├── 06_combined.png            # Combined dashboard view
    ├── animation.gif              # Animated traffic sequence
    ├── occlusion_log.csv          # CSV of occlusion events
    └── metadata.json              # Analysis metadata
```
## occlusion scenario with combined field modeling:
![The example visualization of the occlusion scenarios in AD4CHE highway dataset](occlusion_rec23_ego294_frame3609.png)

## Occlusion Research Context

The occlusion analysis is designed for studying:
- **Ego truck as occluder**: When a truck blocks the view between following cars and merging vehicles
- **Uncertainty in interaction**: Cars behind trucks cannot see vehicles on the merging lane
- **Role-based analysis**: Different vehicle behaviors (merging urgency, yielding, etc.)

### Vehicle Roles:
- **Normal Main**: Standard car-following behavior on main lane
- **Merging**: Vehicles on acceleration lane waiting to join main flow
- **Exiting**: Vehicles moving from inner lanes to rightmost lane for off-ramp
- **Yielding**: Vehicles decelerating to allow merging

## Dependencies

```
numpy
pandas
matplotlib
scipy (optional, for peak detection)
```

## Code Structure

- `AgentRole`, `OcclusionType`: Enum definitions
- `AgentState`, `OcclusionEvent`: Data classes
- `Config`: Configuration parameters
- `RoleClassifier`: Classifies agent behavioral roles
- `OcclusionDetector`: Detects occlusion relationships
- `AD4CHERoleLoader`: Loads AD4CHE data files
- `OcclusionLogger`: Exports occlusion events to CSV
- `RoleOcclusionVisualizer`: Creates visualizations
- `OutputManager`: Manages output file organization
- `analyze_recording()`: Main analysis pipeline
