# exiD-DSVO
Modelling of Dynamic Social Value Orientation based on exiD dataset

(My exiD dataset has been archived in "C:\exiD-tools\data", so the data directory needs to be tailored for other users)
## complete visualization of mutual SVO with symmetric evaluations:
```shell
python exid_enhanced_svo_visualization.py --data_dir "C:\exiD-tools\data" --recording 25 --output_dir "./enhanced_output"
```

## visualization of SVO with APF in selected recording frame (example: 25):
```shell
python exid_svo_apf_visualization.py --data_dir "C:\exiD-tools\data" --recording 25 --output_dir "./output"
```

## car interaction visualization (enhanced version, pending frame enhancements):
```shell
# Interactive visualization (main program)
python exid_optimized_visualization.py --data_dir /path/to/exid/data --recording 25

# Static analysis plots
python exid_interactive_visualization.py --data_dir /path/to/exid/data --recording 25
```

## The following programs need the activation of "drone-dataset-tool38":
```shell
cd "C:\exiD-tools\data"
cd src
conda activate drone-dataset-tools38
```

detailed instructions found at: https://github.com/zxc-tju/exiD-tools/tree/master

## Dataset structure: 
```shell
C:\exiD-tools\data\
├── 00_tracks.csv
├── 00_tracksMeta.csv
├── 00_recordingMeta.csv
├── 00_background.png
├── 01_tracks.csv
├── ...
├── 25_tracks.csv          ← recording
├── 25_tracksMeta.csv      ← Vehicle metadata
├── 25_recordingMeta.csv   ← Recording info
├── 25_background.png      ← Aerial image
├── ...
└── Maps/
    ├── location1.osm      ← Lanelet2 HD map
    └── location1.xodr     ← OpenDrive HD map
```

Acknowledgement of the dataset:
```
@inproceedings{exiDdataset,
               title={The exiD Dataset: A Real-World Trajectory Dataset of Highly Interactive Highway Scenarios in Germany},
               author={Moers, Tobias and Vater, Lennart and Krajewski, Robert and Bock, Julian and Zlocki, Adrian and Eckstein, Lutz},
               booktitle={2022 IEEE Intelligent Vehicles Symposium (IV)},
               pages={958-964},
               year={2022},
               doi={10.1109/IV51971.2022.9827305}}
```
