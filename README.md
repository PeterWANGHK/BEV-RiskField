# exiD-DSVO
Modelling of Dynamic Social Value Orientation based on exiD dataset


**visualization of SVO with APF in selected recording frame (example: 25):
```shell
python exid_svo_apf_visualization.py --data_dir "C:\exiD-tools\data" --recording 25 --output_dir "./output"
```

**car interaction visualization (original version, needs enhancement):
```shell
python exid_real_data_viz.py --data_dir "C:\exiD-tools\data" --recording 25
```


**The following programs need the activation of "drone-dataset-tool38":
```shell
cd "C:\exiD-tools\data"
cd src
conda activate drone-dataset-tools38
```

detailed instructions found at: https://github.com/zxc-tju/exiD-tools/tree/master

the dataset structure: 
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
