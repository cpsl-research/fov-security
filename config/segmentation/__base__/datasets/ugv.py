_base_ = "./base_dataset.py"

scenes_manager = dict(
    type="UgvScenesManager",
    data_dir="/data/shared/ugv/WILK_BASEMENT",
    split_fracs={
        "train": 0.6,
        "val": 0.4,
        "test": 0.0,
    },
    seed=1,
)
data_output_dir = "/data/shared/fov/fov_bev_segmentation/ugv/benign"

lidar_sensor = "lidar"
frames_stride = 10

max_batches = 300
max_range = 20
_ex = 18
extent = [(-_ex, _ex), (-_ex, _ex)]
