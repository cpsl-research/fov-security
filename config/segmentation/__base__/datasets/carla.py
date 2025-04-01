_base_ = "./base_dataset.py"

data_output_dir = "/data/shared/fov/fov_bev_segmentation/carla/benign"
scenes_manager = dict(
    type="CarlaScenesManager",
    data_dir="/data/shared/CARLA/multi-agent-random",
    split_fracs={
        "train": 0.6,
        "val": 0.2,
        "test": 0.2,
    },
    seed=1,
)

lidar_sensor = "lidar-0"

max_batches = 1000
max_range = 100
_ex = 80
extent = [(-_ex, _ex), (-_ex, _ex)]
