_base_ = "./base_dataset.py"

scenes_manager = dict(
    type="nuScenesManager",
    data_dir="/data/shared/nuScenes",
    split="v1.0-trainval",
    max_scenes=25,
)
data_output_dir = "/data/shared/fov/fov_bev_segmentation/nuscenes/benign"

lidar_sensor = "LIDAR_TOP"
frames_stride = 1

max_batches = 1000
max_range = 100
_ex = 80
extent = [(-_ex, _ex), (-_ex, _ex)]
