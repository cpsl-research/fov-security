import json
from time import time

import numpy as np
import torch
from avapi.carla import CarlaScenesManager
from avstack.config import MODELS, Config
from avstack.modules.perception.fov_estimator import (
    FastRayTraceBevLidarFovEstimator,
    SlowRayTraceBevLidarFovEstimator,
)
from avstack.sensors import LidarData
from polylidar import MatrixDouble, Polylidar3D
from torchvision import transforms

import fov  # to set the registry
from fov.segmentation.dataset import BinaryFovDataset
from fov.segmentation.preprocess import point_cloud_to_image


def time_function(func, repeats, *args, **kwargs):
    dts = []
    for _ in range(repeats):
        t0 = time()
        func(*args, **kwargs)
        dts.append(time() - t0)
    return dts


def initialize_dataset(device):
    """Spin up the datasets for carla"""

    # carla datsaet
    data_root = "/data/shared/CARLA/multi-agent-v1"
    CSM = CarlaScenesManager(data_dir=data_root)
    CDM = CSM.get_scene_dataset_by_index(0)

    # transforms
    class ToDevice:
        def __init__(self, device):
            self.device = device

        def __call__(self, image: torch.Tensor):
            return image.to(self.device)

    # fov dataset
    split = "test"
    data_dir = "/data/shared/fov/fov_bev_segmentation"
    trans = transforms.Compose([ToDevice(device=device)])
    seg_dataset = BinaryFovDataset(
        data_dir=data_dir, transform=trans, transform_mask=trans, split=split
    )

    return CDM, seg_dataset


def ray_trace_preproc(pc: LidarData):
    # projection
    pc_bev = pc.project_to_2d_bev(z_min=-3, z_max=3).data.x[:, :2]
    # centering
    centroid = np.mean(pc_bev[:, :2], axis=0)
    pc_bev[:, :2] -= centroid
    # convert to polar coordinates
    pc_bev_azimuth = np.arctan2(pc_bev[:, 1], pc_bev[:, 0])
    pc_bev_range = np.linalg.norm(pc_bev[:, :2], axis=1)
    return pc_bev_range, pc_bev_azimuth


def fast_ray_trace_execute(pc_bev_range: np.ndarray, pc_bev_azimuth: np.ndarray):
    FastRayTraceBevLidarFovEstimator._estimate_fov_from_polar_lidar(
        pc_bev_range=pc_bev_range,
        pc_bev_azimuth=pc_bev_azimuth,
        n_range_bins=128,
        n_azimuth_bins=128,
        range_max=80,
    )


def slow_ray_trace_execute(pc_bev_range: np.ndarray, pc_bev_azimuth: np.ndarray):
    SlowRayTraceBevLidarFovEstimator._estimate_fov_from_polar_lidar(
        pc_bev_range=pc_bev_range,
        pc_bev_azimuth=pc_bev_azimuth,
        n_range_bins=128,
        n_azimuth_bins=128,
        range_max=80,
        az_tolerance=0.05,
        smoothing=1.0,
    )


def polylidar_preproc(pc: LidarData):
    # projection and data type conversion
    pc_bev = pc.project_to_2d_bev(z_min=-3, z_max=3).data.x[:, :2]
    points_mat = MatrixDouble(pc_bev, copy=False)
    return points_mat


def polylidar_execute(polylidar_model, pc_double_mat: MatrixDouble):
    mesh, _, polygons = polylidar_model.extract_planes_and_polygons(pc_double_mat)
    return mesh, polygons


def segmentation_preproc(pc: LidarData):
    pc_img = point_cloud_to_image(pc=pc, max_range=60, img_size=(512, 512))
    return pc_img


def segmentation_execute(model, pc_img: np.ndarray):
    seg_img = model(pc_img)
    return seg_img


def segmentation_mc_dropout_execute(model, pc_img: np.ndarray, n_iters: int):
    probs = [model(pc_img) for _ in range(n_iters)]
    return probs


def run_preprocessing_analysis(pc_test):
    print("RUNNING PREPROCESSING TIMING TESTS")
    timing_preprocessing = {
        "ray_tracing": None,
        "polylidar": None,
        "segmentation": None,
    }

    timing_preprocessing["ray_tracing"] = time_function(
        ray_trace_preproc, repeats=10, pc=pc_test
    )
    timing_preprocessing["polylidar"] = time_function(
        polylidar_preproc, repeats=10, pc=pc_test
    )
    timing_preprocessing["segmentation"] = time_function(
        segmentation_preproc, repeats=10, pc=pc_test
    )
    print(timing_preprocessing)
    print("done.")
    return timing_preprocessing


def run_executation_analysis(pc_test, pc_img_test, device):
    print("RUNNING EXECUTION TIMING TESTS")
    timing_execution = {
        "fast_ray_tracing": None,
        "slow_ray_tracing": None,
        "polylidar": None,
        "segmentation": None,
        "segmentation_mc_dropout": None,
    }

    pc_bev_range_test, pc_bev_azimuth_test = ray_trace_preproc(pc_test)
    print("running fast ray trace execute...")
    timing_execution["fast_ray_tracing"] = time_function(
        fast_ray_trace_execute,
        repeats=10,
        pc_bev_range=pc_bev_range_test,
        pc_bev_azimuth=pc_bev_azimuth_test,
    )
    print("running slow ray trace execute...")
    timing_execution["slow_ray_tracing"] = time_function(
        slow_ray_trace_execute,
        repeats=10,
        pc_bev_range=pc_bev_range_test,
        pc_bev_azimuth=pc_bev_azimuth_test,
    )
    print("running polylidar execute...")
    polylidar_model = Polylidar3D(lmax=1.8, min_triangles=30)
    pc_double_mat_test = polylidar_preproc(pc_test)
    timing_execution["polylidar"] = time_function(
        polylidar_execute,
        repeats=10,
        polylidar_model=polylidar_model,
        pc_double_mat=pc_double_mat_test,
    )

    cfg_unet = Config.fromfile("../../config/segmentation/unet.py")
    cfg_unet_mc_dropout = Config.fromfile(
        "../../config/segmentation/unet_mc_dropout.py"
    )
    unet_model = MODELS.build(cfg_unet["model"]).to(device)
    unet_model_mc_dropout = MODELS.build(cfg_unet_mc_dropout["model"]).to(device)
    unet_model_mc_dropout.enable_eval_dropout()

    print("running segmentation execute...")
    timing_execution["segmentation"] = time_function(
        segmentation_execute, repeats=10, model=unet_model, pc_img=pc_img_test
    )
    print("running segmentation with mc dropout execute...")
    timing_execution["segmentation_mc_dropout"] = time_function(
        segmentation_mc_dropout_execute,
        repeats=10,
        model=unet_model_mc_dropout,
        pc_img=pc_img_test,
        n_iters=5,
    )


def main():
    # set the data
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    CDM, seg_dataset = initialize_dataset(device)
    pc_test = CDM.get_lidar(
        frame=CDM.get_frames("lidar-0", 0)[0], sensor="lidar-0", agent=0
    )
    pc_img_test, pc_mask_test, pc_metadata_test = seg_dataset[0]

    # preprocessing timing analysis
    timing_out = {
        "preprocessing": run_preprocessing_analysis(pc_test),
        "execution": run_executation_analysis(pc_test, pc_img_test, device),
    }

    # save results
    with open("timing_results.json", "w") as f:
        json.dump(timing_out, f)


if __name__ == "__main__":
    main()
