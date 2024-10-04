from typing import List, Tuple

import numpy as np
import torch
from avapi.carla import CarlaScenesManager
from avstack.config import MODELS, Config
from avstack.geometry.fov import points_in_fov
from avstack.modules.perception.fov_estimator import (  # PolyLidarFovEstimator,
    ConcaveHullLidarFOVEstimator,
    FastRayTraceBevLidarFovEstimator,
    SlowRayTraceBevLidarFovEstimator,
)
from torchvision import transforms

import fov  # noqa
from fov.segmentation.dataset import BinaryFovDataset


def get_datasets(device, split="test"):
    """Load in the datasets"""
    # carla dataset
    data_root = "/data/shared/CARLA/multi-agent-v1"
    CSM = CarlaScenesManager(data_dir=data_root)

    class ToDevice:
        def __init__(self, device):
            self.device = device

        def __call__(self, image: torch.Tensor):
            return image.to(self.device)

    # segmentation dataset
    trans = transforms.Compose([ToDevice(device=device)])

    # benign dataset
    data_dir_benign = "/data/shared/fov/fov_bev_segmentation"
    seg_dataset_benign = BinaryFovDataset(
        data_dir=data_dir_benign, transform=trans, transform_mask=trans, split=split
    )

    # adversarial dataset
    data_dir_adversarial = "/data/shared/fov/fov_bev_segmentation_adversarial"
    seg_dataset_adversarial = BinaryFovDataset(
        data_dir=data_dir_adversarial,
        transform=trans,
        transform_mask=trans,
        split=split,
    )
    datasets = {"benign": seg_dataset_benign, "adversarial": seg_dataset_adversarial}

    return CSM, datasets


class WrapperUnet:
    def __init__(self, model):
        self.model = model

    def __call__(self, pc_img, pc_np, metadata) -> np.ndarray:
        pred_img = self.model(pc_img)
        return pred_img


class WrapperPolygon:
    def __init__(
        self,
        model,
        extent: List[Tuple[float, float]],
        img_size: Tuple[int, int],
        device,
    ):
        """Wraps a polygon fitting algorithm to segmentation model"""
        self.model = model
        self.extent = extent
        self.device = device
        self.img_size = img_size
        X, Y = np.meshgrid(range(0, img_size[0]), range(0, img_size[1]))
        self.coords = np.vstack([X.ravel(), Y.ravel()]).T

    def __call__(self, pc_img, pc_np, metadata) -> np.ndarray:
        """Get the polygon then convert to segmentation mask"""
        # load the original point cloud via metadata
        # CDM = CSM.get_scene_dataset_by_name(metadata["scene"])
        # pc = CDM.get_lidar(
        #     frame=metadata["frame"],
        #     sensor=metadata["sensor"],
        #     agent=metadata["agent"],
        # )

        # apply model to the point cloud
        # polygon = self.model.(pc)
        boundary = self.model.call_on_array(pc_np)

        # project polygon into space of image
        dx = (self.extent[0][1] - self.extent[0][0]) / self.img_size[0]
        dy = (self.extent[1][1] - self.extent[1][0]) / self.img_size[1]
        boundary = boundary / np.array([dx, dy]) + np.array(self.img_size) / 2

        # populate the image results
        img_out = np.zeros(self.img_size, dtype=float)
        visible = points_in_fov(self.coords, boundary)
        img_out[self.coords[visible, 0], self.coords[visible, 1]] = 1.0

        # make a torch tensor
        img_out = torch.tensor(img_out).to(self.device)
        return img_out


def get_models(
    device,
    cfg_names=["unet", "unet_mc", "unet_adversarial", "unet_mc_adversarial"],
    img_size=(512, 512),
):
    """Load all unet models"""
    unet_models = {}
    for cfg_name in cfg_names:
        cfg = Config.fromfile(f"../../config/segmentation/{cfg_name}.py")
        model = MODELS.build(cfg["model"]).to(device)
        model.load_weights_subdir(
            f"../../scripts/segmentation_training/{cfg_name}", epoch=-1
        )
        if "mc" in cfg_name:
            model.enable_eval_dropout()
        unet_models[cfg_name] = WrapperUnet(model)

    # store models in dictionary
    models = {
        "fast_ray_trace": WrapperPolygon(
            FastRayTraceBevLidarFovEstimator(
                z_min=-3.0,
                z_max=3.0,
                n_azimuth_bins=128,
                n_range_bins=128,
                range_max=70.0,
            ),
            img_size=img_size,
            extent=[(-80, 80), (-80, 80)],
            device=device,
        ),
        "slow_ray_trace": WrapperPolygon(
            SlowRayTraceBevLidarFovEstimator(
                z_min=-3.0,
                z_max=3.0,
                n_azimuth_bins=128,
                n_range_bins=128,
                range_max=70.0,
                az_tol=0.05,
                smoothing=1.0,
            ),
            img_size=img_size,
            extent=[(-80, 80), (-80, 80)],
            device=device,
        ),
        "concave_hull": WrapperPolygon(
            ConcaveHullLidarFOVEstimator(
                concavity=1,
                length_threshold=4,
                max_height=3,
            ),
            img_size=img_size,
            extent=[(-80, 80), (-80, 80)],
            device=device,
        ),
        # "polylidar": WrapperPolygon(
        #     PolyLidarFovEstimator(
        #         lmax=1.8,
        #         min_triangles=30,
        #     )
        # ),
        **unet_models,
    }
    return models


def get_title_mappings():
    # map the model names to titles
    model_titles = {
        "gt": "Ground Truth FOV",
        "fast_ray_trace": "Quantized Ray Trace",
        "slow_ray_trace": "Continuous Ray Trace",
        "concave_hull": "Concave Hull Polygon",
        "unet": "MLE UNet",
        "unet_mc": "MCD UNet",
        "unet_adversarial": "MLE UNet + Adv-Train",
        "unet_mc_adversarial": "MCD UNet + Adv-Train",
    }

    # map the adversary models to titles
    adv_titles = {
        "none": "Benign Data",
        "uniform": "Uniformly Spoofed",
        "cluster": "Clustered Spoofed",
    }
    return model_titles, adv_titles
