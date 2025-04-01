from typing import List, Tuple

import avapi  # noqa
import numpy as np
import torch
from avstack.config import DATASETS, MODELS
from avstack.geometry.fov import points_in_fov
from avstack.modules.perception.fov_estimator import (  # PolyLidarFovEstimator,
    ConcaveHullLidarFOVEstimator,
    FastRayTraceBevLidarFovEstimator,
    SlowRayTraceBevLidarFovEstimator,
)
from torchvision import transforms

import fov  # noqa
from fov.segmentation.dataset import BinaryFovDataset


def get_dataset(cfg, device, split="test"):
    """Load in the datasets"""

    class ToDevice:
        def __init__(self, device):
            self.device = device

        def __call__(self, image: torch.Tensor):
            return image.to(self.device)

    # original dataset
    SM = DATASETS.build(cfg["scenes_manager"])

    # segmentation dataset
    trans = transforms.Compose([
        ToDevice(device=device),
        transforms.Resize(size=cfg.get("model_io_size", cfg["img_size"])),
    ])
    seg_dataset = BinaryFovDataset(
        data_dir=cfg["data_output_dir"],
        max_range=cfg["max_range"],
        extent=cfg["extent"],
        img_size=cfg["img_size"],
        transform=trans,
        transform_mask=trans,
        split=split,
    )

    return SM, seg_dataset


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
        device: str,
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


def get_unet_model(
    cfg,
    device,
    weight_dir: str,
):
    """Load unet model"""
    model = MODELS.build(cfg["model"]).to(device)
    try:
        model.load_weights_subdir(weight_dir, epoch=-1)
    except Exception as e:
        print(f"Cannot load model...config is {cfg}")
        raise e
    if cfg["is_mc"]:
        print("Enabling dropout at test time for MC model")
        model.enable_eval_dropout()
    unet_model = WrapperUnet(model)
    return unet_model


def get_polygon_model(
    model: str,
    extent: List,
    device: str = "cpu",
    img_size=(512, 512),
):
    """Load the polygon model"""
    if model == "fast_ray_trace":
        algorithm = FastRayTraceBevLidarFovEstimator(
            z_min=-3.0,
            z_max=3.0,
            n_azimuth_bins=128,
            n_range_bins=128,
            range_max=70.0,
        )
    elif model == "slow_ray_trace":
        algorithm = SlowRayTraceBevLidarFovEstimator(
            z_min=-3.0,
            z_max=3.0,
            n_azimuth_bins=128,
            n_range_bins=128,
            range_max=70.0,
            az_tol=0.05,
            smoothing=1.0,
        )
    elif model == "concave_hull":
        algorithm = ConcaveHullLidarFOVEstimator(
            concavity=1,
            length_threshold=4,
            max_height=3,
        )
    elif model == "polylidar":
        # model = WrapperPolygon(
        #     PolyLidarFovEstimator(
        #         lmax=1.8,
        #         min_triangles=30,
        #     )
        # )
        raise NotImplementedError(model)
    else:
        raise NotImplementedError(model)

    return WrapperPolygon(
        model=algorithm, extent=extent, img_size=img_size, device=device
    )


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
