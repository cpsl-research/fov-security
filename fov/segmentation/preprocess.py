from typing import TYPE_CHECKING, Dict, List, Tuple


if TYPE_CHECKING:
    from avstack.sensors import LidarData

import numpy as np
from avstack.geometry.fov import points_in_fov
from avstack.modules.perception.fov_estimator import FastRayTraceBevLidarFovEstimator

from .dataset import fov_bev_classes


def preprocess_point_cloud(pc: "LidarData", max_range: float = 100) -> "LidarData":
    """Run BEV projection and centering"""
    # filter the points by max range
    pc_filter = pc.filter_by_range(min_range=0, max_range=max_range, inplace=False)

    # convert the pc to bev and center
    pc_bev = pc_filter.project_to_2d_bev(z_min=-3.0, z_max=3.0)
    pc_bev.data.x[:, :2] -= np.mean(pc_bev.data.x[:, :2], axis=0)

    return pc_bev


def point_cloud_to_image(
    pc: "LidarData",
    max_range: float = 100,
    img_size: Tuple[int, int] = (512, 512),
    extent: List[Tuple[float, float]] = [(-80, 80), (-80, 80)],
    max_bin: float = 255,
    do_preprocess: bool = True,
) -> np.ndarray[np.uint8]:
    """Convert point cloud to image"""

    # preprocess the point cloud
    if do_preprocess:
        pc_bev = preprocess_point_cloud(pc, max_range=max_range)
    else:
        # assume we already did the preprocessing
        pc_bev = pc

    # use histogram to make image
    img, _, _ = np.histogram2d(
        x=pc_bev.data.x[:, 0],
        y=pc_bev.data.x[:, 1],
        bins=img_size,
        range=extent,
        density=False,
    )
    img = (np.minimum(max_bin, img)).astype(np.uint8)
    return img


def point_cloud_to_gt_seg(
    pc: "LidarData",
    max_range: float = 100,
    img_size: Tuple[int, int] = (512, 512),
    extent: List[Tuple[float, float]] = [(-80, 80), (-80, 80)],
) -> Tuple[Dict, np.ndarray]:
    """Perform FOV ground truth estimation
    NOTE: the input parameters must match the point cloud to image
    """

    # preprocess the point cloud
    pc_bev = preprocess_point_cloud(pc, max_range=max_range).data.x

    # filter out the points outside the extent
    c1 = pc_bev[:, 0] < extent[0][1]
    c2 = pc_bev[:, 0] > extent[0][0]
    c3 = pc_bev[:, 1] < extent[1][1]
    c4 = pc_bev[:, 1] > extent[1][0]
    pc_mapped = pc_bev[c1 & c2 & c3 & c4, :2]

    # transform the coordinates in the same way the histogram does above
    dx = (extent[0][1] - extent[0][0]) / img_size[0]
    dy = (extent[1][1] - extent[1][0]) / img_size[1]
    pc_mapped = pc_mapped / np.array([dx, dy]) + np.array(img_size) / 2

    # perform ray tracing on the transformed result
    ray_tracer = FastRayTraceBevLidarFovEstimator()
    boundary = ray_tracer._estimate_fov_from_cartesian_lidar(
        pc_bev=pc_mapped,
        n_range_bins=100,
        n_azimuth_bins=100,
        range_max=np.inf,
        centering=True,
    )

    # remove duplicate entries when cast as integer
    bd = list()
    for sublist in boundary.astype(int).tolist():
        if sublist not in bd:
            bd.append(sublist)

    # convert from the boundary to the gt format
    gt_seg = {
        "imgHeight": img_size[1],
        "imgWidth": img_size[0],
        "objects": [
            {
                "label": "visible",
                "polygon": bd,
            }
        ],
    }

    # make the gt image
    gt_img = fov_bev_classes["invisible"] * np.ones(img_size, dtype=np.uint8)
    X, Y = np.meshgrid(range(0, img_size[0]), range(0, img_size[1]))
    coords = np.vstack([X.ravel(), Y.ravel()]).T
    visible = points_in_fov(coords, boundary)
    gt_img[coords[visible, 0], coords[visible, 1]] = fov_bev_classes["visible"]

    return gt_seg, gt_img
