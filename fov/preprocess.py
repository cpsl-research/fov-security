from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from avstack.sensors import LidarData

import numpy as np


def preprocess_point_cloud_for_bev(
    pc: "LidarData",
    z_min: float = -3.0,
    z_max: float = 3.0,
    max_range: float = 100,
    keep_features: bool = False,
) -> "LidarData":
    """Applies preprocessing and filtering to the point cloud

    Steps:
        - project to ground plane
        - filter to between (zmin, zmax) height in cartesian
        - (if infrastruscture) center the pointcloud about centroid

    Output is lidar data with the features as follows:
        x: cartesian x position
        y: cartesian y position
        range: polar range on [0, inf]
        azimuth: polar azimuth angle on [-pi, pi]
        intensity: returned intensity on [0, 1]
    """
    # steps 1-2: project to ground plane and filter to height
    pc_filter = pc.filter_by_range(min_range=0, max_range=max_range, inplace=False)
    pc_bev = pc_filter.project_to_2d_bev(z_min=z_min, z_max=z_max)

    # remove features, if asked
    if not keep_features:
        pc_bev.data.x = pc_bev.data.x[:, :2]

    # step 3: center on the centroid if infrastructure
    centroid = np.mean(pc_bev.data.x[:, :2], axis=0)
    pc_bev.data.x[:, :2] -= centroid
    pc_bev.reference.x[:2] += centroid

    return pc_bev
