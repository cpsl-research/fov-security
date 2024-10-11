from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from avstack.sensors import LidarData

import numpy as np
import torch
from avstack.modules.perception.fov_estimator import FastRayTraceBevLidarFovEstimator
from torch_geometric.data import Data

from fov.preprocess import preprocess_point_cloud_for_bev


def add_range_az_features(pc_bev: "LidarData"):
    """Augment the BEV point cloud with range, azimuth features"""
    pc_bev_azimuth = np.arctan2(pc_bev.data.x[:, 1], pc_bev.data.x[:, 0])
    pc_bev_range = np.linalg.norm(pc_bev.data.x, axis=1)
    pc_bev.data.x = np.insert(pc_bev.data.x, [2], pc_bev_range[:, None], axis=1)
    pc_bev.data.x = np.insert(pc_bev.data.x, [3], pc_bev_azimuth[:, None], axis=1)

    return pc_bev


def get_edges_point_cloud_bev(
    pc_bev: "LidarData",
    n_closest_edges: int = 5,
) -> torch.Tensor:
    """Takes in a point cloud and gets edges and features

    pc_bev.data.x has features of [x, y, rng, az, intensity]
    """
    n_idx_side = n_closest_edges // 2
    idx_rng, idx_az = 2, 3

    # ----------------------------------------
    # get indices of edges to link together
    # ----------------------------------------
    idx_pm = [i for i in range(-n_idx_side, n_idx_side + 1) if i != 0]
    idxs_add = idx_pm * np.ones((len(pc_bev), 2 * n_idx_side), dtype=int)
    az_idx_sorted = np.argsort(pc_bev.data.x[:, idx_az])
    edge_i = np.repeat(az_idx_sorted, 2 * n_idx_side).astype(int)
    idx_index_az = (
        np.array(range(len(az_idx_sorted)))[:, None] + idxs_add
    ).ravel() % len(az_idx_sorted)
    edge_j = az_idx_sorted[idx_index_az]
    edge_index = torch.tensor(np.c_[edge_i, edge_j].T, dtype=torch.long)

    # ----------------------------------------
    # get the features for each edge
    # ----------------------------------------
    # feature: cartesian edge length
    edge_length = np.linalg.norm(
        pc_bev.data.x[edge_i, :idx_rng] - pc_bev.data.x[edge_j, :idx_rng], axis=1
    )

    # feature: edge azimuth differential (mod to [-np.pi, np.pi]
    edge_delta_az = pc_bev.data.x[edge_i, idx_az] - pc_bev.data.x[edge_j, idx_az]
    edge_delta_az = (edge_delta_az + np.pi) % (2 * np.pi) - np.pi

    # package up the features
    edge_attr = torch.tensor(np.c_[edge_length, edge_delta_az], dtype=torch.long)

    return edge_index, edge_attr


def get_node_ground_truth_bev_boundary(
    pc_bev: "LidarData", d_boundary_threshold: float = 0.5
) -> torch.Tensor:
    """Gets ground truth labels on which points form the vertex set

    Any point within a threshold of the FOV boundary gets a ground
    truth value of true for the boundary detection problem
    """
    # run fov estimator to get polygon
    ray_tracer = FastRayTraceBevLidarFovEstimator()
    polygon = ray_tracer(pc_bev)

    # get the distance of each point to the polygon
    close_to_boundary = (
        polygon.distance_points(pc_bev.data.x[:, :2]) < d_boundary_threshold
    )
    y = torch.tensor(close_to_boundary, dtype=torch.bool)

    return y


def point_cloud_to_bev_graph(
    pc: "LidarData",
    n_closest_edges: int = 50,
    include_target: bool = False,
):
    """Converts the point clouds to graphs

    Nodes: point cloud points -- cartesian or polar??
        features:
            if just cartesian, then: (x, y, intensity)
            if polar, then (range, azimuth, intensity)
            if point then (x, y, range, azimuth, intensity)
    Edges: connections between neighbors of points -- cartesian or polar neighbors?
    Class: tbd
    """
    # preprocess the point cloud into a bev reference
    pc_bev = preprocess_point_cloud_for_bev(pc)
    pc_bev = add_range_az_features(pc_bev)
    nodes = torch.tensor(pc_bev.data.x, dtype=torch.float)

    # get the edge indices and features
    edge_index, edge_attr = get_edges_point_cloud_bev(
        pc_bev=pc_bev, n_closest_edges=n_closest_edges
    )

    # if including target, run ground truth determination
    y = get_node_ground_truth_bev_boundary(pc_bev) if include_target else None

    # for the graph data structure
    graph = Data(
        x=nodes,
        edge_index=edge_index,
        edge_attr=edge_attr,
        pos=pc_bev.data.x[:, :2],
        y=y,
    )

    return graph
