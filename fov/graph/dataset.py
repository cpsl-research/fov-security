from typing import Callable, Optional

import numpy as np
import torch
from avapi.carla import CarlaScenesManager
from torch_geometric.data import InMemoryDataset
from tqdm import tqdm

from fov.graph.preprocess import point_cloud_to_bev_graph


class CarlaFieldOfViewDataset(InMemoryDataset):
    def __init__(
        self,
        carla_root_directory: str,
        graph_root_directory: str,
        include_infrastructure_agents: bool,
        frame_stride: int = 1,
        n_closest_edges: int = 10,
        n_frames_max: int = None,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        log: bool = True,
        force_reload: bool = False,
    ) -> None:
        """Initialize and potentially preprocess the dataset"""
        if include_infrastructure_agents:
            graph_root_directory = graph_root_directory + "_with_infra"
        self.n_closest_edges = n_closest_edges
        self.include_infrastructure_agents = include_infrastructure_agents
        self.carla_root_directory = carla_root_directory
        self.frame_stride = frame_stride
        self.n_frames_max = n_frames_max if n_frames_max is not None else np.inf
        super().__init__(
            root=graph_root_directory,
            transform=transform,
            pre_transform=pre_transform,
            pre_filter=pre_filter,
            log=log,
            force_reload=force_reload,
        )
        self.data, self.slices = torch.load(self.processed_paths[0])

    def raw_file_names(self):
        return []  # TODO: when hosting dataset, fill this in for downloading

    def processed_file_names(self):
        # Will check whether the file(s) in this list is already there in the "root" directory.
        return ["data.pt"]

    def process(self):
        """Load the raw data and convert it to a graph format"""
        CSM = CarlaScenesManager(data_dir=self.carla_root_directory)

        # loop over all the carla scenes
        data_list = []
        print(f"...processing {len(CSM)} scenes")
        i_frame = 0
        for CDM in tqdm(CSM):
            for agent in CDM.get_agents(CDM.frames[0]):
                for frame in CDM.get_frames(sensor="lidar-0", agent=agent.ID)[
                    :: self.frame_stride
                ]:
                    if "static" in agent.obj_type:
                        # possibly an infrastructure agent
                        if self.include_infrastructure_agents:
                            raise NotImplementedError
                        else:
                            continue
                    else:
                        # ground agent
                        pc = CDM.get_lidar(
                            frame=frame, sensor="lidar-0", agent=agent.ID
                        )

                    # convert to graph
                    graph = point_cloud_to_bev_graph(
                        pc=pc,
                        n_closest_edges=self.n_closest_edges,
                        include_target=True,
                    )
                    data_list.append(graph)
                    i_frame += 1
                    if (i_frame % 50) == 0:
                        print(f"processed {i_frame} frames")
                    if i_frame > self.n_frames_max:
                        print(f"Hit max number of frames ({self.n_frames_max})!")
                        break
                else:
                    continue
                break
            else:
                continue
            break

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def download(self):
        pass
