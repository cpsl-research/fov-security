import json
import os
from typing import Callable, Optional

import numpy as np
import torch
from skimage import io
from torch.utils.data import Dataset


fov_bev_classes = {
    "invisible": 0,
    "visible": 1,
}

fov_bev_pallete = {
    0: [128, 64, 128],
    1: [10, 120, 232],
}


fov_bev_pallete_matrix = np.zeros((0, 3), dtype=np.uint8)
for i in range(len(fov_bev_pallete)):
    fov_bev_pallete_matrix = np.concatenate(
        (
            fov_bev_pallete_matrix,
            np.array(fov_bev_pallete[i], dtype=np.uint8)[:, None].T,
        ),
        axis=0,
    )


class BinaryFovDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        split: str,
        transform: Optional[Callable] = None,
        transform_mask: Optional[Callable] = None,
    ):
        self.data_dir = data_dir
        self.transform = transform
        self.transform_mask = transform_mask
        self.split = split

    def __len__(self):
        _, _, files = next(os.walk(os.path.join(self.data_dir, "img", self.split)))
        file_count = len(files)
        return file_count

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # specity the paths
        img_path = os.path.join(self.data_dir, "img", self.split, f"{idx:08d}.png")
        mask_path = os.path.join(
            self.data_dir, "ann", self.split, f"{idx:08d}_segimage.png"
        )

        # load with scikit and do scaling
        image = torch.tensor(
            (io.imread(img_path) / 1000)[np.newaxis], dtype=torch.float
        )
        mask = torch.tensor(io.imread(mask_path)[np.newaxis], dtype=torch.bool)

        # apply transforms
        if self.transform:
            image = self.transform(image)
        if self.transform_mask:
            mask = self.transform_mask(mask)

        return image, mask

    def get_metadata(self, idx):
        meta_path = os.path.join(
            self.data_dir, "ann", self.split, f"{idx:08d}_polygons.json"
        )
        with open(meta_path, "r") as f:
            metadata = json.load(f)
        return metadata

    def get_pointcloud(self, idx):
        pc_path = os.path.join(self.data_dir, "pcs", self.split, f"{idx:08d}.npy")
        pc_bev = np.load(pc_path)
        return pc_bev
