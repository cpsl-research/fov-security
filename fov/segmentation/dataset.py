import json
import os
from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np
import torch
from skimage import io
from torch.utils.data import Dataset

from avstack.geometry import PointMatrix2D
from avstack.geometry.fov import Polygon

from fov.segmentation.preprocess import point_cloud_to_image


def fill_holes(image: np.ndarray):
    """Fills holes in a binary image."""
    # Threshold the image to ensure it's binary (if needed)
    # _, thresh = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

    # Copy the thresholded image
    im_floodfill = image.copy()

    # Mask used for flood fill.
    # Size needs to be 2 pixels larger than the image
    h, w = image.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)

    # Flood fill from point (0, 0)
    cv2.floodFill(im_floodfill, mask, (0,0), 255)

    # Invert flood filled image
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)

    # Combine the two images to get the filled holes
    im_out = image | im_floodfill_inv

    return im_out


class BinaryFovDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        split: str,
        max_range: float,
        extent: List,
        img_size: Tuple,
        transform: Optional[Callable] = None,
        transform_mask: Optional[Callable] = None,
    ):
        self.data_dir = data_dir
        self.transform = transform
        self.transform_mask = transform_mask
        self.split = split
        self._max_range = max_range
        self._extent = extent
        self._img_size = img_size

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
    
    def get_polygon(self, idx):
        poly_path = os.path.join(
            self.data_dir, "ann", self.split, f"{idx:08d}_polygons.json"
        )
        with open(poly_path, "r") as f:
            poly = json.load(f)["objects"][0]["polygon"]
        return np.array(poly)
        
    def get_polygon_avstack(self, idx, SM):
        metadata = self.get_metadata(idx)
        SD = SM.get_scene_dataset_by_name(metadata["scene"])
        reference = SD.get_calibration(
            frame=metadata["frame"],
            agent=metadata["agent"],
            sensor=metadata["sensor"],
        ).reference
        img_size = np.array([metadata["imgWidth"], metadata["imgHeight"]])
        scaling = np.array([metadata["objects"][0]["dx"], metadata["objects"][0]["dy"]])
        poly = (self.get_polygon(idx) - img_size/2) * scaling
        return Polygon(poly, reference)

    def get_pointcloud(self, idx):
        pc_path = os.path.join(self.data_dir, "pcs", self.split, f"{idx:08d}.npy")
        pc_bev = np.load(pc_path)
        return pc_bev

    def get_pointcloud_avstack(self, idx, SM):
        metadata = self.get_metadata(idx)
        SD = SM.get_scene_dataset_by_name(metadata["scene"])
        pc = SD.get_lidar(
            frame=metadata["frame"], sensor=metadata["sensor"], agent=metadata["agent"]
        )
        return pc

    def pc_to_img(self, pc):
        img = point_cloud_to_image(
            pc=pc,
            max_range=self._max_range,
            extent=self._extent,
            img_size=self._img_size,
            max_bin=255,
            do_preprocess=True,
        )
        return img

    def img_to_pts(self, img: np.ndarray, metadata: dict, threshold=0.7) -> np.ndarray:
        # get the indices of points
        closed_image_bin = fill_holes((img > threshold).astype(np.uint8))
        pts_all_scaled = np.vstack([*np.where(closed_image_bin)]).T

        # scale back to points
        img_size = np.array([metadata["imgWidth"], metadata["imgHeight"]])
        scaling = np.array([metadata["objects"][0]["dx"], metadata["objects"][0]["dy"]])
        pts_all = (pts_all_scaled - img_size/2) * scaling

        return pts_all
    
    def img_to_pts_avstack(self, img: np.ndarray, metadata: dict, SM, threshold=0.7) -> np.ndarray:
        pts_all = self.img_to_pts(img=img, metadata=metadata, threshold=threshold)
        SD = SM.get_scene_dataset_by_name(metadata["scene"])
        calib = SD.get_calibration(
            frame=metadata["frame"],
            agent=metadata["agent"],
            sensor=metadata["sensor"],
        )
        pts_avstack = PointMatrix2D(pts_all, calib)
        return pts_avstack