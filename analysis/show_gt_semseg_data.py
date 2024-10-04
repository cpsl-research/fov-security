import os.path as osp

import mmcv
import torch
from mmengine.structures import PixelData
from mmseg.structures import SegDataSample
from mmseg.visualization import SegLocalVisualizer


# `PixelData` is data structure for pixel-level annotations or predictions defined in MMEngine.
# Please refer to below tutorial file of data structures in MMEngine:
# https://github.com/open-mmlab/mmengine/tree/main/docs/en/advanced_tutorials/data_element.md


# `SegDataSample` is data structure interface between different components
# defined in MMSegmentation, it includes ground truth, prediction and
# predicted logits of semantic segmentation.
# Please refer to below tutorial file of `SegDataSample` for more details:
# https://github.com/open-mmlab/mmsegmentation/blob/1.x/docs/en/advanced_guides/structures.md


out_file = "out_file_cityscapes"
save_dir = "./work_dirs"
data_dir = "/data/shared/fov/fov_bev_segmentation/"

image = mmcv.imread(osp.join(data_dir, "img", "train", "00000489.png"), "grayscale")
sem_seg = mmcv.imread(
    osp.join(data_dir, "ann", "train", "00000489_segimage.png"), "unchanged"
)
sem_seg = torch.from_numpy(sem_seg)
gt_sem_seg_data = dict(data=sem_seg)
gt_sem_seg = PixelData(**gt_sem_seg_data)
data_sample = SegDataSample()
data_sample.gt_sem_seg = gt_sem_seg

seg_local_visualizer = SegLocalVisualizer(
    vis_backends=[dict(type="LocalVisBackend")], save_dir=save_dir
)

# The meta information of dataset usually includes `classes` for class names and
# `palette` for visualization color of each foreground.
# All class names and palettes are defined in the file:
# https://github.com/open-mmlab/mmsegmentation/blob/1.x/mmseg/utils/class_names.py

# seg_local_visualizer.dataset_meta = dict(
#     classes=('road', 'sidewalk', 'building', 'wall', 'fence',
#              'pole', 'traffic light', 'traffic sign',
#              'vegetation', 'terrain', 'sky', 'person', 'rider',
#              'car', 'truck', 'bus', 'train', 'motorcycle',
#              'bicycle'),
#     palette=[[128, 64, 128], [244, 35, 232], [70, 70, 70],
#              [102, 102, 156], [190, 153, 153], [153, 153, 153],
#              [250, 170, 30], [220, 220, 0], [107, 142, 35],
#              [152, 251, 152], [70, 130, 180], [220, 20, 60],
#              [255, 0, 0], [0, 0, 142], [0, 0, 70],
#              [0, 60, 100], [0, 80, 100], [0, 0, 230],
#              [119, 11, 32]])
# When `show=True`, the results would be shown directly,
# else if `show=False`, the results would be saved in local directory folder.
seg_local_visualizer.add_datasample(out_file, image, data_sample, show=False)
