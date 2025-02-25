import os
from argparse import ArgumentParser
from glob import glob

import cv2
import numpy as np
from PIL import Image

from fov.segmentation.dataset import fov_bev_pallete_matrix


def main(args):

    # set paths
    gt_dir = os.path.join(args.data_dir, "ann", args.split)
    img_files = glob(os.path.join(args.data_dir, "img", args.split, "*.png"))

    # loop over the image files
    for img_file in sorted(img_files):
        # get the image index
        img_idx = img_file.split("/")[-1].replace(".png", "")

        # get the gt files
        gt_poly_file = os.path.join(gt_dir, f"{img_idx}_polygons.json")
        gt_img_file = os.path.join(gt_dir, f"{img_idx}_segimage.png")

        # load in the original image and map to a palette color
        img_pc = np.array(Image.open(img_file))
        img_colors = np.zeros((*img_pc.shape, 3), dtype=np.uint8)
        img_colors[img_pc > 1, :] = np.array([255, 255, 255], dtype=np.uint8)

        # load in the gt seg image and map the class to a palette color
        img_gtseg_classes = np.array(Image.open(gt_img_file))
        classes = img_gtseg_classes.ravel()
        img_gtseg_colors = fov_bev_pallete_matrix[classes].reshape(
            [*img_gtseg_classes.shape, 3]
        )

        # concatenate the image
        img_show = np.concatenate((img_colors, img_gtseg_colors), axis=1)

        # show the gt image
        print(f"Showing image {img_idx}")
        cv2.imshow("press c to continue, q to exit", img_show)
        try:
            while True:
                if cv2.waitKey(1) & 0xFF == ord("c"):
                    break
                elif cv2.waitKey(1) & 0xFF == ord("q"):
                    raise SystemExit
        except SystemExit:
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir", type=str, default="/data/shared/fov/fov_bev_segmentation"
    )
    parser.add_argument("--split", type=str, default="train")
    args = parser.parse_args()
    main(args)
