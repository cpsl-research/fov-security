import json
import os
import shutil
from argparse import ArgumentParser
from typing import List, Tuple

import numpy as np
from avapi.carla import CarlaScenesManager
from avstack.sensors import LidarData
from PIL import Image
from tqdm import tqdm

from fov.graph.preprocess import get_node_ground_truth_bev_boundary
from fov.preprocess import preprocess_point_cloud_for_bev
from fov.segmentation.preprocess import point_cloud_to_gt_seg, point_cloud_to_image


def apply_adversary(
    rng: np.random.RandomState,
    pc_preproc: LidarData,
    extent: List[Tuple[float, float]] = [(-80, 80), (-80, 80)],
) -> LidarData:
    """Randomly sample an adversary and apply the model"""

    adv_options = ["uniform", "cluster", "none"]
    adv_model = rng.choice(adv_options, p=[0.6, 0.2, 0.2])

    # option 1: random noise throughout the entire point cloud
    if adv_model == "uniform":
        # randomly choose a number of points and locations
        n_pts_adv = rng.randint(low=80, high=120, size=1)[0]
        coords_x = rng.uniform(low=extent[0][0], high=extent[0][1], size=(n_pts_adv, 1))
        coords_y = rng.uniform(low=extent[1][0], high=extent[1][1], size=(n_pts_adv, 1))
        intensity = rng.uniform(low=0.80, high=1.0, size=(n_pts_adv, 1))
        coords_add = np.concatenate((coords_x, coords_y, intensity), axis=1)

        # add the points to the point cloud
        pc_preproc.data.x = np.concatenate((pc_preproc.data.x, coords_add), axis=0)

    # option 2: clusters of points near edges of image
    elif adv_model == "cluster":
        # randomly choose a number of points
        n_pts_adv = rng.randint(low=10, high=160, size=1)[0]

        # choose a cluster centroid
        d_edge = 10
        centroid = rng.uniform(low=40, high=extent[0][1] - d_edge, size=(1, 2))
        centroid *= rng.choice([-1, 1], replace=True, size=(1, 2))

        # randomly choose a spread of the centroid
        d_spread = 3
        coords_add = centroid + rng.uniform(
            low=-d_spread, high=d_spread, size=(n_pts_adv, 2)
        )

        # add intensity to the points
        intensity = np.random.uniform(low=0.8, high=1.0, size=(n_pts_adv, 1))
        coords_add = np.concatenate((coords_add, intensity), axis=1)

        # add the points to the point cloud
        pc_preproc.data.x = np.concatenate((pc_preproc.data.x, coords_add), axis=0)

    # option 3: no attack
    elif adv_model == "none":
        n_pts_adv = 0

    else:
        raise NotImplementedError(adv_model)

    return pc_preproc, adv_model, n_pts_adv


def main(args):
    split_fracs = {
        "train": 0.6,
        "val": 0.2,
        "test": 0.2,
    }
    CSM = CarlaScenesManager(
        data_dir=args.data_input_dir,
        split_fracs=split_fracs,
        seed=1,
    )
    rng = np.random.RandomState(args.seed)

    # print the split distribution
    for split in ["train", "val", "test"]:
        print(f"Dataset has {len(CSM.splits_scenes[split])} scenes in split {split}")

    # set the output directory
    if args.adversarial:
        data_output_dir = args.data_output_dir.rstrip("/") + "_adversarial"
    else:
        data_output_dir = args.data_output_dir

    # remove the entire dataset tree
    print(f"Saving outputs to {data_output_dir}")
    ann_base = os.path.join(data_output_dir, "ann")
    img_base = os.path.join(data_output_dir, "img")
    pcs_base = os.path.join(data_output_dir, "pcs")
    if os.path.exists(data_output_dir):
        shutil.rmtree(data_output_dir)

    # loop over the splits
    for split in ["train", "val", "test"]:
        # make the directory for saving
        ann_folder = os.path.join(ann_base, split)
        img_folder = os.path.join(img_base, split)
        pcs_folder = os.path.join(pcs_base, split)
        for folder in [ann_folder, img_folder, pcs_folder]:
            os.makedirs(folder)

        # loop over all the carla scenes
        print(f"...processing {len(CSM.splits_scenes[split])} scenes for split {split}")
        i_frame = 0
        for scene_name in tqdm(CSM.splits_scenes[split]):
            CDM = CSM.get_scene_dataset_by_name(scene_name)
            for agent in CDM.get_agents(CDM.frames[0]):
                sensor = "lidar-0"
                for frame in CDM.get_frames(sensor=sensor, agent=agent.ID)[
                    :: args.frames_stride
                ]:
                    # get the point cloud
                    if "static" in agent.obj_type:
                        # possibly an infrastructure agent
                        if args.include_infrastructure:
                            raise NotImplementedError
                        else:
                            continue
                    else:
                        # to avoid overfitting, don't do mobile agents with 0 velocity
                        if agent.velocity.norm() < 0.1:
                            continue

                        # ground agent
                        pc = CDM.get_lidar(frame=frame, sensor=sensor, agent=agent.ID)

                    # project the point cloud to bev and center for saving
                    pc_preproc = preprocess_point_cloud_for_bev(pc)

                    # apply adversary model on top of point cloud
                    if args.adversarial:
                        pc_preproc, adv_model, n_pts_adv = apply_adversary(
                            rng, pc_preproc
                        )
                    else:
                        adv_model = "none"
                        n_pts_adv = 0

                    # convert point cloud to image
                    pc_img = point_cloud_to_image(pc_preproc, do_preprocess=False)

                    # save image
                    pimg = Image.fromarray(pc_img)
                    img_file = f"{i_frame:08d}.png"
                    img_path = os.path.join(img_folder, img_file)
                    pimg.save(img_path)

                    # save point cloud projected
                    pc_file = f"{i_frame:08d}.npy"
                    pc_path = os.path.join(pcs_folder, pc_file)
                    np.save(pc_path, pc_preproc.data.x)

                    ##############################################
                    # GET GROUND TRUTH OUTCOMES
                    ##############################################

                    # get the ground truth segmentation mask
                    gt_seg, gt_img = point_cloud_to_gt_seg(pc)

                    # get the ground truth graph
                    pc_bev = preprocess_point_cloud_for_bev(pc)
                    gt_node_class = (
                        get_node_ground_truth_bev_boundary(
                            pc_bev, d_boundary_threshold=0.5
                        )
                        .numpy()
                        .astype(int)
                        .astype(str)
                        .tolist()
                    )

                    ##############################################
                    # PACKAGE UP RESULTS
                    ##############################################

                    # add metadata to the gt seg json
                    metadata = {
                        "scene": CDM.scene,
                        "frame": frame,
                        "agent": agent.ID,
                        "agent_type": agent.obj_type,
                        "agent_velocity": agent.velocity.norm(),
                        "sensor": sensor,
                        "attacked": adv_model != "none",
                        "adv_model": adv_model,
                        "n_pts_adv": int(n_pts_adv),
                    }
                    gt_seg.update(metadata)

                    # save the ground truth files
                    gt_seg_file = f"{i_frame:08d}_polygons.json"
                    gt_img_file = f"{i_frame:08d}_segimage.png"
                    gt_nod_file = f"{i_frame:08d}_nodeclass.txt"
                    for gt, file in zip(
                        [gt_seg, gt_img, gt_node_class],
                        [gt_seg_file, gt_img_file, gt_nod_file],
                    ):
                        gt_path = os.path.join(ann_folder, file)
                        if file.endswith("json"):
                            try:
                                with open(gt_path, "w") as f:
                                    json.dump(gt, f)
                            except TypeError:
                                breakpoint()
                        elif file.endswith("png"):
                            pimg = Image.fromarray(gt)
                            pimg.save(gt_path)
                        elif file.endswith("txt"):
                            with open(gt_path, "w") as f:
                                f.write(", ".join(gt))
                        else:
                            raise NotImplementedError

                    # print status
                    i_frame += 1
                    if (i_frame % 100) == 0:
                        print(f"Processed {i_frame} frames")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--data_input_dir", type=str, default="/data/shared/CARLA/multi-agent-v1"
    )
    parser.add_argument(
        "--data_output_dir", type=str, default="/data/shared/fov/fov_bev_segmentation"
    )
    parser.add_argument("--adversarial", action="store_true")
    parser.add_argument("--frames_stride", type=int, default=5)
    parser.add_argument("--include_infrastructure", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    main(args)
