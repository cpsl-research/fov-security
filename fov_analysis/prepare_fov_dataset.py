import argparse

from avapi.carla import CarlaScenesManager
from avstack.modules.perception.fov_estimator import RayTraceBevLidarFovEstimator


def main(args):
    fov_estimator = RayTraceBevLidarFovEstimator(z_min=-3.0, z_max=3.0)
    for split in ["train", "val"]:
        CSM = CarlaScenesManager(data_dir=args.data_dir_input, split=split)
        for CDM in CSM:
            for frame in CDM.frames:
                for agent_ID in CDM.get_agent_set(frame=frame):
                    try:
                        # get pc from agent
                        pc = CDM.get_lidar(frame=frame, sensor=args.sensor, agent=agent_ID)
                    except KeyError:  # not all agents have lidar data
                        continue

                    # run ray trace on bev to attempt ground truth point cloud
                    fov = fov_estimator(pc=pc)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir_input", type=str, default="/data/shared/CARLA/multi-agent-v1")
    parser.add_argument("--data_dir_output", type=str, default="/data/shared/fov/")
    parser.add_argument("--sensor", type=str, default="lidar-0")
    args = parser.parse_args()

    main(args)