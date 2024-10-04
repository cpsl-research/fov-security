import argparse

from fov.graph.dataset import CarlaFieldOfViewDataset


def main(args):
    dataset = CarlaFieldOfViewDataset(
        carla_root_directory=args.data_dir_input,
        graph_root_directory=args.data_dir_output,
        include_infrastructure_agents=False,
        n_frames_max=100,
        force_reload=args.force_reload,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir_input", type=str, default="/data/shared/CARLA/multi-agent-v1"
    )
    parser.add_argument("--data_dir_output", type=str, default="/data/shared/fov")
    parser.add_argument("--sensor", type=str, default="lidar-0")
    parser.add_argument("--force_reload", action="store_true", default=False)
    args = parser.parse_args()

    main(args)
