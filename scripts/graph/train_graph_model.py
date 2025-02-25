from argparse import ArgumentParser

import torch
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import ToDevice

from fov.graph.dataset import CarlaFieldOfViewDataset
from fov.graph.models import GATModel
from fov.train import BinaryGraphClassification


def main(args):
    # set the device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device {device}")

    # get the dataset
    full_dataset = CarlaFieldOfViewDataset(
        carla_root_directory=args.data_input,
        graph_root_directory=args.data_output,
        include_infrastructure_agents=False,
        n_frames_max=1000,
        force_reload=False,
        transform=ToDevice(device=device),
    )

    # make train and test data loaders
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, (0.5, 0.3, 0.2)
    )
    train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=10)
    test_loader = DataLoader(test_dataset, batch_size=10)

    # instantiate a new model
    model = GATModel(
        in_channels=-1,
        hidden_channels=256,
        num_layers=5,
        out_channels=1,
        v2=True,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # set up train/test infrastructure
    infrastructure = BinaryGraphClassification(
        model=model,
        save_folder=args.log_dir,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
    )

    # run the training
    infrastructure.train(epochs=10, val_freq=1)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--data_input", type=str, default="/data/shared/CARLA/multi-agent-v1"
    )
    parser.add_argument(
        "--data_output", type=str, default="/data/shared/fov/fov_bev_graph"
    )
    parser.add_argument("--log_dir", type=str, default="models/graph_training")
    args = parser.parse_args()
    main(args)
