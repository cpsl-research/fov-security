import os
from argparse import ArgumentParser

import numpy as np
import torch
from avstack.config import MODELS, Config
from torch.utils.data import DataLoader
from torchvision import transforms

from fov.segmentation.dataset import BinaryFovDataset
from fov.train import BinarySegmentation


class ToDevice:
    def __init__(self, device):
        self.device = device

    def __call__(self, image: torch.Tensor):
        return image.to(self.device)


def main(args):
    # load the config
    cfg = Config.fromfile(args.config)

    # set the device
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    # device = "cpu"
    print(f"Using device {device}")
    data_dir = cfg["data_output_dir"]
    print(f"Training on data from {data_dir}")

    # use transformations to set the model
    trans = transforms.Compose([
        ToDevice(device=device),
        transforms.Resize(size=cfg["model_io_size"]),
    ])

    # make train and test data loaders
    splits = ["train", "val", "test"]
    datasets = {
        split: BinaryFovDataset(
            data_dir,
            transform=trans,
            transform_mask=trans,
            split=split,
            max_range=cfg["max_range"],
            extent=cfg["extent"],
            img_size=cfg["img_size"],
        )
        for split in splits
    }
    loaders = {
        split: DataLoader(dataset, batch_size=cfg["batch_size"], shuffle=split == "train")
        for split, dataset in datasets.items()
    }

    # instantiate a new model
    model = MODELS.build(cfg["model"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])

    # parse the model name
    model_name = args.config.split("/")[-1].replace(".py", "")
    save_folder = os.path.join(args.out_dir, model_name)
    os.makedirs(save_folder, exist_ok=True)
    cfg.dump(os.path.join(save_folder, model_name + ".py"))
    print(f"Saving training results to {save_folder}")

    # set up train/test infrastructure
    infrastructure = BinarySegmentation(
        model=model,
        save_folder=save_folder,
        optimizer=optimizer,
        train_loader=loaders["train"],
        val_loader=loaders["val"],
        test_loader=loaders["test"],
    )

    # run the training
    infrastructure.train(
        epochs=cfg["epochs"],
        early_stopping=cfg["early_stopping"],
        early_stopping_frac=cfg["early_stopping_frac"],
        val_freq=cfg["val_freq"],
        i_metrics_iter=np.inf,
        max_batches=cfg["max_batches"],
    )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", type=str)
    parser.add_argument("--out_dir", type=str, default="segmentation_training")
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    main(args)
