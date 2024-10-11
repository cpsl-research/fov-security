import os
from argparse import ArgumentParser

import numpy as np
import torch
from avstack.config import MODELS, Config
from sklearn.model_selection import KFold
from torch.utils.data import ConcatDataset, DataLoader, SubsetRandomSampler
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
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = "cpu"
    print(f"Using device {device}")

    # use transformations to set the model
    trans = transforms.Compose([ToDevice(device=device)])

    # make train and test data loaders
    data_dir = cfg["dataset_path"]
    print(f"Training on data from {data_dir}")
    splits = ["train", "val", "test"]
    datasets = {
        split: BinaryFovDataset(
            data_dir, transform=trans, transform_mask=trans, split=split
        )
        for split in splits
    }
    dataset_concat = ConcatDataset([datasets["train"], datasets["val"]])

    # split using k-fold cross validation
    batch_size_map = {8: 40, 16: 25, 32: 15}
    kfold = KFold(n_splits=args.k_folds, shuffle=True)
    for fold, (train_ids, val_ids) in enumerate(kfold.split(dataset_concat)):
        print(f"Running fold {fold}")

        # subsample data for k-fold
        train_subsampler = SubsetRandomSampler(train_ids)
        val_subsampler = SubsetRandomSampler(val_ids)

        # nest the cross validation over some parameters
        for dr in [0.05, 0.10, 0.15]:
            for lr in [0.0001, 0.001, 0.01]:
                for flc in [8, 16, 32]:

                    # set the batch size based on the flc layers
                    train_loader = DataLoader(
                        dataset_concat,
                        batch_size=batch_size_map[flc],
                        sampler=train_subsampler,
                    )
                    val_loader = DataLoader(
                        dataset_concat,
                        batch_size=batch_size_map[flc],
                        sampler=val_subsampler,
                    )

                    # ----------------------------------------
                    # set the parameter we want to study
                    # ----------------------------------------

                    cfg["model"]["first_layer_channels"] = flc
                    cfg["model"]["p_dropout"] = dr

                    # instantiate a new model
                    model = MODELS.build(cfg["model"]).to(device)
                    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

                    # parse the model name
                    model_name = args.config.split("/")[-1].replace(".py", "")
                    save_folder = os.path.join(
                        args.out_dir,
                        f"{model_name}-flc-{flc}-dr-{dr}-lr-{lr}-fold-{fold}",
                    )
                    print(f"Saving training results to {save_folder}")

                    # set up train/test infrastructure
                    infrastructure = BinarySegmentation(
                        model=model,
                        save_folder=save_folder,
                        optimizer=optimizer,
                        train_loader=train_loader,
                        val_loader=val_loader,
                        test_loader=None,
                    )

                    # run the training
                    infrastructure.train(
                        epochs=100,
                        early_stopping=4,
                        early_stopping_frac=0.02,
                        val_freq=1,
                        i_metrics_iter=np.inf,
                    )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--config", type=str, default="../config/segmentation/unet_mc.py"
    )
    parser.add_argument(
        "--k-folds",
        type=int,
        default=5,
    )
    parser.add_argument("--out_dir", type=str, default="segmentation_training")
    args = parser.parse_args()
    main(args)
