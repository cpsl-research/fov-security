import os
from datetime import datetime
from time import time
from typing import TYPE_CHECKING, Callable, Dict, Optional

from tqdm import tqdm


if TYPE_CHECKING:
    from torch.nn import Module
    from torch.utils.data import DataLoader
    from torch.optim import Optimizer
    from torch.nn.modules.loss import _Loss

import torch
import torcheval.metrics.functional as torch_metrics
from torch.utils.tensorboard import SummaryWriter


class TrainTestInfrastructure:
    def __init__(
        self,
        model: "Module",
        save_folder: str,
        train_loader: Optional["DataLoader"] = None,
        val_loader: Optional["DataLoader"] = None,
        test_loader: Optional["DataLoader"] = None,
        optimizer: Optional["Optimizer"] = None,
    ):
        """Set up a training infrastructure"""
        self.model = model
        self.save_folder = save_folder
        self.optimizer = optimizer
        self.criterion = self.get_criterion()
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self._n_epochs_coast = 0

        os.makedirs(save_folder, exist_ok=True)

    @staticmethod
    def parse_data_inputs_labels(data):
        raise NotImplementedError()

    def train(
        self,
        epochs: int,
        early_stopping: int = 5,
        early_stopping_frac: float = 0.05,
        val_freq: int = 5,
        *args,
        **kwargs,
    ):
        """Run training process"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_folder = os.path.join(self.save_folder, "model_{}".format(timestamp))
        os.makedirs(out_folder)
        print("Starting training process...saving output to {}".format(out_folder))

        writer = SummaryWriter(f"{out_folder}/log")
        best_vloss = 1_000_000.0
        last_losses = []

        for epoch in range(epochs):
            print("EPOCH {}:".format(epoch))

            # make a pass over the data with training on
            self.model.train()
            avg_loss = self._train_one_epoch(
                epoch=epoch, writer=writer, *args, **kwargs
            )

            # write to tensorboard
            writer.add_scalar("Loss/train", avg_loss, epoch)

            if (((epoch + 1) % val_freq) == 0) and (self.val_loader is not None):
                running_vloss = 0.0
                # Set the model to evaluation mode, disabling dropout and using population
                # statistics for batch normalization.
                self.model.eval()

                # Disable gradient computation and reduce memory consumption.
                running_vtime = 0.0
                all_vlabels = []
                all_voutputs = []
                print("  beginning validation processing...")
                with torch.no_grad():
                    for i, vdata in tqdm(
                        enumerate(self.val_loader), total=len(self.val_loader)
                    ):
                        vinputs, vlabels = self.parse_data_inputs_labels(vdata)
                        starttime = time()
                        voutputs = self.model(vinputs)
                        endtime = time()
                        running_vtime += (endtime - starttime) / vinputs.shape[
                            0
                        ]  # divide by batch size
                        vloss = self.loss(voutputs, vlabels)
                        running_vloss += vloss.detach().item()
                        all_vlabels.append(vlabels.detach())
                        all_voutputs.append(voutputs.detach())

                # print the loss to console
                avg_vloss = running_vloss / (i + 1)
                avg_vtime = running_vtime / len(self.val_loader)
                print("LOSS train {} valid {}".format(avg_loss, avg_vloss))

                # run the metrics on the output
                vmetrics = self.metrics(
                    torch.cat(all_voutputs, dim=0),
                    torch.cat(all_vlabels, dim=0),
                    run_auprc=True,
                    loss_fn=None,
                )

                # Log the running loss averaged per batch
                # for both training and validation
                writer.add_scalar("Loss/validation", avg_vloss, epoch)
                writer.add_scalar("Metrics/inference_time", avg_vtime, epoch)
                for k, v in vmetrics.items():
                    if v is None:
                        continue
                    else:
                        try:
                            writer.add_scalar(f"Metrics/{k}", v, epoch)
                        except AssertionError:
                            pass  # don't know how to add vector data...

                # Track best performance, and save the model's state
                if avg_vloss < best_vloss:
                    best_vloss = avg_vloss
                    model_path = os.path.join(out_folder, "epoch_{}.pth".format(epoch))
                    print("saving model at epoch {}".format(epoch))
                    torch.save(self.model.state_dict(), model_path)

                # keep track of early stopping
                if len(last_losses) < early_stopping:
                    last_losses.append(avg_vloss)
                else:
                    last_losses.pop(0)
                    last_losses.append(avg_loss)
                    pct_change = (last_losses[0] - last_losses[-1]) / last_losses[0]
                    if pct_change < early_stopping_frac:
                        print(
                            f"Stopping early after minimal ({pct_change}) "
                            f"improvement in {early_stopping} iterations "
                            f"versus threshold ({early_stopping_frac})"
                        )
                        break

            writer.flush()

    def _train_one_epoch(
        self,
        epoch: int,
        writer: Optional[SummaryWriter] = None,
        i_metrics_iter: int = 10,
        i_write_iter: int = 10,
        *args,
        **kwargs,
    ):
        """Run one epoch of training"""
        i_write_iter = min(i_write_iter, len(self.train_loader))
        running_loss = 0.0
        last_loss = 0.0
        _last_metrics = {}

        for i, data in enumerate(self.train_loader):
            # print(torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated())
            # breakpoint()

            # zero gradients every batch
            self.optimizer.zero_grad()

            # execute model on batch
            run_metrics = ((i + 1) % i_metrics_iter) == 0
            loss, metrics = self._train_one_batch(
                data, run_metrics=run_metrics, *args, **kwargs
            )
            if run_metrics:
                _last_metrics = metrics

            # write things
            running_loss += loss.detach().item()
            if (writer is not None) and ((i + 1) % i_write_iter == 0):
                last_loss = running_loss / i_write_iter  # loss per batch
                print("  batch {} loss: {}".format(i + 1, last_loss))

                # -- add loss to writer
                tb_x = epoch * len(self.train_loader) + i + 1
                writer.add_scalar("Loss/train", last_loss, tb_x)
                running_loss = 0.0

                # -- add metrics to writer
                for k, v in _last_metrics.items():
                    writer.add_scalar("Metrics/train/{}".format(k), v, tb_x)

            # backpropagation
            loss.backward()
            self.optimizer.step()

        return last_loss

    def _train_one_batch(self, *args, **kwargs):
        raise NotImplementedError

    @torch.no_grad()
    def test(self, *args, **kwargs):
        self.model.eval()
        for data in self.test_loader:
            metrics = self._test_one_batch(data, *args, **kwargs)

    def _test_one_batch(self):
        raise NotImplementedError

    def loss(self, outputs, labels):
        """Can be overriden in subclass if needed"""
        return self.criterion(outputs, labels)

    def metrics(self, outputs, labels):
        raise NotImplementedError


class _BinaryInfrastructure(TrainTestInfrastructure):
    def get_criterion(self) -> "_Loss":
        return torch.nn.BCELoss(reduction="none")

    def loss(self, outputs, labels, pos_weight: float = 1.0, neg_weight: float = 1.0):
        """Evaluate the loss function with weighting"""
        intermediate_loss = self.criterion(outputs, labels.float())
        w_y_label = labels * pos_weight + ~labels * neg_weight
        loss = torch.mean(w_y_label * intermediate_loss)
        return loss

    def _train_one_batch(
        self,
        data,
        pos_weight: float = 1.0,
        neg_weight: float = 1.0,
        run_metrics: bool = False,
        *args,
        **kwargs,
    ):
        """Run model in training mode on one batch of data"""
        # run model and get losses
        inputs, labels = self.parse_data_inputs_labels(data)
        outputs = self.model(inputs)
        loss = self.loss(outputs, labels, pos_weight=pos_weight, neg_weight=neg_weight)
        if run_metrics:
            metrics = self.metrics(
                outputs=outputs,
                labels=labels,
                loss_fn=self.loss,
                loss=loss,
            )
        else:
            metrics = {}
        return loss, metrics

    @staticmethod
    @torch.no_grad()
    def metrics(
        outputs,
        labels,
        loss_fn: Optional[Callable] = None,
        loss: Optional[float] = None,
        threshold: float = 0.7,
        pos_weight: float = 1.0,
        neg_weight: float = 1.0,
        run_auprc: bool = False,
        thresholds_test=torch.linspace(0.5, 1.0, steps=10),
    ) -> Dict:
        """Get metrics on the results of one batch"""
        # return the metrics
        out_view = outputs.view(-1)
        lab_view = labels.view(-1)
        return {
            "loss": loss_fn(
                outputs, labels, pos_weight=pos_weight, neg_weight=neg_weight
            )
            if (loss is None) and (loss_fn is not None)
            else loss,
            "n_total": len(lab_view),
            "n_positive": lab_view.sum(),
            "auprc": torch_metrics.binary_auprc(
                input=out_view,
                target=lab_view,
            )
            if run_auprc
            else None,
            "recall": torch_metrics.binary_recall(
                input=out_view,
                target=lab_view,
                threshold=threshold,
            ),
            "precision": torch_metrics.binary_precision(
                input=out_view,
                target=lab_view,
                threshold=threshold,
            ),
            "f1": torch_metrics.binary_f1_score(
                input=out_view,
                target=lab_view,
                threshold=threshold,
            ),
            "accuracy": torch_metrics.binary_accuracy(
                input=out_view,
                target=lab_view,
                threshold=threshold,
            ),
            "f1-by-threshold": torch.tensor(
                [
                    torch_metrics.binary_f1_score(
                        input=out_view, target=lab_view, threshold=thresh
                    )
                    for thresh in thresholds_test
                ]
            ),
            "thresholds": thresholds_test,
        }


class BinarySegmentation(_BinaryInfrastructure):
    @staticmethod
    def parse_data_inputs_labels(data):
        image, mask = data
        return image, mask


class BinaryGraphClassification(_BinaryInfrastructure):
    @staticmethod
    def parse_data_inputs_labels(data):
        return data, data.y
