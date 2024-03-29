# miner/trainers/partial_ner_trainer.py

import logging
from typing import Optional, Dict, Literal

import wandb
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import SGD
from seqeval.scheme import IOB2
from seqeval.metrics import f1_score, precision_score, recall_score

from miner.optimizer import SAM
from miner.modules import PartialNER
from miner.utils import LRScheduler


class PartialNERTrainer():
    """Partial Named entity recognizer's trainer. Train and validate on given
    train and validation datasets. Learning is automatically shrunk after each
    epoch by 40%. The model with the greatest validation F-score is saved at
    the end of each epoch.

    Parameters
    ----------
    ner: miner.modules.PartialNER
        Partial Named entity recognizer module to train.
    lr: float
        Initial learning rate.
    momentum: float
        Gradient momentum.
    epochs: int
        Maximum number of training epochs.
    device: str, {"cpu", "cuda"}
        Wether or not to use the GPU for computation.
    ner_path: str
        Path to the lcoal file containing the trained model
    idx2label: Dict[int, str]
        Maps the unique integer ids to their string label.
    loss_fn: str, optional
        The loss function to choose between {"nll", "c_nll", "gce"}.
    clip: float
        Gradient clipping norm value.
    sam: bool
        If you want to use Google's Sharpness Aware Minimization [1]_ or not.

    Attributes
    ----------
    ner: miner.modules.PartialNER
        Partial Named entity recognizer module to train.
    epochs: int
        Maximum number of training epochs.
    device: str
        Wether or not to use the GPU for computation.
    ner_path: str
        Path to the lcoal file containing the trained model
    idx2label: Dict[int, str]
        Maps the unique integer ids to their string label.
    loss_fn: str, default="nll"
        The loss function to choose between {"nll", "c_nll", "gce"}
    clip: float
        Gradient clipping norm value.
    sam: bool
        If you want to use Google's Sharpness Aware Minimization [1]_ or not.
    optimizer: Union[SAM, SGP]
        Optimizing algorithm. If ``sam`` is set to true, ``self.optimizer``
        will be an instance of ``SAM``.
    lrs: miner.utils.LRScheduler
        Learning rate scheduler.

    References
    ----------
    ..  [1] Foret, P., Kleiner, A., Mobahi, H., & Neyshabur, B. (2020).
            Sharpness-aware minimization for efficiently improving
            generalization. arXiv preprint arXiv:2010.01412.
    """

    def __init__(
        self, ner: PartialNER, lr: float, epochs: int,
        device: Literal["cpu", "cuda"], ner_path: str,
        momentum: float, sam: bool, idx2label: Dict[int, str], clip: float,
        loss_fn: Optional[str]=None
    ):
        self.sam = sam
        self.path = ner_path
        self.device = device
        self.ner = ner
        self.epochs = epochs
        self.clip = clip
        self.idx2label = idx2label
        self.loss_fn = "nll" if loss_fn is None else loss_fn
        if sam:
            self.optimizer= SAM(
                ner.parameters(),
                SGD,
                lr=lr,
                momentum=momentum
            )
        else:
            self.optimizer = SGD(
                ner.parameters(),
                lr=lr,
                momentum=momentum
            )
        self.lrs = LRScheduler(
            optimizer=self.optimizer,
            patience=1,
            factor=0.6
        )

    def _fit(self, train_dataloader: DataLoader):
        logging.info("Training...")
        self.ner.train()
        losses = []
        for x, x_augmented, y in tqdm(train_dataloader):
            self.optimizer.zero_grad()
            x = {key: val.squeeze(1) for key, val in x.items()}
            x_augmented = {key: val.squeeze(1) for key, val in x_augmented.items()}
            loss = self.ner(x, x_augmented, y, loss_fn=self.loss_fn)
            losses.append(loss.item())
            loss.backward()
            if self.sam:
                self.optimizer.first_step(zero_grad=True)
                loss = self.ner(x, x_augmented, y, loss_fn=self.loss_fn).backward()
                if self.clip is not None:
                    nn.utils.clip_grad_norm_(
                        self.ner.parameters(),
                        self.clip
                    )
                self.optimizer.second_step(zero_grad=True)
            else:
                self.optimizer.step()
        return np.mean(losses)

    @torch.no_grad()
    def _validate(self, val_dataloader: DataLoader):
        logging.info("Validating...")
        self.ner.eval()
        y_true = []
        y_pred = []
        for x, _, y in tqdm(val_dataloader):
            x = {key: val.squeeze(1) for key, val in x.items()}
            result = self.ner.viterbi_decode(x)
            y_pred.extend(result)
            y_true.extend(y.tolist())
        for i, y in enumerate(y_pred):
            for j, _ in enumerate(y):
                y_pred[i][j] = self.idx2label[y_pred[i][j]]
                y_true[i][j] = self.idx2label[y_true[i][j]]
            y_true[i] = y_true[i][:len(y_pred[i])]
        metrics = {
            "f1": f1_score(y_true, y_pred, mode="strict", scheme=IOB2),
            "p": precision_score(y_true, y_pred, mode="strict", scheme=IOB2),
            "r": recall_score(y_true, y_pred, mode="strict", scheme=IOB2)
        }
        return metrics

    def train(
        self, train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader]=None, wandb_: bool=False
    ):
        """Trains the partial named entity recognizer and saves the one with
        highest validation F-score or the last one at the end of each epoch.

        Parameters
        ----------
        train_dataloader: torch.utils.data.DataLoader
            Iterable object used for training.
        val_dataloader: torch.utils.data.DataLoader, optional
            Iterable object used for validation.
        wandb_: bool, default=False
            If you want to log the training data on **Weigh and Biases** or
            not.
        """
        best_loss = 1e100
        train_loss = 0.0
        best_f1 = 0.0
        f1 = 0.0
        try:
            for epoch in tqdm(range(self.epochs)):
                lr = self.lrs.optimizer.param_groups[0]["lr"]
                train_loss = self._fit(train_dataloader=train_dataloader)
                if val_dataloader is not None:
                    metrics = self._validate(val_dataloader)
                    f1 = metrics["f1"]
                    if wandb_:
                        logs = {"lr": lr, "train_loss": train_loss, **metrics}
                        wandb.log(logs)
                logging.info(
                    f"Epoch: {epoch + 1}\n"
                    + f"LR: {lr}\n"
                    + f"Train Loss: {train_loss}\n"
                    + f"F1: {f1}"
                )
                if val_dataloader is not None and f1 >= best_f1:
                    torch.save({
                        "model_state_dict": self.ner.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "loss": train_loss,
                    }, self.path)
                    best_f1 = f1
                elif train_loss <= best_loss:
                    torch.save({
                        "model_state_dict": self.ner.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "loss": train_loss,
                    }, self.path)
                    best_loss = train_loss
                self.lrs()
        except KeyboardInterrupt:
            logging.warning("Exiting from training early.")
            if train_loss <= best_loss:
                torch.save({
                    "model_state_dict": self.ner.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "loss": train_loss,
                }, self.path)

