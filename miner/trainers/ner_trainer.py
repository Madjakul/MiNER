# miner/trainers/ner_trainer.py

import logging
from typing import Optional, Dict, Literal, Union

import wandb
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import SGD, lr_scheduler
from seqeval.scheme import IOB2
from seqeval.metrics import f1_score, precision_score, recall_score

from miner.modules import NER
from miner.optimizer import SAM


class LRScheduler():
    """Learning rate scheduler. If the validation loss does not decrease for
    the given number of `patience` epochs, then the learning rate will decrease
    by a given `factor`.

    Parameters
    ----------
    optimizer: ``torch.optim.SGD``
        The optimizer we are using.
    patience: ``int``
        How many epochs to wait before updating the learning rate.
    factor: ``float``
        Factor by which the learning rate should be updated.

    Attributes
    ----------
    optimizer: ``miner.optimizer.SAM``
        The optimizer we are using.
    patience: ``int``
        How many epochs to wait before updating the learning rate.
    min_lr: ``float``
        Last learning rate value to reduce to while updating.
    factor: ``float``
        Factor by which the learning rate should be updated.
    lr_scheduler: ``torch.optim.lr_scheduler.ReduceLROnPlateau``

    References
    ----------
    ..  Sovit Ranjan RathSovit Ranjan Rath et al. 2021. Using learning rate
        scheduler and early stopping with pytorch. (October 2021). Retrieved
        November 21, 2022 from
        https://debuggercafe.com/using-learning-rate-scheduler-and-early-stopping-with-pytorch/
    """

    def __init__(self, optimizer: Union[SGD, SAM], patience: int, factor: float):
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler.StepLR(
            optimizer=optimizer,
            step_size=patience,
            gamma=factor,
        )

    def __call__(self):
        self.lr_scheduler.step()


class NER_Trainer():
    """Named entity recognizer's trainer. Train and validate on given train and
    validation datasets. Learning is automatically reduced upon hitting a
    plateau. Stops the training early if needed. The model with the lowest
    validation loss is saved at the end of each epoch.

    Parameters
    ----------
    ner: ``miner.modules.NER``
        Named entity recognizer module to train.
    lr: ``float``
        Initial leanring rate.
    momentum: ``float``
        Gradient momentum.
    patience: ``int``
        Number of steps without improvement of the training loss before
        stopping the training. Number of steps without improvement of the
        validation loss before reducing the learning rate.
    epochs: ``int``
        Maximum number of training epochs.
    max_length: ``int``
        Maximum sequence length.
    device: ``str``, {"cpu", "cuda"}
        Wether or not to use the GPU for computation.
    accumulation_steps: ``int``
        Number of steps during the gradient is accumulated.
    ner_path: ``str``
        Path to the lcoal file containing the trained model

    Attributes
    ----------
    ner: ``miner.modules.NER``
        Named entity recognizer module to train.
    epochs: ``int``
        Maximum number of training epochs.
    max_length: ``int``
        Maximum sequence length.
    device: ``str``, {"cpu", "cuda"}
        Wether or not to use the GPU for computation.
    accumulation_steps: ``int``
        Number of steps during the gradient is accumulated.
    path: ``str``
        Path to the lcoal file containing the trained model
    optimizer: ``torch.optim.SGD``
        Stochastic gradient descent optimizer.
    lrs: ``miner.trainers.ner_trainer.LRScheduler``
        Object of the learning rate scheduler class.
    early_stopping: ``miner.trainers.ner_trainer.EarlyStopping``
        Object of the early stopper class.
    """

    def __init__(
        self, ner: NER, lr: float, patience: int, epochs: int, max_length: int,
        device: Literal["cpu", "cuda"], ner_path: str,
        momentum: float,sam: bool, idx2label: Dict[int, str], clip: float,
        loss_fn: Optional[str]=None
    ):
        self.sam = sam
        self.path = ner_path
        self.device = device
        self.ner = ner
        self.epochs = epochs
        self.max_length = max_length
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
            patience=patience,
            factor=0.6
        )

    def _fit(self, train_dataloader: DataLoader):
        logging.info("Training...")
        self.ner.train()
        losses = []
        for x, y, _ in tqdm(train_dataloader):
            self.optimizer.zero_grad()
            x = {key: val.squeeze(1) for key, val in x.items()}
            loss = self.ner(x, y, loss_fn=self.loss_fn)
            losses.append(loss.item())
            loss.backward()
            if self.sam:
                self.optimizer.first_step(zero_grad=True)
                loss = self.ner(x, y, loss_fn=self.loss_fn).backward()
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
        torch.set_printoptions(profile="full")
        self.ner.eval()
        y_true = []
        y_pred = []
        for x, y, _ in tqdm(val_dataloader):
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
        """Trains the named entity recognizer and saves the best one at the end
        of each epoch.

        Parameters
        ----------
        train_dataloader: ``torch.utils.data.DataLoader``
            Iterable object used for training.
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

