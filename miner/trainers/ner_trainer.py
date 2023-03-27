# miner/trainers/ner_trainer.py

import logging
from typing import Literal

import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import SGD, lr_scheduler

from miner.modules import NER


class EarlyStopping():
    """Early stoppping to stop the training when the loss does not improve
    after certain epochs.

    Parameters
    ----------
    patience: ``int``
        How many epochs to wait before stopping when loss is not improving.
    min_delta: ``int``
        Minimum difference between new loss and old loss for new loss to be
        considered as an improvement.

    Attributes
    ----------
    patience: ``int``
        How many epochs to wait before stopping when loss is not improving.
    min_delta: ``int``
        Minimum difference between new loss and old loss for new loss to be
        considered as an improvement.
    counter: ``int``
        Number of epochs without any loss improvement, to wait before early
        stopping.
    best_loss: ``float``
        Last iteration loss.
    early_stop: ``bool``
        Set to ``True`` if the loss is not improving above `min_delta value
        after `patience` epochs.

    References
    ----------
    ..  Sovit Ranjan RathSovit Ranjan Rath et al. 2021. Using learning rate
        scheduler and early stopping with pytorch. (October 2021). Retrieved
        November 21, 2022 from
        https://debuggercafe.com/using-learning-rate-scheduler-and-early-stopping-with-pytorch/
    """

    def __init__(self, patience: int, min_delta: float):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, loss: float):
        if self.best_loss is None:
            self.best_loss = loss
        elif self.best_loss - loss > self.min_delta:
            self.counter = 0
            self.best_loss = min(self.best_loss, loss)
        elif self.best_loss - loss <= self.min_delta:
            self.best_loss = min(self.best_loss, loss)
            self.counter +=1
            logging.info(
                f"Early stoppping counter {self.counter} of {self.patience}."
            )
            if self.counter >= self.patience:
                logging.info("Early stopping.")
                self.early_stop = True


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
    optimizer: ``torch.optim.SGD``
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

    def __init__(self, optimizer: SGD, patience: int, factor: float):
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer,
            mode="min",
            factor=factor,
            patience=patience,
            threshold=1.0,
            verbose=False
        )

    def __call__(self, loss: float):
        self.lr_scheduler.step(loss)


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
        self, ner: NER, lr: float, momentum: float, patience: int,
        min_delta: float, epochs: int, max_length: int,
        device: Literal["cpu", "cuda"], accumulation_steps: int, ner_path: str
    ):
        self.path = ner_path
        self.device = device
        self.ner = ner
        self.epochs = epochs
        self.accumulation_steps = accumulation_steps
        self.max_length = max_length
        self.optimizer = SGD(
            ner.parameters(),
            lr=lr,
            momentum=momentum
        )
        self.lrs = LRScheduler(
            optimizer=self.optimizer,
            patience=patience,
            factor=0.4
        )
        self.early_stopping = EarlyStopping(
            patience=patience,
            min_delta=min_delta
        )

    def _fit(self, train_dataloader: DataLoader):
        logging.info("Training...")
        self.ner.train()
        losses = []
        idx = 0
        for x, y in tqdm(train_dataloader):
            self.ner.zero_grad()
            loss = self.ner(x, y) / self.accumulation_steps
            losses.append(loss.item())
            loss.backward()
            if ((idx + 1) % self.accumulation_steps == 0) \
                or ((idx + 1) == len(train_dataloader)):
                nn.utils.clip_grad_norm_(self.ner.parameters(), 5.0)    # type: ignore
                self.optimizer.step()
            idx += 1
        val_loss = np.mean(losses)
        return val_loss

    def _validate(self, val_dataloader: DataLoader):
        logging.info("Validating...")
        torch.set_printoptions(profile="full")
        self.ner.eval()
        losses = []
        for x, y in tqdm(val_dataloader):
            loss = self.ner(x, y)
            losses.append(loss.item())
        val_loss = np.mean(losses)
        return val_loss

    def train(self, train_dataloader: DataLoader, val_dataloader: DataLoader):
        """Trains the named entity recognizer and saves the best one at the end
        of each epoch.

        Parameters
        ----------
        train_dataloader: ``torch.utils.data.DataLoader``
            Iterable object used for training.
        val_dataloader: ``torch.utils.data.DataLoader``
            Iterable object used for validation.
        """
        best_loss = 1e100
        train_loss = 0.0
        val_loss = 0.0
        try:
            for epoch in tqdm(range(self.epochs)):
                train_loss = self._fit(train_dataloader=train_dataloader,)
                val_loss = self._validate(val_dataloader=val_dataloader)
                self.lrs(val_loss)              # type: ignore
                self.early_stopping(train_loss) # type: ignore
                if self.early_stopping.early_stop:
                    break
                logging.info(
                    f"Epoch: {epoch + 1}\n"
                    + f"LR: {self.lrs.optimizer.param_groups[0]['lr']}\n"
                    + f"Train Loss: {train_loss}\n"
                    + f"Val Loss: {val_loss}"
                )
                if val_loss <= best_loss:
                    torch.save({
                        "model_state_dict": self.ner.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "loss": train_loss,
                    }, self.path)
                    best_loss = val_loss
        except KeyboardInterrupt:
            logging.warning("Exiting from training early.")
            if val_loss >= best_loss:
                torch.save({
                    "model_state_dict": self.ner.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "loss": train_loss,
                }, self.path)

    def save(self, path: str):
        """Saves the model to designated path.

        Parameters
        ----------
        path: ``str``
            Path to the locally saved model.
        """
        logging.info(f"Saving the NER model to {path}.")
        torch.save(self.ner, path)

