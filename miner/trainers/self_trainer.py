# miner/trainers/self_trainer.py

import copy
import logging
from typing import Union, Optional, Literal, Dict

import wandb
import torch
import numpy as np
from tqdm import tqdm
from seqeval.scheme import IOB2
from seqeval.metrics import f1_score
from torch.utils.data import DataLoader
from torch.optim import Adam, lr_scheduler

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

    def __init__(self, optimizer: Union[SAM, Adam], epochs: int):
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler.LinearLR(
            optimizer=optimizer,
            start_factor=0.1,
            total_iters=epochs // 4
        )

    def __call__(self):
        self.lr_scheduler.step()


class SelfTrainer():
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
        self, ner: NER, lr: float, train_dataloader: DataLoader, epochs: int,
        batch_size: int, max_length: int, device: Literal["cpu", "cuda"],
        accumulation_steps: int, ner_path: str, sam: bool,
        idx2label: Dict[int, str],
    ):
        self.sam = sam
        self.path = ner_path
        self.device = device
        self.batch_size = batch_size
        self.ner = ner
        self._ner = copy.deepcopy(self.ner)
        self.epochs = epochs
        self.accumulation_steps = accumulation_steps
        self.max_length = max_length
        self.idx2label = idx2label
        self.O = [k for k, v in idx2label.items() if v == "O"][0]
        self.pad = [k for k, v in idx2label.items() if v == "PAD"][0]
        if sam:
            self.optimizer = SAM(
                ner.parameters(),
                Adam,
                lr=lr
            )
        else:
            self.optimizer = Adam(
                ner.parameters(),
                lr=lr,
                betas=(0.9, 0.98),
                eps=0.9
            )
        self.lrs = LRScheduler(
            optimizer=self.optimizer,
            epochs=epochs
        )

    def _fit(self, train_dataloader: DataLoader):
        logging.info("Training...")
        losses = []
        x_list, y_list = [], []
        idx = 0
        for x, _ in tqdm(train_dataloader):
            self.ner.train()
            self.ner.zero_grad()
            self._ner.eval()
            with torch.no_grad():
                _y = self._ner.viterbi_decode(x)
            _y = [s + [self.pad] * (self.max_length - len(s)) for s in _y]
            _y = torch.LongTensor(_y).to(self.device)
            loss = self.ner(x, _y) / self.accumulation_steps
            losses.append(loss.item())                                      # Only the first loss is saved for statistics
            loss.backward()
            x_list.append(x)                                                # Save the inputs
            y_list.append(_y)                                                # Save the targets
            if ((idx + 1) % self.accumulation_steps == 0) \
            or ((idx + 1) == len(train_dataloader)):
                if self.sam:
                    self.optimizer.first_step(zero_grad=True)                   # First step
                    for i in range(len(y_list)):
                        loss = (
                            self.ner(x_list[i], y_list[i])
                            / self.accumulation_steps
                        )
                        loss.backward()
                    self.optimizer.second_step(zero_grad=True)                  # Second step
                    x_list, y_list = [], []                                     # Clear
                elif not self.sam:
                    self.optimizer.step()
            idx += 1
        train_loss = np.mean(losses)
        return train_loss

    @torch.no_grad()
    def _validate(self, val_dataloader: DataLoader):
        logging.info("Validating...")
        torch.set_printoptions(profile="full")
        self.ner.eval()
        y_true = []
        y_pred = []
        for x, y in val_dataloader:
            result = self.ner.viterbi_decode(x)
            y_pred.extend(result)
            y_true.extend(y.tolist())
        for i, y in enumerate(y_pred):
            for j, _ in enumerate(y):
                y_pred[i][j] = self.idx2label[y_pred[i][j]]
                y_true[i][j] = self.idx2label[y_true[i][j]]
            y_true[i] = y_true[i][:len(y_pred[i])]
            y_true[i][-1] = self.idx2label[self.O] # Replace the </s> tagged with PAD by an O for proper alignement
        return f1_score(y_true, y_pred, mode="strict", scheme=IOB2)

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
        f1 = 0.0
        try:
            for epoch in tqdm(range(self.epochs)):
                lr = self.lrs.optimizer.param_groups[0]["lr"]
                train_loss = self._fit(train_dataloader=train_dataloader)
                if val_dataloader is not None:
                    f1 = self._validate(val_dataloader)
                if wandb_:
                    wandb.log({"lr": lr, "train_loss": train_loss, "f1": f1})
                logging.info(
                    f"Epoch: {epoch + 1}\n"
                    + f"LR: {lr}\n"
                    + f"Train Loss: {train_loss}\n"
                    + f"F1: {f1}"
                )
                if train_loss <= best_loss:
                    torch.save({
                        "model_state_dict": self.ner.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "loss": train_loss,
                    }, "./tmp/ner_.pt")
                    best_loss = train_loss
                self.lrs()
                self._ner = copy.deepcopy(self.ner)
        except KeyboardInterrupt:
            logging.warning("Exiting from training early.")
            if train_loss <= best_loss:
                torch.save({
                    "model_state_dict": self.ner.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "loss": train_loss,
                }, "./tmp/ner_.pt")

    def save(self, path: str):
        """Saves the model to designated path.

        Parameters
        ----------
        path: ``str``
            Path to the locally saved model.
        """
        logging.info(f"Saving the NER model to {path}.")
        torch.save(self.ner, path)

