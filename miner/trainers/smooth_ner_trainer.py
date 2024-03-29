# miner/trainers/smooth_ner_trainer.py

import logging
from typing import Optional, Literal, Dict

import wandb
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW, lr_scheduler
from transformers import get_linear_schedule_with_warmup
from seqeval.metrics import f1_score, precision_score, recall_score
from seqeval.scheme import IOB2


from miner.modules import SmoothNER
from miner.utils.data import SmoothNERDataset
from miner.utils import align_labels


class SmoothNERTrainer():
    """This class is responsible for training a SmoothNER model, which is
    designed for named entity recognition tasks. It supports training,
    validation, and logging of training metrics.

    Parameters
    ----------
    smooth_ner: SmoothNER
        The SmoothNER model to be trained.
    lr : float
        The learning rate for the optimizer.
    max_length: int
        The maximum sequence length for input data.
    ner_path: str
        The path where the trained model will be saved.
    epochs: int
        The number of training epochs.
    accumulation_steps : int
        The number of gradient accumulation steps before updating the model.
    device: str, {"cpu", "cuda"}
        The device on which the model will be trained ("cpu" or "cuda").
    idx2label: Dict[int, str]
        A dictionary mapping label indices to their corresponding string
        labels.

    Attributes
    ----------
    best_train_loss: float
        The best training loss achieved during training.
    best_f1: float
        The best F1 score achieved during validation.
    smooth_ner: SmoothNER
        The SmoothNER model to be trained.
    max_length: int
        The maximum sequence length for input data.
    idx2label: Dict[int, str]
        A dictionary mapping label indices to their corresponding string
        labels.
    ner_path: str
        The path where the trained model will be saved.
    epochs: int
        The number of training epochs.
    accumulation_steps : int
        The number of gradient accumulation steps before updating the model.
    device: Literal["cpu", "cuda"]
        The device on which the model will be trained ("cpu" or "cuda").
    optimizer: AdamW
        The optimizer used for training the model.
    scheduler: lr_scheduler.LambdaLR
        The learning rate scheduler for controlling the learning rate during
        training.
    """

    def __init__(
        self, smooth_ner: SmoothNER, lr: float, max_length: int, ner_path: str,
        epochs: int, accumulation_steps: int, device: Literal["cpu", "cuda"],
        idx2label: Dict[int, str]
    ):
        self.best_train_loss = np.inf
        self.best_f1 = .0
        self.smooth_ner = smooth_ner
        self.max_length = max_length
        self.idx2label = idx2label
        self.ner_path = ner_path
        self.epochs = epochs
        self.accumulation_steps = accumulation_steps
        self.device = device
        self.optimizer = AdamW(
            self.smooth_ner.parameters(),
            lr=lr,
            betas=(0.9, 0.98)
        )
        self.scheduler: lr_scheduler.LambdaLR

    def __call__(
        self, train_dataloader: DataLoader, wandb_: bool=False,
        val_dataset: Optional[SmoothNERDataset]=None
    ):
        try:
            self.train(train_dataloader, wandb_, val_dataset)
        except KeyboardInterrupt:
            logging.warning("Exiting from training early...")

    def _fit(self, train_dataloader: DataLoader):
        logging.info("Training...")
        losses = []
        step = 1
        self.smooth_ner.train()
        for x, x_augmented, y in tqdm(train_dataloader):
            self.optimizer.zero_grad()
            x = {key: val.squeeze(1) for key, val in x.items()}
            x_augmented = {
                key: val.squeeze(1) for key, val in x_augmented.items()
            }
            loss = self.smooth_ner(x, x_augmented, y) / self.accumulation_steps
            loss.backward()
            losses.append(loss.item())
            if step % self.accumulation_steps == 0:
                self.optimizer.step()
                self.scheduler.step()
            step += 1
        return np.mean(losses)

    @torch.inference_mode()
    def _validate(self, val_dataset: SmoothNERDataset):
        logging.info("Validating...")
        y_pred = []
        y_true = []
        with torch.no_grad():
            for idx in tqdm(range(len(val_dataset.labels))):
                inputs = val_dataset.tokenize(idx)
                local_y_pred = self.smooth_ner.decode(inputs)[0]
                local_y_pred = align_labels(
                    inputs,
                    local_y_pred,
                    self.idx2label
                )
                if len(local_y_pred) != len(val_dataset.labels[idx]):
                    continue
                y_pred.append(local_y_pred)
                y_true.append(val_dataset.labels[idx])
        precision = precision_score(y_true, y_pred, mode="strict", scheme=IOB2)
        recall = recall_score(y_true, y_pred, mode="strict", scheme=IOB2)
        f1 = f1_score(y_true, y_pred, mode="strict", scheme=IOB2)
        return precision, recall, f1

    def save_model(self):
        """Saves the model at ``self.smooth_ner`` in a local folder.
        """
        torch.save({
            "model_state_dict": self.smooth_ner.state_dict(),
        }, self.ner_path)

    def train(
        self, train_dataloader: DataLoader, wandb_: bool=False,
        val_dataset: Optional[SmoothNERDataset]=None
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
        steps_per_epoch = len(train_dataloader)
        num_training_steps = int(
            np.ceil((steps_per_epoch * self.epochs) / self.accumulation_steps)
        )
        self.scheduler = get_linear_schedule_with_warmup(
            optimizer=self.optimizer,
            num_warmup_steps=int(num_training_steps * 0.2),
            num_training_steps=num_training_steps
        )
        for epoch in tqdm(range(self.epochs)):
            f1 = .0
            precision = .0
            recall = .0
            train_loss = self._fit(train_dataloader)
            lr = self.scheduler.get_last_lr()[0]
            if val_dataset is not None:
                precision, recall, f1 = self._validate(val_dataset)
                if f1 > self.best_f1:
                    self.best_f1 = f1
                    self.save_model()
            elif train_loss < self.best_train_loss:
                self.best_train_loss = train_loss
                self.save_model()
            logging.info(
                f"Epoch: {epoch + 1}\n"
                + f"LR: {lr}\n"
                + f"Train Loss: {train_loss}\n"
                + f"F1: {f1}\n"
            )
            if wandb_: self.wandb_log(lr, train_loss,precision, recall, f1)

    @staticmethod
    def wandb_log(
        lr: float, train_loss: float, precision: float, recall: float,
        f1: float
    ):
        """Top-level wrapper function that logs some training metrics to your
        **Weight and Biases** project.

        Parameters
        ----------
        lr: float
            Last training loss.
        train_loss:
            Last training loss.
        precision: float
            Last validation precision.
        recall: float
            Last validation recall.
        f1: float
            Last validation F-score
        """
        logs = {
            "lr": lr,
            "train_loss": train_loss,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }
        wandb.log(logs)

