# miner/trainers/smooth_ner_trainer.py

import logging
from typing import Optional, Literal

import wandb
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW

from miner.modules import SmoothNER
from miner.utils import LRScheduler


class SmoothNERTrainer():
    """Trains the smooth NER

    Parameters
    ----------

    Attributes
    ----------
    """

    def __init__(
        self, smooth_ner: SmoothNER, lr: float, max_length: int, ner_path: str,
        epochs: int, accumulation_steps: int, device: Literal["cpu", "cuda"]
    ):
        self.best_train_loss = np.inf
        self.best_val_loss = np.inf
        self.smooth_ner = smooth_ner
        self.max_length = max_length
        self.ner_path = ner_path
        self.epochs = epochs
        self.accumulation_steps = accumulation_steps
        self.device = device
        self.optimizer = AdamW(self.smooth_ner.parameters(), lr=lr)
        self.lrs = LRScheduler(
            optimizer=self.optimizer,
            patience=1,
            factor=0.6
        )

    def __call__(
        self, train_dataloader: DataLoader, wandb_: bool=False,
        val_dataloader: Optional[DataLoader]=None
    ):
        try:
            self.train(train_dataloader, wandb_, val_dataloader)
        except KeyboardInterrupt:
            logging.warning("Exiting from training early...")

    def _fit(self, train_dataloader: DataLoader):
        logging.info("Training...")
        losses = []
        step = 1
        self.smooth_ner.train()
        for x, y in tqdm(train_dataloader):
            self.optimizer.zero_grad()
            x = {key: val.squeeze(1) for key, val in x.items()}
            loss = self.smooth_ner(x, y) / self.accumulation_steps
            loss.backward()
            losses.append(loss.item())
            if step % self.accumulation_steps == 0:
                self.optimizer.step()
            step += 1
        return np.mean(losses)

    @torch.inference_mode()
    def _validate(self, val_dataloader: DataLoader):
        logging.info("Validating...")
        self.smooth_ner.eval()
        losses = []
        for x, y in tqdm(val_dataloader):
            x = {key: val.squeeze(1) for key, val in x.items()}
            loss = self.smooth_ner(x, y) / self.accumulation_steps
            losses.append(loss.item())
        return np.mean(losses)

    def save_model(self):
        torch.save({
            "model_state_dict": self.smooth_ner.state_dict(),
        }, self.ner_path)

    def train(
        self, train_dataloader: DataLoader, wandb_: bool=False,
        val_dataloader: Optional[DataLoader]=None
    ):
        for epoch in tqdm(range(self.epochs)):
            val_loss = None
            lr = self.lrs.optimizer.param_groups[0]["lr"]
            train_loss = self._fit(train_dataloader)
            if val_dataloader is not None:
                val_loss = self._validate(val_dataloader)
                if val_loss < self.best_val_loss:
                    self.val_loss = val_loss
                    self.save_model()
            elif train_loss < self.best_train_loss:
                self.best_train_loss = train_loss
                self.save_model()
            logging.info(
                f"Epoch: {epoch + 1}\n"
                + f"LR: {lr}\n"
                + f"Train Loss: {train_loss}\n"
                + f"val_loss: {val_loss}"
            )
            if wandb_: self.wandb_log(lr, train_loss, val_loss)
            self.lrs()

    @staticmethod
    def wandb_log(lr: float, train_loss: float, val_loss: float):
        logs = {
            "lr": lr,
            "train_loss": train_loss,
            **({"val_loss": val_loss} if val_loss is not None else {})
        }
        wandb.log(logs)

