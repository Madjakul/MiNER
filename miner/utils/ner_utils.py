# miner/utils/crf_utils.py

from typing import Union

from torch.optim import SGD, AdamW, lr_scheduler

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
    """

    def __init__(
        self, optimizer: Union[SGD, AdamW, SAM], patience: int, factor: float
    ):
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler.StepLR(
            optimizer=optimizer,
            step_size=patience,
            gamma=factor,
        )

    def __call__(self):
        self.lr_scheduler.step()

