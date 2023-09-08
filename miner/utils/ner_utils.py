# miner/utils/crf_utils.py

from typing import List, Dict, Union

from torch.optim import SGD, AdamW, lr_scheduler
import transformers

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


def align_labels(inputs: transformers.BatchEncoding, labels: List[int], idx2label: Dict[int, str]):
        word_ids = inputs.word_ids()                    # type: ignore
        label_ids = []
        previous_word_idx = -1
        # print(word_ids, "\n")
        # print(labels, "\n")
        for idx, word_idx in enumerate(word_ids):
            if word_idx != previous_word_idx and word_idx is not None:         # type: ignore
                label_ids.append(idx2label[labels[idx]])
            previous_word_idx = word_idx
        # print(label_ids, "\n\n\n")
        return label_ids

