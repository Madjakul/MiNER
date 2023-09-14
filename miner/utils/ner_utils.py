# miner/utils/crf_utils.py

from typing import List, Dict, Union

from torch.optim import SGD, AdamW, lr_scheduler
import transformers

from miner.optimizer import SAM


class LRScheduler():
    """Learning rate scheduler wrapper.

    Parameters
    ----------
    optimizer: Union[SGD, AdamW, SAM]
        The optimizer used.
    patience: int
        How many epochs/steps to wait before updating the learning rate.
    factor: float
        Learning rate shrinking factor.

    Attributes
    ----------
    optimizer: Union[SGD, AdamW, SAM]
        The optimizer used.
    patience: int
        How many epochs/steps to wait before updating the learning rate.
    lr_scheduler: lr_scheduler.StepLR
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


def align_labels(
    inputs: transformers.BatchEncoding, labels: List[int],
    idx2label: Dict[int, str]
):
    """Get the label of the first token of each subword in order to align a
    model's output to the real labels. Transforms the output integers into
    strings.

    Parameters
    ----------
    inputs: transformers.BatchEncoding
        ``transformers`` tokenizer's output.
    labels: List[int]
        List of predicted labels.
    idx2label: Dict[int, str]
        Dictionary mapping the output integers to their string value.

    Returns
    -------
    label_ids: List[str]
        List of predicted labels as list of strings.
    """
    word_ids = inputs.word_ids()                                    # type: ignore
    label_ids = []
    previous_word_idx = -1
    for idx, word_idx in enumerate(word_ids):
        if word_idx != previous_word_idx and word_idx is not None:  # type: ignore
            label_ids.append(idx2label[labels[idx]])
        previous_word_idx = word_idx
    return label_ids

