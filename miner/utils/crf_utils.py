# miner/utils/crf_utils.py

from typing import Optional, Literal

import torch
import torch.nn as nn


UNLABELED_INDEX = -1
IMPOSSIBLE_SCORE = -100


def create_possible_tag_masks(num_tags: int, tags: torch.Tensor):
    """Transforms a vector of single integers representing the labels into a
    multi-label binary matrix.

    Parameters
    ----------
    num_tags: ``int``
        Number of labels.
    tags: ``torch.Tensor``
        Vector of integers.

    Returns
    -------
    masks: ``torch.Tensor``
        Multi-label binary matrix.
    """
    copy_tags = tags.clone()
    no_annotation_idx = (copy_tags == UNLABELED_INDEX)
    copy_tags[copy_tags == UNLABELED_INDEX] = 0

    tags_ = torch.unsqueeze(copy_tags, 2)
    masks = torch.zeros(
        tags_.size(0),
        tags_.size(1),
        num_tags,
        dtype=torch.uint8,
        device=tags.device
    )
    masks.scatter_(2, tags_, 1)
    masks[no_annotation_idx] = 1
    return masks

def get_batch_size(
    model: nn.Module, max_length: int, dataset_size: int,
    device: Literal["cpu", "cuda"], max_batch_size: Optional[int]=None,
    num_iterations: int=5
):
    """Performs the forward pass with dummy input and target variables in order
    to determine the maximum batch size supported.

    Parameters
    ----------
    model: ``miner.modules.NER``
        NER model.
    max_length: ``int``
        Sequence max length.
    dataset_size: ``int``
        Maximum number of examples in the dataset.
    device: ``str``, {"cpu", "cuda"}
        Device where the computation is performed.
    max_batch_size: ``int``, Optional
        Desired maximum batch size
    num_iterations: ``int``
        Maximum number of training steps before breaking. Default to 5.

    Returns
    -------
    batch_size: ``int``
        Maximum batch size supported on your device.
    """
    model.train(True)
    optimizer = torch.optim.Adam(model.parameters())

    batch_size = 2
    while True:
        if max_batch_size is not None and batch_size >= max_batch_size:
            batch_size = max_batch_size
            break
        if batch_size >= dataset_size:
            batch_size = batch_size // 2
            break
        try:
            for _ in range(num_iterations):
                # dummy inputs and targets
                inputs = torch.randint(
                    10, 1000,
                    (batch_size, max_length),
                    device=device
                )
                targets = torch.randint(
                    1, 5,
                    (batch_size, max_length),
                    device=device
                )
                masks = torch.ones(*(batch_size, max_length), device=device)
                x = {"input_ids": inputs, "attention_mask": masks}
                loss = model(x, targets)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            batch_size *= 2
        except RuntimeError:
            batch_size //= 2
            break
    del model, optimizer
    torch.cuda.empty_cache()
    return batch_size

