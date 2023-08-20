# miner/utils/crf_utils.py

import torch


UNLABELED_INDEX = -1
IMPOSSIBLE_SCORE = -100


def create_possible_tag_masks(
    num_tags: int, tags: torch.LongTensor,
) -> torch.ByteTensor:
    """Creates a mask-like sparse tensor where the index of the correct tag has
    a value of 1, allowing for multilabel targets.

    Parameters
    ----------
    num_tags: int
        Number of different tags in the dataset.
    tags: torch.LongTensor
        Tensor of target labels. (batch_size, sequence_length).

    Returns
    -------
    masks: torch.ByteTensor
        Mask-like sparse tensor indicating the target label.
        (batch_size, sequence_length, num_tags).
    """
    copy_tags = tags.clone()
    no_annotation_idx = (copy_tags == UNLABELED_INDEX)
    copy_tags[no_annotation_idx].fill_(0)

    masks = torch.zeros(
        copy_tags.size(0),
        copy_tags.size(1),
        num_tags,
        dtype=torch.uint8,
        device=tags.device
    )
    masks.scatter_(2, copy_tags.unsqueeze(2), 1)
    masks[no_annotation_idx].fill_(1)
    return masks    # type: ignore

