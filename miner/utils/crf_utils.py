# miner/utils/crf_utils.py

import torch


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

