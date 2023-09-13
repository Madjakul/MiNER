# miner/utils/crf_utils.py

import torch


UNLABELED_INDEX = -1
IMPOSSIBLE_SCORE = -100


def create_possible_tag_masks(num_tags: int, tags: torch.LongTensor):
    """Creates a mask-like sparse tensor where the index of the correct tag has
    a value of 1, allowing for multilabel targets.

    Parameters
    ----------
    num_tags: int
        Number of different tags in the dataset.
    tags: torch.LongTensor
        Target labels. (batch_size, sequence_length).

    Returns
    -------
    masks: torch.ByteTensor
        Mask-like sparse tensor indicating the target label.
        (batch_size, sequence_length, num_tags).
    """
    copy_tags = tags.clone()
    no_annotation_idx = (copy_tags == UNLABELED_INDEX)
    copy_tags[no_annotation_idx] = 0
    masks = torch.zeros(
        copy_tags.size(0),
        copy_tags.size(1),
        num_tags,
        dtype=torch.uint8,
        device=tags.device
    )
    masks.scatter_(2, copy_tags.unsqueeze(2), 1)
    masks[no_annotation_idx] = 1    # (batch_size, sequence_length, num_tags)
    return masks    # type: ignore

def custom_argmax(input_tensor: torch.FloatTensor, threshold: float):
    """Performances a custom ``argmax`` where the index containing the greatest
    value has to be greater than a threshold, otherwise it returns the index 0.

    Parameters
    ----------
    input_tensor: torch.FloatTensor
        Logits.
    threshold: float
        Minimum value an index needs to have in order to be returned by the
        ``argmax`` function.

    Returns
    -------
    max_indices: torch.LongTensor
        Result of the custom ``argmax``. (batch_size, sequence_length).
    """
    max_values, max_indices = torch.max(input_tensor, dim=-1)
    mask = max_values > threshold
    max_indices[~mask] = 0
    return max_indices

