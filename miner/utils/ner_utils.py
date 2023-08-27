# miner/utils/crf_utils.py

from typing import List, Dict

import torch
from torch.nn.utils.rnn import pad_sequence


def select_values(
    y_pred: List[List[int]], y_true: List[List[int]], mask: torch.ByteTensor,
    idx2label: Dict[int, str], max_length: int
):
    padded_tensor = pad_sequence(
        [torch.tensor(sublist) for sublist in y_pred],
        batch_first=True,
        padding_value=0
    )
    padded_y_pred = torch.zeros(
        padded_tensor.size(0),
        max_length,
        dtype=torch.long
    )
    padded_y_pred[:, :padded_y_pred.size(1)] = padded_y_pred

    padded_tensor = pad_sequence(
        [torch.tensor(sublist) for sublist in y_true],
        batch_first=True,
        padding_value=0
    )
    padded_y_true = torch.zeros(
        padded_tensor.size(0),
        max_length,
        dtype=torch.long
    )
    padded_y_true[:, :padded_y_true.size(1)] = padded_y_true

    selected_y_pred = torch.where(
        mask==1,
        padded_y_pred,
        torch.neg(torch.ones_like(padded_y_pred))
    )
    selected_y_true = torch.where(
        mask ==1,
        padded_y_true,
        torch.neg(torch.ones_like(padded_y_true))
    )

    y_pred_ = [
        [idx2label[j] for j in i if j != -1] for i in selected_y_pred.tolist()
    ]
    y_true_ = [
        [idx2label[j] for j in i if j != -1] for i in selected_y_true.tolist()
    ]
    return y_pred_, y_true_

