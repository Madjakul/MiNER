# miner/utils/math_utils.py

import torch


def log_sum_exp(tensor: torch.Tensor, dim=-1, keepdim=False):
    """Compute log sum exp a numerically stable way for the forward algorithm.

    Parameters
    ----------
    tensor: torch.Tensor
        Input tensor.
    dim: int
        Output dimension. Default is -1 for automatique determination.
    keepdim: bool
        If the output dimension shall be the same as the input's.

    Returns
    -------
    max_score: torch.Tensor
        Rank 0 tensor containing the highest scores of the input.
    """
    max_score, _ = tensor.max(dim, keepdim=keepdim)
    if keepdim:
        stable_vec = tensor - max_score
    else:
        stable_vec = tensor - max_score.unsqueeze(dim)
    return max_score + (stable_vec.exp().sum(dim, keepdim=keepdim)).log()

