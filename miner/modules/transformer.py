# miner/modules/transformers.py

from typing import Literal

from transformers import RobertaForMaskedLM


class RoBERTa():
    """Transformer model for short english sentences. Based on RoBERTa [1]_.

    Parameters
    ----------
    device: str,  {"cuda", "cpu"}
        The hardware that will perform the computations.

    Attributes
    ----------
    model: transformers.RobertaForMaskedLM
        Transformer model for masked language modeling.

    References
    ----------
    ..  [1] Yinhan Liu et al. 2019. Roberta: A robustly optimized Bert
        pretraining approach. (July 2019). Retrieved January 31, 2023 from
        https://arxiv.org/abs/1907.11692
    """

    def __init__(self, device: Literal["cuda", "cpu"]):
        self.model = RobertaForMaskedLM.from_pretrained(
            "roberta-base"
        ).to(device)    # type: ignore

