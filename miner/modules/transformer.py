# miner/modules/transformers.py

from typing import Literal

from transformers import (
    RobertaForMaskedLM, CamembertForMaskedLM, LongformerForMaskedLM
)


class RoBERTa():
    """Transformer model for short english sentences. Based on RoBERTa [1]_.

    Parameters
    ----------
    device: ``str``,  {"cuda", "cpu"}
        Wether or not to use GPU for computation.

    Attributes
    ----------
    model: ``transformers.RobertaForMaskedLM``
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


class CamemBERT():
    """Transformer model for short french sentences. Based on CamemBERT [1]_.

    Parameters
    ----------
    device: ``str``,  {"cuda", "cpu"}
        Wether or not to use GPU for computation.

    Attributes
    ----------
    model: ``transformers.CamembertForMaskedLM``
        Transformer model for masked language modeling.

    References
    ----------
    ..  [1] Louis Martin et al. 2020. Camembert: A tasty French language model.
        (May 2020). Retrieved January 31, 2023 from
        https://arxiv.org/abs/1911.03894
    """

    def __init__(self, device: Literal["cuda", "cpu"]):
        self.model = CamembertForMaskedLM.from_pretrained(
            "camembert-base"
        ).to(device)    # type: ignore


class Longformer():
    """Transformer model for long english sentences. Based on Longformer [1]_.

    Parameters
    ----------
    device: ``str``,  {"cuda", "cpu"}
        Wether or not to use GPU for computation.

    Attributes
    ----------
    model: ``transformers.LongformerForMaskedLM``
        Transformer model for masked language modeling.

    References
    ----------
    ..  [1] Iz Beltagy, Matthew E. Peters, and Arman Cohan. 2020. Longformer:
        The long-document transformer. (December 2020). Retrieved January 31,
        2023 from https://arxiv.org/abs/2004.05150
    """

    def __init__(self, device: Literal["cuda", "cpu"]):
        self.model = LongformerForMaskedLM.from_pretrained(
            "allenai/longformer-base-4096"
        ).to(device)    # type: ignore

