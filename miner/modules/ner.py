# miner/modules/ner.py

import logging
from typing import Literal, Dict

import torch
import torch.nn as nn
from transformers import (
    RobertaModel, CamembertModel, LongformerModel
)



class NER(nn.Module):
    """Named Entity Recognizer. RoBERTa-like model with a fuzzy/partial
    conditional random filed on top.

    Parameters
    ----------
    lang: ``str``, {"en", "fr"}
        language of the training data.
    lm_path: ``str``
        Path to a local or online checkpoint.
    num_labels: ``int``
        Number of possible labels. The unknown labels are not taken into
        account contrary to the padding label.
    padding_idx: ``int``
        Integer mapped to the padding label.
    max_length: ``int``
        Maximum length of a sequence.
    device: ``str``, {"cpu", "cuda"}
        Wether or not to use the GPU for computation.

    Attributes
    ----------
    device: ``str``, {"cpu", "cuda"}
        Wether or not to use the GPU for computation.
    transformer: ``transformers.CamembertModel``, ``transformers.LongformerModel``, ``transformers.RobertaModel``
        Pretrained LLM checkpoint to load.
    linear_dropout: ``torch.nn.Dropout``
        Linear dropout.
    fc: ``torch.nn.Linear``
        Fully connected layer. (batch_size, max_length, num_labels).
    crf: ``miner.modules.PartialCRF``
        Partial/fuzzy crf layer.
    """

    def __init__(
        self, lang: Literal["en", "fr"], lm_path: str, num_labels: int,
        max_length: int, device: Literal["cpu", "cuda"], dropout: float=0.2,
    ):
        super(NER, self).__init__()
        self.device = device
        logging.info(f"Loading LM checkpoint from {lm_path}")
        if lang == "fr":
            self.transformer = CamembertModel.from_pretrained(lm_path)
        elif max_length > 512:
            self.transformer = LongformerModel.from_pretrained(lm_path)
        else:
            self.transformer = RobertaModel.from_pretrained(lm_path)
        self.linear_dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(768, num_labels)    # (batch_size, max_length, num_labels)

    def forward(
        self, inputs: Dict[str, torch.Tensor], outputs: torch.Tensor
    ):
        """Performs the forward pass.

        Parameters
        ----------
        inputs: ``dict``
            Input dictionary from **HuggingFace**'s tokenizer with format
            {"input_ids": torch.tensor(), "attention_mask": torch.tensor()}.
        outputs: ``torch.Tensor``
            List of true labels.

        Returns
        -------
        nll: ``torch.Tensor``
            Result of the negative log-likelihood performed over the
            inference's results, expressed as a rank 0 tensor.
        """
        masks = inputs["attention_mask"].squeeze(1).to(self.device)
        h = self.transformer(                                           # type: ignore
            input_ids=inputs["input_ids"].squeeze(1).to(self.device),
            attention_mask=masks
        ).last_hidden_state.to(self.device)
        logits = self.fc(self.linear_dropout(h))

