# miner/modules/ner.py

from typing import Dict

import torch
import torch.nn as nn
from transformers import (
    RobertaModel, CamembertModel, LongformerModel
)

from miner.modules.partial_crf import PartialCRF


class NER(nn.Module):
    """Named Entity Recognizer.

    Parameters
    ----------

    Attributes
    ----------

    References
    ----------

    Examples
    --------
    """

    def __init__(
        self, lang: str, lm_path: str, num_labels: int, padding_idx: int,
        max_length: int, device: str
    ):
        super(NER, self).__init__()
        self.device = device
        if lang == "fr":
            self.transformer = CamembertModel.from_pretrained(lm_path)
        elif max_length > 512:
            self.transformer = LongformerModel.from_pretrained(lm_path)
        else:
            self.transformer = RobertaModel.from_pretrained(lm_path)
        self.linear_dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(768, num_labels)    # (batch_size, max_length, num_labels)
        self.partial_crf = PartialCRF(          # (batch_size, max_length)
            num_tags=num_labels,
            padding_idx=padding_idx
        )

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
        masks: ``torch.Tensor``
            Labels to not include in the computation of the loss (e.g. outside
            and padding).

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
        nll = self.partial_crf(logits, outputs, mask=masks)
        return nll

    def viterbi_decode(self, inputs: Dict[str, torch.Tensor]):
        """Decodes the hidden states of a given batch.

        Parameters
        ----------
        inputs: ``dict``
            Input dictionary from **HuggingFace**'s tokenizer with format
            {"input_ids": torch.tensor(), "attention_mask": torch.tensor()}.
        masks: ``torch.Tensor``
            Labels to not include in the computation of the loss (e.g. outside
            and padding).

        Returns
        -------
        tag_seq: ``torch.Tensor``
            Predicted sequence of tag.
        """
        masks = inputs["attention_mask"].squeeze(1).to(self.device)
        h = self.transformer(   # type: ignore
            input_ids=inputs["input_ids"].squeeze(1).to(self.device),
            attention_mask=masks
        ).last_hidden_state.to(self.device)
        logits = self.fc(self.linear_dropout(h))
        tag_seq = self.partial_crf.viterbi_decode(logits, mask=masks)   # type: ignore
        return tag_seq

