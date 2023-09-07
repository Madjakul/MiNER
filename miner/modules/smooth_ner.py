# miner/modules/ner.py

import logging
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from transformers import (
    RobertaModel, CamembertModel, LongformerModel
)


class SmoothNER(nn.Module):
    """NER model for label smoothing.
    """

    def __init__(
        self, lang: Literal["en", "fr"], lm_path: str, num_labels: int,
        max_length: int, device: Literal["cpu", "cuda"], dropout: float
    ):
        super(SmoothNER, self).__init__()
        self.device = device
        logging.info(f"Loading LM checkpoint from {lm_path}")
        if lang == "fr" and max_length <= 512:
            self.transformer = CamembertModel.from_pretrained(lm_path)
        elif max_length > 512 and lang == "en":
            self.transformer = LongformerModel.from_pretrained(lm_path)
        elif lang == "en" and max_length <= 512:
            self.transformer = RobertaModel.from_pretrained(lm_path)
        else:
            raise ValueError(
                f"Wrong combination of language ({lang}) and maximum sequence"
                + f" length ({max_length})."
            )
        self.linear_dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(768, num_labels)    # (batch_size, max_length, num_labels)
        self.kl_loss = nn.KLDivLoss(reduction="batchmean")

    def forward(
        self, inputs: transformers.BatchEncoding, targets: torch.FloatTensor
    ):
        """Computes the KL-Divergence over the predicted sequence and the
        target sequence.

        Parameters
        ----------

        Returns
        -------
        """
        h = self.transformer(**inputs).last_hidden_state
        logits = self.fc(self.linear_dropout(h))
        mask = inputs["attention_mask"].unsqueeze(2).expand(logits.shape[0], logits.shape[1], logits.shape[2])
        log_p = F.log_softmax(logits, dim=2) * mask
        kl_loss = self.kl_loss(log_p, targets)
        return kl_loss

    @torch.inference_mode()
    def decode(self, inputs: transformers.BatchEncoding):
        """Returns a sequence of tags.

        Parameters
        ----------

        Returns
        -------
        """
        h = self.transformer(**inputs).last_hidden_state
        logits = self.fc(self.linear_dropout(h))
        p = F.softmax(logits, dim=2)
        tags = p.argmax(dim=2)
        return tags

