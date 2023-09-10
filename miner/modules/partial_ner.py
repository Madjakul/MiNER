# miner/modules/partial_ner.py

import logging
from typing import Optional, Literal

import torch
import torch.nn as nn
import transformers
from transformers import (
    RobertaModel, CamembertModel, LongformerModel
)

from miner.modules.partial_crf import PartialCRF


class PartialNER(nn.Module):
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
        self, lm_path: str, num_labels: int,
        device: Literal["cpu", "cuda"], dropout: float,
        q: Optional[float]=None, padding_idx: Optional[int]=None
    ):
        super(PartialNER, self).__init__()
        self.device = device
        logging.info(f"Loading LM checkpoint from {lm_path}")
        self.transformer = RobertaModel.from_pretrained(lm_path)
        self.linear_dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(768, num_labels)    # (batch_size, max_length, num_labels)
        self.crf = PartialCRF(
            num_tags=num_labels,
            device=device,
            q=q,
            padding_idx=padding_idx
        )

    def forward(
        self, inputs: transformers.BatchEncoding, inputs_augmented: transformers.BatchEncoding, outputs: torch.LongTensor,
        loss_fn: Optional[Literal["nll", "c_nll", "gce"]]=None
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
        h = self.transformer(**inputs).last_hidden_state
        logits = self.fc(self.linear_dropout(h))
        loss = self.crf(
            emissions=logits,
            tags=outputs,
            mask=inputs["attention_mask"],
            loss_fn="nll" if loss_fn is None else loss_fn
        )
        h_augmented = self.transformer(**inputs_augmented).last_hidden_state
        logits_augmented = self.fc(self.linear_dropout(h_augmented))
        loss_augmented = self.crf(
            emissions=logits_augmented,
            tags=outputs,
            mask=inputs_augmented["attention_mask"],
            loss_fn="nll" if loss_fn is None else loss_fn
        )
        return loss + loss_augmented

    @torch.inference_mode()
    def viterbi_decode(self, inputs: transformers.BatchEncoding):
        """Pass
        """
        h = self.transformer(**inputs).last_hidden_state
        logits = self.fc(self.linear_dropout(h))
        outputs = self.crf.viterbi_decode(
            logits,
            mask=inputs["attention_mask"]
        )
        return outputs

    @torch.inference_mode()
    def marginal_probabilities(self, inputs: transformers.BatchEncoding):
        """Pass
        """
        h = self.transformer(**inputs).last_hidden_state
        logits = self.fc(self.linear_dropout(h))
        p = self.crf.marginal_probabilities(
            logits,
            mask=inputs["attention_mask"]
        ).transpose(0, 1)
        return p

