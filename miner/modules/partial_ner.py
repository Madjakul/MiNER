# miner/modules/partial_ner.py

import logging
from typing import Optional, Literal

import torch
import torch.nn as nn
import transformers
from transformers import RobertaModel

from miner.modules.partial_crf import PartialCRF


class PartialNER(nn.Module):
    """Partial Named Entity Recognizer (NER) Model.

    This class defines a Partial NER model for named entity recognition tasks.
    It extends the PyTorch ``nn.Module`` class and integrates with the
    **Hugging Face** ``transformers`` library for handling pre-trained language
    models.

    Parameters
    ----------
    lm_path: str
        The path or identifier of a pre-trained language model checkpoint.
    num_labels: int
        The number of unique labels or tags for NER.
    device: str, {"cpu", "cuda"}
        The device on which the model will be instantiated ("cpu" or "cuda").
    dropout: float
        The dropout probability to apply to the model's hidden states.
    q: float, optional
        The q-value for the partial CRF layer. If None, no partial CRF is used.
    padding_idx : Optional[int], optional
        The padding index for the input sequences. If None, the default index
        is used.

    Attributes
    ----------
    device: str
        The device on which the model is instantiated.
    transformer: transformers.RobertaModel
        The pre-trained transformer model used for feature extraction.
    linear_dropout: nn.Dropout
        The dropout layer applied to the model's linear layer.
    fc: nn.Linear
        The linear layer mapping features to label scores.
    crf: PartialCRF
        The partial conditional random field layer for structured prediction.
    """

    def __init__(
        self, lm_path: str, num_labels: int, device: Literal["cpu", "cuda"],
        dropout: float, q: Optional[float]=None,
        padding_idx: Optional[int]=None
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
        self, inputs: transformers.BatchEncoding,
        inputs_augmented: transformers.BatchEncoding,
        outputs: torch.LongTensor,
        loss_fn: Optional[Literal["nll", "c_nll", "gce"]]=None
    ):
        """Performs the forward pass.

        Parameters
        ----------
        inputs: transformer.BatchEncoding
            Original sentence, tokenized with ``transformers``.
        inputs_augmented: torch.BatchEncoding
            Language augmented input, tokenized with ``transformers``.
        outputs: torch.LongTensor
            List of true labels.
        loss_fn: str, {"nll", "c_nll", "gce"}, optional
            The desired loss function to use.

        Returns
        -------
        torch.FloatTensor
            Sum over the loss of the original input and the augmented input.
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
        """Computes the mostly likely label sequence.

        Parameters
        ----------
        inputs: transformers.BatchEncoding
            Input sentence tokenized with ``transformers``.

        Returns
        -------
        outputs: List[List[int]]
            Most likely tag sequence of each input in the batch.
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
        """Computes the marginal probability of each token of a given sequence
        to belong to a class.

        Parameters
        ----------
        inputs: transformers.BatchEncoding
            Input sentence tokenized with ``transformers``.

        Returns
        -------
        p: torch.FloatTensor
            Marginal probabilities. (batch_size, sequence_length, num_tags).

        """
        h = self.transformer(**inputs).last_hidden_state
        logits = self.fc(self.linear_dropout(h))
        p = self.crf.marginal_probabilities(
            logits,
            mask=inputs["attention_mask"]
        ).transpose(0, 1)
        return p

