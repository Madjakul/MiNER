# miner/modules/ner.py

import logging
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from transformers import RobertaModel


class SmoothNER(nn.Module):
    """Named Entity Recognizer (NER) model with label smoothing.

    This class defines a NER model with label smoothing, which is used to train
    models for NER tasks. It extends the PyTorch ``nn.Module`` class and
    integrates with the **Hugging Face** ``transformers`` library for handling
    pre-trained language models.

    Parameters
    ----------
    lm_path : str
        The path or identifier of a pre-trained language model checkpoint.
    num_labels : int
        The number of unique labels or tags for NER.
    device : str, {"cpu", "cuda"}
        The device on which the model will be instantiated ("cpu" or "cuda").
    dropout : float
        The dropout probability to apply to the model's hidden states.

    Attributes
    ----------
    device : str
        The device on which the model is instantiated.
    transformer : transformers.RobertaModel
        The pre-trained transformer model used for feature extraction.
    linear_dropout : nn.Dropout
        The dropout layer applied to the model's linear layer.
    fc : nn.Linear
        The linear layer mapping features to label scores.
    kl_loss : nn.KLDivLoss
        The Kullback-Leibler Divergence loss used for label smoothing.
    """

    def __init__(
        self, lm_path: str, num_labels: int, device: Literal["cpu", "cuda"],
        dropout: float
    ):
        super(SmoothNER, self).__init__()
        self.device = device
        logging.info(f"Loading LM checkpoint from {lm_path}")
        self.transformer = RobertaModel.from_pretrained(lm_path)
        self.linear_dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(768, num_labels)    # (batch_size, max_length, num_labels)
        self.kl_loss = nn.KLDivLoss(reduction="batchmean")

    def forward(
        self, inputs: transformers.BatchEncoding,
        augmented_inputs: transformers.BatchEncoding,
        targets: torch.FloatTensor
    ):
        """Computes the KL-Divergence over the predicted sequence and the
        target sequence.

        Parameters
        ----------
        inputs : transformers.BatchEncoding
            Original input sequence tokenized with ``transformers``.
        augmented_inputs : transformers.BatchEncoding
            Augmented input sequence tokenized with ``transformers``.
        targets : torch.FloatTensor
            Target sequence for label smoothing.

        Returns
        -------
        kl_loss : torch.Tensor
            The computed KL-Divergence loss.
        """
        h = self.transformer(**inputs).last_hidden_state
        logits = self.fc(self.linear_dropout(h))
        log_p = F.log_softmax(logits, dim=2)
        kl_loss = self.kl_loss(log_p, targets)
        h_augmented = self.transformer(**augmented_inputs).last_hidden_state
        logits_augmented = self.fc(self.linear_dropout(h_augmented))
        log_p_augmented = F.log_softmax(logits_augmented, dim=2)
        kl_loss_augmented = self.kl_loss(log_p_augmented, targets)
        return kl_loss + kl_loss_augmented

    @torch.inference_mode()
    def get_proba(self, inputs: transformers.BatchEncoding):
        """Returns sequence probabilities.

        Parameters
        ----------
        inputs : transformers.BatchEncoding
            Input sequence tokenized with ``transformers``.

        Returns
        -------
        p : torch.Tensor
            Sequence probabilities.
        """
        h = self.transformer(**inputs).last_hidden_state
        logits = self.fc(self.linear_dropout(h))
        p = F.softmax(logits, dim=2)
        return p

    @torch.inference_mode()
    def decode(self, inputs: transformers.BatchEncoding):
        """Returns sequences of tags.

        Parameters
        ----------
        inputs : transformers.BatchEncoding
            Input sequence tokenized with ``transformers``.

        Returns
        -------
        tags : List[List[int]]
            A sequence of predicted tags.
        """
        h = self.transformer(**inputs).last_hidden_state
        logits = self.fc(self.linear_dropout(h))
        p = F.softmax(logits, dim=2)
        tags = p.argmax(dim=2)
        return tags.tolist()

