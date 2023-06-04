# miner/modules/partial_crf.py

from typing import Optional

import torch
import torch.nn as nn

from miner.modules.base_crf import BaseCRF
from miner.utils import (
    IMPOSSIBLE_SCORE, create_possible_tag_masks, log_sum_exp
)


class PartialCRF(BaseCRF):
    """Partial/Fuzzy Conditional random field[1]_.

    Parameters
    ----------
    num_tags: ``int``
        Number of possible tags.
    padding_idx: ``int``
        Integer representing the padding tag.

    References
    ----------
    ..  [1] Kajyuuen. Pytorch-partial-crf/partial_crf.py at master ·
        Kajyuuen/pytorch-partial-CRF. Retrieved November 8, 2022 from
        https://github.com/kajyuuen/pytorch-partial-crf/blob/master/pytorch_partial_crf/partial_crf.py
    """
    def __init__(
        self, num_tags: int, padding_idx: Optional[int]=None,
        corrected_loss: Optional[bool]=None, gamma: Optional[float]=None
    ):
        super().__init__(num_tags, padding_idx, corrected_loss, gamma)

    def _reset_parameters(self) -> None:
        nn.init.uniform_(self.start_transitions, -0.1, 0.1)
        nn.init.uniform_(self.end_transitions, -0.1, 0.1)
        nn.init.uniform_(self.transitions, -0.1, 0.1)

    def forward(
        self, emissions: torch.Tensor, tags: torch.Tensor,
        mask: Optional[torch.Tensor]=None
    ):
        """Compute the negative log-likelihood of an observed sequence of tags.

        Parameters
        ----------
        emissions: ``torch.Tensor``
            Emission scores of each tokens.
        tags: ``torch.Tensor``
            Seqeuence of true labels.
        mask: ``torch.Tensor``
            Binary tensor mapping the padding labels to 0.

        Returns
        -------
        torch.sum(forward_score - gold_score): ``torch.Tensor``
            Negative log-likelihood.
        """
        if mask is None:
            mask = torch.ones_like(tags, dtype=torch.uint8)
        possible_tags = create_possible_tag_masks(self.num_tags, tags)

        gold_score = self._numerator_score(emissions, mask, possible_tags).double()
        forward_score = self._denominator_score(emissions, mask).double()
        nll = forward_score - gold_score
        print(nll)
        if self.corrected_loss:
            nlu = -(1 - (-nll).exp()).log()
            if torch.isnan(nlu).any() or torch.isinf(nlu).any():
                nl = (1 - (-nll).exp())
                nl = nl + (nl < 1e-4).to(nl).detach() * (1e-4 - nl).detach()
                nlu = - nl.log()
            print(nlu)
            return torch.sum(nll * self.gamma + nlu * (1 - self.gamma))
        return torch.sum(nll * self.gamma)

    def _denominator_score(
        self, emissions: torch.Tensor, mask: torch.Tensor
    ):
        """ Computes the partition score.

        Parameters
        ----------
        emissions: ``torch.Tensor``
            (batch_size, sequence_length, num_tags).
        mask: ``torch.Tensor``
            Show padding tags. 0 don't calculate score. (batch_size,
            sequence_length).

        Returns
        -------
        scores: ``torch.Tensor``
            (batch_size).
        """
        batch_size, sequence_length, num_tags = emissions.data.shape

        emissions = emissions.transpose(0, 1).contiguous()
        mask = mask.float().transpose(0, 1).contiguous()

        # Start transition score and first emissions score
        alpha = self.start_transitions.view(1, num_tags) + emissions[0]

        for i in range(1, sequence_length):

            emissions_score = emissions[i].view(batch_size, 1, num_tags)      # (batch_size, 1, num_tags)
            transition_scores = self.transitions.view(1, num_tags, num_tags)  # (1, num_tags, num_tags)
            broadcast_alpha = alpha.view(batch_size, num_tags, 1)             # (batch_size, num_tags, 1)

            inner = broadcast_alpha + emissions_score + transition_scores     # (batch_size, num_tags, num_tags)

            alpha = (log_sum_exp(inner, 1) * mask[i].view(batch_size, 1) +
                     alpha * (1 - mask[i]).view(batch_size, 1))

        # Add end transition score
        stops = alpha + self.end_transitions.view(1, num_tags)

        return log_sum_exp(stops) * 1e-2 # (batch_size,)

    def _numerator_score(
        self, emissions: torch.Tensor, mask: torch.Tensor,
        possible_tags: torch.Tensor
    ):
        """Computes the sentence's score.

        Parameters
        ----------
        emissions: ``torch.Tensor``
            (batch_size, sequence_length, num_tags).
        mask: ``torch.Tensor``
            Show padding tags. 0 don't calculate score. (batch_size,
            sequence_length).

        Returns
        -------
        scores: ``torch.Tensor``
            (batch_size).
        """

        batch_size, sequence_length, num_tags = emissions.data.shape

        emissions = emissions.transpose(0, 1).contiguous()
        mask = mask.float().transpose(0, 1).contiguous()
        possible_tags = possible_tags.float().transpose(0, 1)

        # Start transition score and first emission
        first_possible_tag = possible_tags[0]

        alpha = self.start_transitions + emissions[0]      # (batch_size, num_tags)
        alpha[(first_possible_tag == 0)] = IMPOSSIBLE_SCORE

        for i in range(1, sequence_length):
            current_possible_tags = possible_tags[i-1] # (batch_size, num_tags)
            next_possible_tags = possible_tags[i]      # (batch_size, num_tags)

            # Emissions scores
            emissions_score = emissions[i]
            emissions_score[(next_possible_tags == 0)] = IMPOSSIBLE_SCORE
            emissions_score = emissions_score.view(batch_size, 1, num_tags)

            # Transition scores
            transition_scores = self.transitions.view(
                1, num_tags, num_tags
            ).expand(batch_size, num_tags, num_tags).clone()
            transition_scores[(current_possible_tags == 0)] = IMPOSSIBLE_SCORE
            transition_scores.transpose(1, 2)[(next_possible_tags == 0)] =\
                IMPOSSIBLE_SCORE

            # Broadcast alpha
            broadcast_alpha = alpha.view(batch_size, num_tags, 1)

            # Add all scores
            inner = broadcast_alpha + emissions_score + transition_scores # (batch_size, num_tags, num_tags)
            alpha = (log_sum_exp(inner, 1) * mask[i].view(batch_size, 1) +
                     alpha * (1 - mask[i]).view(batch_size, 1))

        # Add end transition score
        last_tag_indexes = mask.sum(0).long() - 1
        end_transitions = (
            self.end_transitions.expand(batch_size, num_tags)
            * possible_tags.transpose(0, 1).view(sequence_length
            * batch_size, num_tags)[
                last_tag_indexes
                + torch.arange(batch_size, device=possible_tags.device)
                * sequence_length
            ]
        )
        end_transitions[(end_transitions == 0)] = IMPOSSIBLE_SCORE
        stops = alpha + end_transitions

        return log_sum_exp(stops) * 1e-2 # (batch_size,)

    def _forward_algorithm(
        self, emissions: torch.Tensor, mask: torch.Tensor,
        reverse_direction: bool=False
    ):
        """Computes the log probabilities.

        Parameters
        ----------
        emissions: ``torch.Tensor``
            (batch_size, sequence_length, num_tags).
        mask:  ``torch.ByteTensor``
            Show padding tags. 0 don't calculate score. (batch_size,
            sequence_length).
        reverse: ``bool``
            This parameter decide algorithm direction.

        Returns
        -------
        log_probabilities: ``torch.Tensor``
            (sequence_length, batch_size, num_tags).
        """
        batch_size, sequence_length, num_tags = emissions.data.shape

        broadcast_emissions = emissions.transpose(0, 1).unsqueeze(2).contiguous() # (sequence_length, batch_size, 1, num_tags)
        mask = mask.float().transpose(0, 1).contiguous()                          # (sequence_length, batch_size)
        broadcast_transitions = self.transitions.unsqueeze(0)                     # (1, num_tags, num_tags)
        sequence_iter = range(1, sequence_length)

        # backward algorithm
        if reverse_direction:
            # Transpose transitions matrix and emissions
            broadcast_transitions = broadcast_transitions.transpose(1, 2)         # (1, num_tags, num_tags)
            broadcast_emissions = broadcast_emissions.transpose(2, 3)             # (sequence_length, batch_size, num_tags, 1)
            sequence_iter = reversed(sequence_iter)

            # It is beta
            log_proba = [self.end_transitions.expand(batch_size, num_tags)]
        # forward algorithm
        else:
            # It is alpha
            log_proba = [emissions.transpose(0, 1)[0] + self.start_transitions.view(1, -1)]

        for i in sequence_iter:
            # Broadcast log probability
            broadcast_log_proba = log_proba[-1].unsqueeze(2) # (batch_size, num_tags, 1)

            # Add all scores
            # inner: (batch_size, num_tags, num_tags)
            # broadcast_log_proba:   (batch_size, num_tags, 1)
            # broadcast_transitions: (1, num_tags, num_tags)
            # broadcast_emissions:   (batch_size, 1, num_tags)
            inner = (
                broadcast_log_proba
                + broadcast_transitions
                + broadcast_emissions[i]
            )
            # Append log proba
            log_proba.append(
                log_sum_exp(inner, 1)
                * mask[i].view(batch_size, 1)
                + log_proba[-1]
                * (1 - mask[i]).view(batch_size, 1)
            )

        if reverse_direction:
            log_proba.reverse()

        return torch.stack(log_proba)

