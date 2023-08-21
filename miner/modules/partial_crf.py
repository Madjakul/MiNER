# miner/modules/partial_crf.py

from typing import Optional, Literal

import torch

from miner.modules.base_crf import BaseCRF
from miner.utils import IMPOSSIBLE_SCORE, create_possible_tag_masks


class PartialCRF(BaseCRF):
    """Partial/Fuzzy Conditional random field.

    Parameters
    ----------
    q: float, optional
        Hyperparameter used to modify the generalized cross-entropy.

    Attributes
    ----------
    q: float, default=0.7
        Hyperparameter used to modify the generalized cross-entropy.
    """

    def __init__(
        self, num_tags: int, device: Literal["cpu", "cuda"],
        q: Optional[float]=None, padding_idx: Optional[int]=None
    ) -> None:
        super().__init__(num_tags, device, padding_idx)
        self.q = q if q is not None else 0.7

    def _numerator_score(
        self, emissions: torch.FloatTensor, mask: torch.ByteTensor,
        possible_tags: torch.ByteTensor,
    ) -> torch.FloatTensor:
        """
        Computes the log of the emission/unary score plus the transition score
        for the whole sequence.

        Parameters
        ----------
        emissions: torch.FloatTensor
            Unary/emission score of each tokens.
            (batch_size, sequence_length, num_tags).
        mask: torch.ByteTensor
            Masked used to to discard subwords, special tokens or padding from
            being added to the log-probability. (batch_size, sequence_length).
        possible_tags: torch.ByteTensor
            Mask corresponding to the target label(s).
            (batch_size, sequence_length, num_tags).

        Returns
        -------
        torch.FloatTensor
            Log probability of the emission/unary score plus the transition
            score for the whole sequence. (batch_size,)
        """
        batch_size, sequence_length, num_tags = emissions.data.shape
        emissions = emissions.transpose(0, 1).contiguous()                  # type: ignore
        mask = mask.float().transpose(0, 1).contiguous()                    # type: ignore
        possible_tags = possible_tags.float().transpose(0, 1)               # type: ignore

        # Start transition score and first emission
        first_possible_tag = possible_tags[0]
        alpha = self.start_transitions + emissions[0]                       # (batch_size, num_tags)
        alpha[(first_possible_tag == 0)] = IMPOSSIBLE_SCORE

        for i in range(1, sequence_length):
            current_possible_tags = possible_tags[i-1]                      # (batch_size, num_tags)
            next_possible_tags = possible_tags[i]                           # (batch_size, num_tags)

            # Emissions scores
            emissions_score = emissions[i]
            emissions_score[(next_possible_tags == 0)] = IMPOSSIBLE_SCORE
            emissions_score = emissions_score.view(batch_size, 1, num_tags)

            # Transition scores
            transition_scores = self.transitions.unsqueeze(0).expand(
                batch_size, num_tags, num_tags
            ).clone()
            transition_scores[(current_possible_tags == 0)] = IMPOSSIBLE_SCORE
            transition_scores.transpose(1, 2)[(next_possible_tags == 0)] = IMPOSSIBLE_SCORE

            # Broadcast alpha
            broadcast_alpha = alpha.unsqueeze(2)

            # Add all scores
            inner = broadcast_alpha + emissions_score + transition_scores   # (batch_size, num_tags, num_tags)
            alpha = (
                torch.logsumexp(inner, 1) * mask[i].unsqueeze(1)
                + alpha * (1 - mask[i]).unsqueeze(1)
            )

        # Add end transition score
        last_tag_indexes = mask.sum(0).long() - 1
        end_transitions = (
            self.end_transitions.expand(batch_size, num_tags)
            * possible_tags.transpose(0, 1).view(
                sequence_length * batch_size, num_tags
            )[
                last_tag_indexes
                + torch.arange(batch_size, device=possible_tags.device)
                * sequence_length
            ]
        )
        end_transitions[(end_transitions == 0)] = IMPOSSIBLE_SCORE
        stops = alpha + end_transitions                                     # (batch_size, num_tags)
        print(stops)
        return torch.logsumexp(stops, 1)                                    # (batch_size,) # type: ignore

    def _denominator_score(
        self, emissions: torch.FloatTensor, mask: torch.ByteTensor,
    ) -> torch.FloatTensor:
        """
        Computes the log-partition score for the whole sequence.

        Parameters
        ----------
        emissions: torch.FloatTensor
            Unary/emission score of each tokens.
            (batch_size, sequence_length, num_tags).
        mask: torch.ByteTensor
            Masked used to to discard subwords, special tokens or padding from
            being added to the log-probability. (batch_size, sequence_length).

        Returns
        -------
        torch.FloatTensor
            Log-partition score. (batch_size,)
        """
        _, sequence_length, num_tags = emissions.data.shape
        emissions = emissions.transpose(0, 1).contiguous()                  # type: ignore
        mask = mask.float().transpose(0, 1).contiguous()                    # type: ignore
        # Start transition score and first emissions score
        alpha = self.start_transitions.view(1, num_tags) + emissions[0]

        for i in range(1, sequence_length):
            emissions_score = emissions[i].unsqueeze(1)                     # (batch_size, 1, num_tags)
            transition_scores = self.transitions.unsqueeze(0)               # (1, num_tags, num_tags)
            broadcast_alpha = alpha.unsqueeze(2)                            # (batch_size, num_tags, 1)
            inner = broadcast_alpha + emissions_score + transition_scores   # (batch_size, num_tags, num_tags)
            alpha = (
                torch.logsumexp(inner, 1) * mask[i].unsqueeze(1)
                + alpha * (1 - mask[i]).unsqueeze(1)
            )

        # Add end transition score
        stops = alpha + self.end_transitions.unsqueeze(0)
        return torch.logsumexp(stops, 1)                                    # (batch_size,) # type: ignore

    def forward(
        self, emissions: torch.FloatTensor, tags: torch.LongTensor,
        loss_fn: Literal["nll", "c_nll", "gce"]="nll",
        mask: Optional[torch.ByteTensor]=None
    ) -> torch.FloatTensor:
        """Performs the forward pass depending on the loss function chosen: the
        classic negative log-likelihood, the corrected negative log-likelihood
        where the the negative log-unlikelihood [1]_ is computed and used as a
        regularizer [2]_ and the generelized cross-entropy [3]_ [4]_.

        Parameters
        ----------
        emissions: torch.FloatTensor
            Unary/emission score of each tokens.
            (batch_size, sequence_length, num_tags).
        tags: torch.LongTensor
            Tensor containing the target labels. (batch_size, sequence_length).
        loss_fn: str, {"nll", "c_nll", "gce"}, default="nll"
            Loss function to use: "nll" for negative log-likelihood, "c_nll"
            for corrected negative log-likelihood or "gce" for generalized
            cross-entropy.
        mask: torch.ByteTensor, optional
            Masked used to to discard subwords, special tokens or padding from
            being added to the log-probability. (batch_size, sequence_length).

        Returns
        -------
        torch.FloatTensor
            Mean of the losses over the mini-batch. (0,)

        References
        ----------
        ..  [1] Welleck, Sean, et al. "Neural text generation with unlikelihood
                training." arXiv preprint arXiv:1908.04319 (2019).
        ..  [2] Jiang, Haoming, et al. "Named entity recognition with small
                strongly labeled and large weakly labeled data." arXiv preprint
                arXiv:2106.08977 (2021).
        ..  [3] Zhang, Zhilu, and Mert Sabuncu. "Generalized cross entropy loss
                for training deep neural networks with noisy labels." Advances
                in neural information processing systems 31 (2018).
        ..  [4] Meng, Yu, et al. "Distantly-supervised named entity recognition
                with noise-robust learning and language model augmented
                self-training." arXiv preprint arXiv:2109.05003 (2021).
        """
        if mask is None:
            mask = torch.ones_like(tags, dtype=torch.uint8, device=self.device) # type: ignore
        possible_tags = create_possible_tag_masks(self.num_tags, tags)          # (batch_size, sequence_length, num_tags)
        # If you want NLL
        if loss_fn == "nll":
            gold_score = self._numerator_score(emissions, mask, possible_tags)  # (batch_size,) # type: ignore
            forward_score = self._denominator_score(emissions, mask)            # (batch_size,) # type: ignore
            print(gold_score)
            print(forward_score)
            nll = forward_score - gold_score                                    # (batch_size,)
            return torch.mean(nll)                                              # Mean instead of sum # type: ignore
        pred = self.marginal_probabilities(emissions, mask).transpose(0, 1)     # (batch_size, sequence_length, num_tags)
        batch_size, sequence_length, num_tags = pred.shape
        p_mask = (
            mask.unsqueeze(2).expand(batch_size, sequence_length, num_tags)     # type: ignore
            != 0.
        )
        # If you want corrected NLL
        if loss_fn == "c_nll":
            gold_score = self._numerator_score(emissions, mask, possible_tags)  # (batch_size,) # type: ignore
            forward_score = self._denominator_score(emissions, mask)            # (batch_size,) # type: ignore
            nll = forward_score - gold_score                                    # (batch_size,)
            nlu = -(1 - (-nll).exp()).log()
            nlu[torch.isnan(nlu) | torch.isinf(nlu)].fill_(1e-4)
            weights = []
            weights_bar = []
            for sequence in pred:
                p = torch.masked_select(pred, p_mask)                           # (possible_tags==1,)
                hist = torch.histc(sequence, bins=20, min=0., max=1.)
                hist_mask = hist > 0
                hist = hist[hist_mask]
                max_weight = torch.max(hist)
                local_weights = (max_weight - hist) / max_weight
                weights.append(torch.sum(local_weights))
                weights_bar.append(torch.sum(1 - local_weights))
            weights = torch.stack(weights)
            weights_bar = torch.stack(weights_bar)
            c_nll = weights * nll + weights_bar * nlu
            return torch.mean(c_nll)                                            # type: ignore
        # If you want GCE
        if loss_fn == "gce":
            p = torch.masked_select(pred, p_mask)                               # (possible_tags==1,)
            gce = (1 - p**self.q) / self.q
            gce = torch.mean(gce)                                               # (loss.view(-1)*weights).sum() / weights.sum()
            return gce                                                          # type: ignore
        raise ValueError(f"Invalid loss function: {loss_fn}")

