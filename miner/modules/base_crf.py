# miner/modules/base_crf.py

from abc import abstractmethod
from typing import Optional, Literal, List

import torch
import torch.nn as nn

from miner.utils import IMPOSSIBLE_SCORE


class BaseCRF(nn.Module):
    """Abstract method for the conditional random field (CRF) [1]_ .

    Parameters
    ----------
    num_tags: int
        Number of possible tags (counting the padding one if needed).
    padding_idx: int, optional
        Padding index.
    device: str, {"cpu", "cuda"}
        Wether to do computation on GPU or CPU.

    Attributes
    ----------
    num_tags: int
        Number of possible tags (counting the padding one if needed).
    start_transitions: torch.nn.Parameter
        Begining scores of the transition matrix. Initialized with values
        values sampled from a uniform distribution in [-1; 1]. (num_tags).
    device: str, {"cpu", "cuda"}
        Wether to do computation on GPU or CPU.
    end_transitions: torch.nn.Parameter
        Ending scores of the transition matrix. Initialized with values
        values sampled from a uniform distribution in [-1; 1]. (num_tags).
    transitions: torch.nn.Parameter
        Transition matrix. Initialized using xavier [2]_'s method. Values are
        sampled from a uniform distribution in [-1; 1]. (num_tags, num_tags).

    References
    ----------
    ..  [1] Lafferty, John, Andrew McCallum, and Fernando CN Pereira.
            "Conditional random fields: Probabilistic models for segmenting and
            labeling sequence data." (2001).
    ..  [2] Glorot, Xavier, and Yoshua Bengio. "Understanding the difficulty of
            training deep feedforward neural networks." Proceedings of the
            thirteenth international conference on artificial intelligence and
            statistics. JMLR Workshop and Conference Proceedings, 2010.
    """
    def __init__(
        self, num_tags: int, device: Literal["cpu", "cuda"],
        padding_idx: Optional[int]=None
    ) -> None:
        super(BaseCRF, self).__init__()
        self.device = device
        self.num_tags = num_tags
        self.start_transitions = nn.Parameter(
            nn.init.uniform_(
                torch.empty(num_tags, device=self.device), -1.0, 1.0
            )
        )
        self.end_transitions = nn.Parameter(
            nn.init.uniform_(
                torch.randn(num_tags, device=self.device), -1.0, 1.0
            )
        )
        init_transition = torch.empty(num_tags, num_tags, device=self.device)
        if padding_idx is not None:
            init_transition[:, padding_idx] = IMPOSSIBLE_SCORE
            init_transition[padding_idx, :] = IMPOSSIBLE_SCORE
        self.transitions = nn.Parameter(
            nn.init.xavier_uniform_(init_transition)
        )

    @abstractmethod
    def forward(
        self, emissions: torch.FloatTensor, tags: torch.LongTensor,
        mask: Optional[torch.ByteTensor]=None
    ) -> torch.FloatTensor:
        raise NotImplementedError()

    def _forward_algorithm(
        self, emissions: torch.FloatTensor, mask: torch.ByteTensor,
        reverse_direction: bool=False
    ) -> torch.FloatTensor:
        """Computes the logarithm of the unary/emission scores of each token
        plus their transition score. Despite its name, this function is used to
        compute the `forward-backward algorithm https://en.wikipedia.org/wiki/Forward%E2%80%93backward_algorithm`__.

        Parameters
        ----------
        emissions: torch.FloatTensor
            Unary/emission score of each tokens.
            (batch_size, sequence_length, num_tags).
        mask: torch.ByteTensor
            Masked used to to discard subwords, special tokens or padding from
            being added to the log-probability. (batch_size, sequence_length).
        reverse_direction: bool, default=False
            ``True`` if you want to use the backward algorithm. ``False`` if
            you want to use the forward algorithm.

        Returns
        -------
        torch.FloatTensor
            Log-scores for each token. (sequence_length, batch_size, num_tags).
        """
        batch_size, sequence_length, num_tags = emissions.data.shape
        broadcast_emissions = \
            emissions.transpose(0, 1).unsqueeze(2).contiguous()             # (sequence_length, batch_size, 1, num_tags)
        mask = mask.float().transpose(0, 1).contiguous()                    # (sequence_length, batch_size) # type: ignore
        broadcast_transitions = self.transitions.unsqueeze(0)               # (1, num_tags, num_tags)
        sequence_iter = range(1, sequence_length)
        # backward algorithm
        if reverse_direction:
            # Transpose transitions matrix and emissions
            broadcast_transitions = broadcast_transitions.transpose(1, 2)   # (1, num_tags, num_tags)
            broadcast_emissions = broadcast_emissions.transpose(2, 3)       # (sequence_length, batch_size, num_tags, 1)
            sequence_iter = reversed(sequence_iter)
            # It is beta
            log_proba = [self.end_transitions.expand(batch_size, num_tags)] # [(batch_size, num_tags)]
        # forward algorithm
        else:
            # It is alpha
            log_proba = [                                                   # [(batch_size, num_tags)]
                emissions.transpose(0, 1).contiguous()[0]
                + self.start_transitions.unsqueeze(0)
            ]
        for i in sequence_iter:
            # Broadcast log probability
            broadcast_log_proba = log_proba[-1].unsqueeze(2)                # (batch_size, num_tags, 1)
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
                torch.logsumexp(inner, dim=1) * mask[i].unsqueeze(1)
                + log_proba[-1] * (1 - mask[i]).unsqueeze(1)
            )
        if reverse_direction:
            log_proba.reverse()
        return torch.stack(log_proba)                                       # type: ignore

    def marginal_probabilities(
        self, emissions: torch.FloatTensor,
        mask: Optional[torch.ByteTensor]=None
    ) -> torch.FloatTensor:
        """Computes the probability of each token.

        Parameters
        ----------
        emissions: torch.FloatTensor
            Unary/emission score of each tokens.
            (batch_size, sequence_length, num_tags).
        mask: torch.ByteTensor, optional
            Masked used to to discard subwords, special tokens or padding from
            being added to the log-probability. (batch_size, sequence_length).

        Returns
        -------
        torch.FloatTensor
            Marginal probability of each token to belong to a given class.
            (sequence_length, sequence_length, num_tags).
        """
        if mask is None:
            batch_size, sequence_length, _ = emissions.data.shape
            mask = torch.ones(                  # type: ignore
                [batch_size, sequence_length],
                dtype=torch.uint8,
                device=self.device
            )
        alpha = self._forward_algorithm(        # (sequence_length, batch_size, num_tags)
            emissions,
            mask,                               # type: ignore
            reverse_direction=False
        )
        beta = self._forward_algorithm(         # (sequence_length, batch_size, num_tags)
            emissions,
            mask,                               # type: ignore
            reverse_direction=True
        )
        z = torch.logsumexp(                    # (batch_size)
            alpha[alpha.size(0) - 1] + self.end_transitions,
            dim=1
        )
        proba = alpha + beta - z.view(1, -1, 1) # (sequence_length, batch_size, num_tags)
        return torch.exp(proba)                 # (sequence_length, batch_size, num_tags) # type: ignore

    def viterbi_decode(
        self, emissions: torch.Tensor, mask: Optional[torch.ByteTensor]=None
    ) -> List[List[int]]:
        """
        Dynamically computes the best sequence of tags.

        Parameters
        ----------
        emissions: torch.FloatTensor
            Unary/emission score of each tokens.
            (batch_size, sequence_length, num_tags).
        mask: torch.ByteTensor, optional
            Masked used to to discard subwords, special tokens or padding from
            being added to the log-probability. (batch_size, sequence_length).

        Returns
        -------
        best_tags_list: List[int]
            Best sequence of tag for each sequence in the batch.
            (batch_size, ``torch.where(mask.shape[i]==1)``).
        """
        batch_size, sequence_length, _ = emissions.shape
        if mask is None:
            mask = torch.ones(                                                  # type: ignore
                [batch_size, sequence_length],
                dtype=torch.uint8,
                device=self.device
            )
        emissions = emissions.transpose(0, 1).contiguous()
        mask = mask.transpose(0, 1).contiguous()                                # type: ignore
        # Start transition and first emission score
        score = self.start_transitions + emissions[0]
        history = []
        for i in range(1, sequence_length):
            broadcast_score = score.unsqueeze(2)
            broadcast_emissions = emissions[i].unsqueeze(1)
            next_score = \
                broadcast_score + self.transitions + broadcast_emissions
            next_score, indices = next_score.max(dim=1)
            score = torch.where(mask[i].unsqueeze(1) == 1, next_score, score)   # type: ignore
            history.append(indices)
        # Add end transition score
        score += self.end_transitions
        # Compute the best path
        seq_ends = mask.sum(dim=0) - 1                                          # type: ignore
        best_tags_list = []
        for i in range(batch_size):
            _, best_last_tag = score[i].max(dim=0)
            best_tags = [best_last_tag.item()]
            for hist in reversed(history[:seq_ends[i]]):
                best_last_tag = hist[i][best_tags[-1]]
                best_tags.append(best_last_tag.item())
            best_tags.reverse()
            best_tags_list.append(best_tags)
        return best_tags_list

