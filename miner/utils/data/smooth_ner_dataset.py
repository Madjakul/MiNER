# miner/utils/data/smooth_ner_dataset.py

import math
import random
from typing import Literal, Optional, List

import torch
from torch.utils.data import Dataset
import transformers
from transformers import AutoTokenizer, RobertaForMaskedLM

from miner.modules import PartialNER
from miner.utils.crf_utils import custom_argmax, create_possible_tag_masks


class SmoothNERDataset(Dataset):
    """Custom dataset used to train the smooth NER.

    Parameters
    ----------
    partial_ner: PartialNER
        Previously trained partial NER, used to generate labels.
    corpus: List[List[str]]
        List of word-level segmented texts.
    lm_path: str
        Path to a ``transformers`` pretrained language model.
    max_length: int
        Maximum text length.
    device: str, {"cpu", "cuda"}
        The device where the computations are done.
    labels: List[List[str]], optional

    Attributes
    ----------
    partial_ner: PartialNER
        Previously trained partial NER, used to generate labels.
    corpus: List[List[str]]
        List of word-level segmented texts.
    max_length: int
        Maximum text length.
    device: str, {"cpu", "cuda"}
        The device where the computations are done.
    labels: List[List[str]], optional
        List of word-level labels of ``self.corpus``.
    tokenizer: AutoTokenizer
        "roberta-base" tokenizer from **transformers**.
    lm: RobertaForMaskedLM
        Pre-trained RoBERTa used to perform language augmentation.
    """

    def __init__(
        self, partial_ner: PartialNER, corpus: List[List[str]], lm_path: str,
        max_length: int, device: Literal["cpu", "cuda"],
        labels: Optional[List[List[str]]]=None
    ):
        self.partial_ner = partial_ner
        self.corpus = corpus
        self.labels = labels
        self.max_length = max_length
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(
            "roberta-base", add_prefix_space=True
        )
        self.lm = RobertaForMaskedLM.from_pretrained(lm_path).to(device)


    def __len__(self):
        return len(self.corpus)

    def __getitem__(self, idx: int):
        x = self.tokenize(idx)
        x_augmented = self.augmented_tokenize(x)
        y = self.enhanced_marginal_probabilities(x)
        return x, x_augmented, y

    def tokenize(self, idx: int):
        """Tokenizes a segmented text from ``self.iterable_corpus`` at a given
        index.

        Parameters
        ----------
        idx: int
            Index of the segmented text to tokenize.

        Returns
        -------
        inputs: transformers.BatchEncoding
            Tokenized text.
        """
        inputs = self.tokenizer(
            self.corpus[idx],
            is_split_into_words=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return inputs.to(self.device)

    def augmented_tokenize(self, inputs: transformers.BatchEncoding):
        """Randomly replaces 15% of the tokens in the input sentence with other
        tokens sampled from ``self.lm``'s top 3 output.

        Parameters
        ----------
        inputs: transformers.BatchEncoding
            Original tokenized text

        Returns
        -------
        augmented_inputs: transformers.BatchEncoding
            Augmented tokenized text.
        """
        augmented_inputs = inputs.copy()
        augmented_inputs["input_ids"] = inputs["input_ids"].clone()
        max_idx = (
            inputs["input_ids"][0] != self.tokenizer.pad_token_id
        ).sum() - 2
        nb_token_to_mask = int(math.ceil(.15 * max_idx))
        masked_ids = random.sample(range(1, max_idx + 1), nb_token_to_mask)
        for idx in masked_ids:
            augmented_inputs["input_ids"][0, idx] = \
                self.tokenizer.mask_token_id
        with torch.no_grad():
            logits = self.lm(**augmented_inputs)["logits"]
        top_k_tokens = torch.topk(logits[0, masked_ids], k=3, dim=1).indices
        for idx, position in enumerate(masked_ids):
            replacement_id = top_k_tokens[idx, random.randint(0, 2)]
            augmented_inputs["input_ids"][0, position] = replacement_id
        return augmented_inputs

    def enhanced_marginal_probabilities(
        self, inputs: transformers.BatchEncoding
    ):
        """Enhances high-confidence predictions while demote low-confidence
        ones via squaring and normalizing the current predictions on the
        original sequence [1]_.

        Parameters
        ----------
        inputs: transformers.BatchEncoding
            Tokenzed input text.

        Returns
        -------
        output: torch.FloatTensor
            Float tensor containing the enhanced marginal probabilities of each
            token to belong to a given class. (batch_size, sequence_length,
            num_tags)

        References
        ----------
        ..  [1] Xie, Junyuan, Ross Girshick, and Ali Farhadi. "Unsupervised
                deep embedding for clustering analysis." International
                conference on machine learning. PMLR, 2016.
        """
        tmp_p = self.partial_ner.marginal_probabilities(inputs)
        mask = inputs["attention_mask"].unsqueeze(2).expand(
            tmp_p.shape[0],
            tmp_p.shape[1],
            tmp_p.shape[2]
        )
        p = tmp_p * mask
        squared_p = p ** 2
        denominator = squared_p.sum(dim=2)
        enhanced_p = squared_p / denominator.unsqueeze(2)
        outputs = custom_argmax(enhanced_p, threshold=0.7)
        outputs = create_possible_tag_masks(num_tags=p.shape[2], tags=outputs)
        return outputs.squeeze(0).float()

