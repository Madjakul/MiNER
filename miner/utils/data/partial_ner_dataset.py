# miner/utils/data/ner_dataset.py

import math
import random
from typing import Literal, List

import torch
from torch.utils.data import Dataset
import transformers
from transformers import AutoTokenizer, RobertaForMaskedLM


class PartialNERDataset(Dataset):
    """Custom Dataset used to train the partial NER.

    Parameters
    ----------
    device: str, {"cpu", "cuda"}
        Deveice where the computations are performed.
    do_augment: bool
        If you want to perform language augmentation or not.
    max_length: int
        Maximum sequence length.
    iterable_corpus: List[List[str]]
        Corpus containing lists of segmented texts.
    labels: List[str]
        List of possible labels.
    iterable_labels: List[List[str]]
        Corpus containing lists of labels mapped to the text at the same index
        in ``iterable_corpus``.
    lm_path: str
        Path to a **transformers** pre-trained language model.

    Attributes
    ----------
    device: str, {"cpu", "cuda"}
        Deveice where the computations are performed.
    do_augment: bool
        If you want to perform language augmentation or not.
    max_length: int
        Maximum sequence length.
    iterable_corpus: List[List[str]]
        Corpus containing lists of segmented texts.
    iterable_labels: List[List[str]]
        Corpus containing lists of labels mapped to the text at the same index
        in ``iterable_corpus``.
    label2idx: Dict[str, int]
        Maps the string label to a unique integer id. Also adds a mapping for
        the "UNK" label to -1.
    tokenizer: AutoTokenizer
        "roberta-base" tokenizer from **transformers**.
    lm: RobertaForMaskedLM
        Pre-trained RoBERTa used to perform language augmentation.
    """

    def __init__(
        self, device: Literal["cpu", "cuda"], do_augment: bool,
        max_length: int, iterable_corpus: List[List[str]], labels: List[str],
        iterable_labels: List[List[str]], lm_path: str
    ):
        self.iterable_corpus = iterable_corpus
        self.iterable_labels = iterable_labels
        self.label2idx = {label: idx for idx, label in enumerate(labels)}
        self.label2idx["B-UNK"] = -1
        self.label2idx["I-UNK"] = -1
        self.tokenizer = AutoTokenizer.from_pretrained(
            "roberta-base", add_prefix_space=True
        )
        self.lm = RobertaForMaskedLM.from_pretrained(lm_path).to(device)
        self.max_length = max_length
        self.device = device
        self.do_augment = do_augment

    def __getitem__(self, idx):
        x = self.tokenize(idx)
        y = torch.tensor(
            self.align_labels(x, self.iterable_labels[idx]),
            dtype=torch.int64,
            device=self.device
        )
        if self.do_augment:
            x_augmented = self.augmented_tokenize(x)
            return x, x_augmented, y
        return x, torch.empty((2, 3), dtype=torch.float), y

    def __len__(self):
        return len(self.iterable_corpus)

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
            self.iterable_corpus[idx],
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

    def align_labels(
        self, inputs: transformers.BatchEncoding, labels: List[str]
    ):
        """Align the sub-words with labels. All the sub-words are given the
        the label of the original word. The padding token are given an "O"
        label.

        Parameters
        ----------
        inputs: transformers.BatchEncoding
            Tokenized text.
        labels: List[str]
            Word-level labels of the original text.

        Returns
        -------
        label_ids: List[int]
            Token-level labels of the tokenized text.
        """
        word_ids = inputs.word_ids()    # type: ignore
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(self.label2idx["O"])
            else:
                label_ids.append(self.label2idx[labels[word_idx]])
        return label_ids

