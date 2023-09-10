# miner/utils/data/ner_dataset.py

import math
import random
from typing import Literal, List

import torch
import transformers
from torch.utils.data import Dataset
from transformers import AutoTokenizer, RobertaForMaskedLM


class PartialNERDataset(Dataset):
    """Custom Dataset taiking care of the training and the inference of the
    NER.

    Parameters
    ----------
    lang: ``str``, {"en", "fr"}
        Language of the training data.
    device: ``str``, {"cpu", "cuda"}
        Wether or not to use the GPU for computation.
    max_length: ``int``
        Maximum length of a sentence.
    iterable_corpus: ``list``
        List of tokens per document.
    labels: ``list``
        List of possible labels in natural language.
    iterable_labels: ``list``, ``None``
        List of label per document.

    Attributes
    ----------
    device: ``str``, {"cpu", "cuda"}
        Wether or not to use the GPU for computation.
    max_length: ``int``
        Maximum length of a sentence.
    iterable_corpus: ``list``
        List of tokens per document.
    iterable_labels: ``list``, ``None``
        List of label per document.
    label2idx: ``dict``
        maps each label to a unique integer.
    tokenizer: ``transformers.AutoTokenizer``
        LLM tokenizer.
    inputs: ``list``
        List of dictionaries returned by ``self.tokenizer``.
    outputs: ``list``
        List of labels from ``self.iterable_labels`` expressed as a list of
        integers.
    """

    def __init__(
        self, device: Literal["cpu", "cuda"], do_augment: bool,
        max_length: int, iterable_corpus: List[str], labels: List[str],
        iterable_labels: List[str], lm_path: str
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
        """Replace some tokens in the input sentence.

        Parameters
        ----------

        Returns
        -------
        """
        augmented_inputs = inputs.copy()
        augmented_inputs["input_ids"] = inputs["input_ids"].clone()
        max_idx = (
            inputs["input_ids"][0] != self.tokenizer.pad_token_id
        ).sum() - 2
        nb_token_to_mask = int(math.ceil(.15 * max_idx))
        masked_ids = random.sample(range(1, max_idx + 1), nb_token_to_mask)
        for idx in masked_ids:
            augmented_inputs["input_ids"][0, idx] = self.tokenizer.mask_token_id
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
        word_ids = inputs.word_ids()                    # type: ignore
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(self.label2idx["O"])
            else:
                label_ids.append(self.label2idx[labels[word_idx]])
        return label_ids

