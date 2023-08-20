# miner/utils/data/ner_dataset.py

from typing import Literal, Union, Optional, List

import numpy as np
import torch
import transformers
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class NER_Dataset(Dataset):
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
        self, lang: Literal["en", "fr"], device: Literal["cpu", "cuda"],
        max_length: int, iterable_corpus: List[str], labels: List[str],
        iterable_labels: Optional[List[str]]=None
    ):
        self.iterable_corpus = iterable_corpus
        self.iterable_labels = iterable_labels
        self.label2idx = {label: idx for idx, label in enumerate(labels)}
        self.label2idx["B-UNK"] = -1
        self.label2idx["I-UNK"] = -1
        if lang == "fr":
            self.tokenizer = AutoTokenizer.from_pretrained(
                "camembert-base",
            )
        elif max_length > 512:
            self.tokenizer = AutoTokenizer.from_pretrained(
                "allenai/longformer-base-4096",
            )
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(
                "roberta-base",
            )
        self.max_length = max_length
        self.device = device
        self.word_ids = []
        self.inputs, self.outputs, self.masks = self._compute_dataset()

    def __getitem__(self, idx):
        x = self.inputs[idx]
        y = self.outputs[idx]
        mask = self.masks[idx]
        return x, y, mask

    def __len__(self):
        if self.inputs is None:
            return 0
        return len(self.inputs)

    def _tokenize(
        self, pretokenized_text: Union[str, List[str], List[List[str]]]
    ):
        inputs = self.tokenizer(
            pretokenized_text,
            is_split_into_words=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return inputs

    def _compute_dataset(self):
        outputs = []
        inputs = []
        masks = []
        for idx, text in enumerate(self.iterable_corpus):
            local_inputs = self._tokenize(text)
            local_outputs = [
                self.label2idx[label] for label in self.iterable_labels[idx]    # type: ignore
            ]
            local_outputs, local_mask = self._align_labels(local_inputs, local_outputs)
            inputs.append(local_inputs)
            outputs.append(local_outputs)
            masks.append(local_mask)
        inputs = torch.Tensor(inputs, dtype=torch.float32, device=self.device)
        outputs = torch.Tensor(outputs, dtype=torch.int64, device=self.device)
        masks = torch.Tensor(masks, dtype=torch.uint8, device=self.device)
        return inputs, outputs, masks

    def _align_labels(
        self, inputs: transformers.BatchEncoding, labels: List[int]
    ):
        word_ids = inputs.word_ids()                    # type: ignore
        self.word_ids.append(word_ids)
        label_ids = []
        local_mask = []
        previous_word_idx = -1
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(self.label2idx["O"])
                local_mask.append(0)
            elif word_idx != previous_word_idx:         # type: ignore
                label_ids.append(labels[word_idx])
                local_mask.append(1)
            else:
                label_ids.append(self.label2idx["O"])
                local_mask.append(0)
            previous_word_idx = word_idx
        return label_ids, local_mask

