# miner/utils/data/ner_dataset.py

from typing import Literal, Union, Optional, List

import numpy as np
import torch
import transformers
from torch.utils.data import Dataset
from transformers import AutoTokenizer


UNLABELED_INDEX = -1
IMPOSSIBLE_SCORE = -100


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
        # self.label2idx["PAD"] = len(self.label2idx)
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
        self.inputs, self.outputs = self._compute_dataset()

    def __getitem__(self, idx):
        x = self.inputs[idx]
        y = torch.LongTensor(self.outputs[idx], device=self.device)
        z = np.nan_to_num(
            np.array(self.word_ids[idx], dtype=np.float32),
            nan=-1
        )
        z = torch.LongTensor(z, device=self.device)
        return x, y, z

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
        for idx, text in enumerate(self.iterable_corpus):
            local_inputs = self._tokenize(text)
            local_outputs = [
                self.label2idx[label] for label in self.iterable_labels[idx]    # type: ignore
            ]
            local_outputs = self._align_labels(local_inputs, local_outputs)
            inputs.append(local_inputs)
            outputs.append(local_outputs)
        return inputs, outputs

    def _align_labels(
        self, inputs: transformers.BatchEncoding, labels: List[int]
    ):
        word_ids = inputs.word_ids()                    # type: ignore
        self.word_ids.append(word_ids)
        label_ids = []
        previous_word_idx = -1
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(IMPOSSIBLE_SCORE)
            elif word_idx != previous_word_idx:         # type: ignore
                label_ids.append(labels[word_idx])
            else:
                label_ids.append(IMPOSSIBLE_SCORE)
            previous_word_idx = word_idx
        return label_ids

    @staticmethod
    def create_possible_tag_masks(num_tags: int, tags: torch.ByteTensor):
        """Transforms a vector of single integers representing the labels into a
        multi-class binary matrix.

        Parameters
        ----------
        num_tags: ``int``
            Number of labels.
        tags: ``torch.Tensor``
            Vector of integers.

        Returns
        -------
        masks: ``torch.Tensor``
            Multi-class binary matrix.
        """
        copy_tags = tags.clone()
        no_annotation_idx = (copy_tags == UNLABELED_INDEX)
        copy_tags[copy_tags == UNLABELED_INDEX] = 0

        tags_ = torch.unsqueeze(copy_tags, 2)
        masks = torch.zeros(
            tags_.size(0),
            tags_.size(1),
            num_tags,
            dtype=torch.uint8,
            device=tags.device
        )
        masks.scatter_(2, tags_, 1)
        masks[no_annotation_idx] = 1
        return masks

