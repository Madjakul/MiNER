# miner/utils/data/ner_dataset.py

from typing import Union, Optional, List, Dict

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class NER_Dataset(Dataset):
    """Custom Dataset to take care of the training and the inference of the
    NER.

    Parameters
    ----------

    Attributes
    ----------

    Examples
    --------
    """

    def __init__(
        self, lang: str, device: str, max_length: int,
        iterable_corpus: List[str], labels: List[str],
        iterable_labels: Optional[List[str]]=None
    ):
        self.iterable_corpus = iterable_corpus
        self.iterable_labels = iterable_labels
        self.label2idx = {label: idx for idx, label in enumerate(labels)}
        self.label2idx["PAD"] = len(self.label2idx)
        self.label2idx["B-UNK"] = -1
        self.label2idx["I-UNK"] = -1
        if lang == "fr":
            self.tokenizer = AutoTokenizer.from_pretrained(
                "camembert-base",
                add_prefix_space=True
            )
        elif max_length > 512:
            self.tokenizer = AutoTokenizer.from_pretrained(
                "allenai/longformer-base-4096",
                add_prefix_space=True
            )
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(
                "roberta-base",
                add_prefix_space=True
            )
        self.max_length = max_length
        self.device = device
        self.inputs, self.outputs = self._compute_dataset()

    def __call__(self, inputs: List[str]):
        inputs = self._tokenize(inputs)
        return inputs

    def __getitem__(self, idx):
        x = self.inputs[idx]
        y = torch.LongTensor(self.outputs[idx]).to(self.device)
        return x, y

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
        self, inputs: Dict[str, torch.Tensor], labels: List[int]
    ):
        word_ids = inputs.word_ids()                    # type: ignore
        label_ids = []
        for idx, word_idx in enumerate(word_ids):
            if idx == 0:
                label_ids.append(self.label2idx["O"])
            elif word_idx is None:
                label_ids.append(self.label2idx["PAD"])
            elif word_idx != previous_word_idx:         # type: ignore
                label_ids.append(labels[word_idx])
            else:
                label_ids.append(labels[word_idx])
            previous_word_idx = word_idx
        return label_ids

