# miner/utils/data/transformer_dataset.py

import logging
from typing import Literal, Union, List

import datasets
from transformers import AutoTokenizer, DataCollatorForLanguageModeling

from miner.modules import CamemBERT, RoBERTa, Longformer


class TransformerDataset():
    """Custom dataset used to pretrain Transformers checkpoints from
    **HuggingFace**.

    Parameters
    ----------
    lang: ``str``, {"en", "fr"}
        Language of the training corpus.
    train_corpus: ``list``
        List of training sentences.
    valid_corpus: ``list``
        List of validation sentences.
    max_length: ``int``
        Maximum sequence length.
    mlm_probability: ``float``
        Proportion of words to mask from the training and validation corpus.

    Attributes
    ----------
    mlm_ds:
        Maps the tokenising function to the **HuggingFace**'s ``datasets``.
    max_length: ``int``
        Maximum sequence length.
    train_corpus: ``list``
        List of training sentences.
    valid_corpus: ``list``
        List of validation sentences.
    tokenizer: ``transformers.AutoTokenizer``
        Object from ``AutoTokenizer``. The object depends on the language
        model used.
    data_collator: ``transformers.DataCollatorForLanguageModeling``
        Data collator to mask a given proportion of word from the corpus before
        returning a tokenized and encoded version of it.
    """

    def __init__(
        self, lang: Literal["en", "fr"], train_corpus: List[str],
        valid_corpus: List[str], max_length: int, mlm_probability: float=0.15
    ):
        self.mlm_ds = None
        self.max_length = max_length
        self.train_corpus = [" ".join(text) for text in train_corpus]
        self.valid_corpus = [" ".join(text) for text in valid_corpus]
        if lang == "fr":
            logging.info(f"Using camembert-base tokenizer")
            self.tokenizer = AutoTokenizer.from_pretrained(
                "camembert-base",
                add_prefix_space=True
            )
        elif max_length > 512:
            logging.info(f"Using allenai/longformer-base-4096 tokenizer")
            self.tokenizer = AutoTokenizer.from_pretrained(
                "allenai/longformer-base-4096",
                add_prefix_space=True
            )
        else:
            logging.info(f"Using roberta-base tokenizer")
            self.tokenizer = AutoTokenizer.from_pretrained(
                "roberta-base",
                add_prefix_space=True
            )
        self._build_mlm_dataset()
        self.data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm_probability=mlm_probability,
            return_tensors="pt"
        )

    def _build_mlm_dataset(self):
        train_ds = {"text": self.train_corpus}
        valid_ds = {"text": self.valid_corpus}
        ds = datasets.DatasetDict({
            "train": datasets.Dataset.from_dict(train_ds),
            "valid": datasets.Dataset.from_dict(valid_ds)
        })
        self.mlm_ds = ds.map(self._tokenize, batched=True)
        self.mlm_ds.remove_columns(["text"])

    def _tokenize(self, batch):
        return self.tokenizer(
            batch["text"],
            truncation=True,
            max_length=self.max_length,
            return_special_tokens_mask=True
        )

    def add_vocab(
        self, corpus: List[List[str]],
        lm: Union[RoBERTa, CamemBERT, Longformer]
    ):
        """Adds new tokens to a pretrained LLM.

        Parameters
        ----------
        corpus: ``list``
            List of tokens per document.
        lm: ``miner.modules.RoBERTa``, ``miner.modules.CamemBERT``, ``miner.modules.Longformer``
            Pretrained large language model.
        """
        new_tokens = [token for text in corpus for token in text]
        new_tokens = set(new_tokens) - set(self.tokenizer.vocab.keys())
        logging.info( f"Adding {len(new_tokens)} new tokens to the vocabulary")
        self.tokenizer.add_tokens(list(new_tokens))
        logging.info("Resizing the Language model")
        lm.model.resize_token_embeddings(len(self.tokenizer))
