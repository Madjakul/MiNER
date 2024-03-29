# miner/utils/data/transformer_dataset.py

import logging
from typing import List

import torch
import datasets
from transformers import AutoTokenizer, DataCollatorForLanguageModeling

from miner.modules import RoBERTa
from miner.utils.data import preprocessing as pp


class TransformerDataset():
    """Custom dataset used to pretrain Transformers checkpoints from
    **HuggingFace**.

    Parameters
    ----------
    train_corpus: List[str]
        List of training texts.
    valid_corpus: List[str]
        List of validation texts.
    max_length: int
        Maximum sequence length.
    mlm_probability: float
        Proportion of words to mask from the training and validation corpus.

    Attributes
    ----------
    mlm_ds:
        Maps the tokenising function to the **HuggingFace**'s ``datasets``.
    max_length: int
        Maximum sequence length.
    train_corpus: list
        List of training sentences.
    valid_corpus: List[str]
        List of validation sentences.
    tokenizer: transformers.AutoTokenizer
        Object from ``AutoTokenizer``. The object depends on the language
        model used.
    data_collator: transformers.DataCollatorForLanguageModeling
        Data collator to mask a given proportion of word from the corpus before
        returning a tokenized and encoded version of it.
    """

    def __init__(
        self, train_corpus: List[str], valid_corpus: List[str],
        max_length: int, mlm_probability: float
    ):
        self.mlm_ds = None
        self.max_length = max_length
        self.train_corpus = train_corpus
        self.valid_corpus = valid_corpus
        logging.info(f"Using roberta-base tokenizer")
        self.tokenizer = AutoTokenizer.from_pretrained(
            "roberta-base",
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
            padding="max_length",
            return_special_tokens_mask=True,
            return_tensors="pt"
        )

    def add_vocab( self, corpus: List[str], lm: RoBERTa):
        """Adds new tokens to a pretrained LLM. The embedding of the added
        tokens are initialized using the mean of the already existing tokens
        plus some noise in order to avoid diverging too much from the initial
        distributions, thus converging faster during pretraining [1]_.

        Parameters
        ----------
        corpus: ``list``
            List of tokens per document.
        lm: miner.modules.RoBERTa
            Pretrained large language model.

        References
        ----------
        ..  [1] Hewitt John. 2021. Initializing new word embeddings for
            pretrained language models. (2021). Retrieved April 24, 2023 from
            https://nlp.stanford.edu/~johnhew/vocab-expansion.html
        """
        new_tokens = [
            token for text in corpus for token in pp.escape(text).split()
        ]
        new_tokens = set(new_tokens) - set(self.tokenizer.vocab.keys()) # New tokens don't already exist
        logging.info( f"Adding {len(new_tokens)} new tokens to the vocabulary")
        self.tokenizer.add_tokens(list(new_tokens))
        logging.info("Resizing the Language model")
        lm.model.resize_token_embeddings(len(self.tokenizer))
        # Computing the distribution of the new embeddings
        params = lm.model.state_dict()
        # embeddings = params["transformer.wte.weight"]
        embeddings_key = "roberta.embeddings.word_embeddings.weight"
        embeddings = params[embeddings_key]
        pre_expansion_embeddings = embeddings[:-3, :]
        mu = torch.mean(pre_expansion_embeddings, dim=0)
        n = pre_expansion_embeddings.size()[0]
        sigma = (
            (pre_expansion_embeddings - mu).T @ (pre_expansion_embeddings - mu)
        ) / n
        dist = torch.distributions.multivariate_normal.MultivariateNormal(
            mu,
            covariance_matrix=1e-5*sigma
        )
        # Loading the new embeddings in the model
        new_embeddings = torch.stack(
            tuple((dist.sample() for _ in range(3))),
            dim=0
        )
        embeddings[-3:, :] = new_embeddings
        params[embeddings_key][-3:, :] = new_embeddings
        lm.model.load_state_dict(params)

