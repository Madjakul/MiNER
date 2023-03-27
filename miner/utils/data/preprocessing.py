# miner/utils/data/preprocessing.py

import os
import re
from typing import Union

from spacy.tokens import Doc
from spacy.lang.fr import French
from spacy.lang.en import English


def load_gazetteers(path: str):
    """Loads every files in a given directory and treats each of them as a
    dictionary's key.

    Parameters
    ----------
    path: ``str``
        Path to the directory containing the dictionaries.

    Returns
    -------
    gazetteers: ``dict``
        Dictionary with format {"name_of_file": ["list", "of", "entries"]}.

    Warnings
    --------
    Make sure to have `txt` files as dictionaries with explicit names. One line
    corresponds to one entry. The tagging made afterward is case insensitive.
    """
    gazetteers = {}
    for file in os.listdir(path):
        with open(os.path.join(path, file), "r", encoding="utf-8") as f:
            gazetteers[file.rstrip(".txt")] = list(
                set(f.read().lower().splitlines()) # Making sure each entry is unique
            )
    return gazetteers

def tokenize(nlp: Union[English, French], text: str):
    escaped_text = re.sub(
        r"([^A-Za-zÀ-ÖØ-öø-ÿ0-9\s]+)", r" \1 ", text
    ).split()
    return Doc(nlp.vocab, escaped_text)

def read_conll(path: str):
    """Reads a ``conll`` file and returns a tuple containing the list of tokens
    per doc and tags epr doc.

    Parameters
    ----------
    path: ``str``
        Path to the conll file.

    Returns
    -------
    token_docs: ``list``
        List of tokens per document.
    tag_docs: ``list``
        List of labels per document.
    """
    with open(path, "r", encoding="utf-8") as f:
        raw_text = f.read().strip()

    raw_docs = re.split(r"\n\t?\n", raw_text)
    token_docs = []
    tag_docs = []
    for doc in raw_docs:
        tokens = []
        tags = []
        for line in doc.split("\n"):
            token, tag = line.split("\t")
            tokens.append(token)
            tags.append(tag)
        token_docs.append(tokens)
        tag_docs.append(tags)
    return token_docs, tag_docs

