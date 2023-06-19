import spacy
from spacy import tokens
from typing import List, Any, Union
from spacy.util import DummyTokenizer
from spacy.lang.en import English
from spacy.tokens import Doc
from spacy.vocab import Vocab


def flatten(raw: List[List[Any]]) -> List[Any]:
    """ Turns [['I', 'like', 'cookies', '.'], ['Do', 'you', '?']] to ['I', 'like', 'cookies', '.', 'Do', 'you', '?']"""
    return [tok for sent in raw for tok in sent]


class PreTokenizedPreSentencizedTokenizer(DummyTokenizer):
    """Custom tokenizer to be used in spaCy when the text is already pretokenized."""

    def __init__(self, vocab: spacy.vocab.Vocab):
        """Initialize tokenizer with a given vocab
        :param vocab: an existing vocabulary (see https://spacy.io/api/vocab)
        """
        self.vocab = vocab

    def __call__(self, inp: Union[List[str], str, List[List[str]]]) -> tokens.Doc:
        """Call the tokenizer on input `inp`.
        :param inp: either a string to be split on whitespace, or a list of tokens
        :return: the created Doc object
        """
        if isinstance(inp, str):
            words = inp.split()
            spaces = [True] * (len(words) - 1) + ([True] if inp[-1].isspace() else [False])
            return tokens.Doc(self.vocab, words=words, spaces=spaces)
        elif isinstance(inp, list):
            # Check if we have a flat list or a list of list
            if len(inp) == 0:
                return tokens.Doc(self.vocab, words=inp)
            if isinstance(inp[0], str):
                return tokens.Doc(self.vocab, words=inp)
            elif isinstance(inp[0], list):
                sent_starts = flatten([[1] + [0] * (len(sent) - 1) for sent in inp])
                return tokens.Doc(self.vocab, words=flatten(inp), sent_starts=sent_starts)
        else:
            raise ValueError("Unexpected input format. Expected string, or list of tokens, or list of list of string.")

class SpacyPretokenizedTokenizer:
    """Custom tokenizer to be used in spaCy when the text is already pretokenized."""

    def __init__(self, vocab: Vocab):
        """Initialize tokenizer with a given vocab
        :param vocab: an existing vocabulary (see https://spacy.io/api/vocab)
        """
        self.vocab = vocab

    def __call__(self, inp: Union[List[str], str]) -> Doc:
        """Call the tokenizer on input `inp`.
        :param inp: either a string to be split on whitespace, or a list of tokens
        :return: the created Doc object
        """
        if isinstance(inp, str):
            words = inp.split()
            spaces = [True] * (len(words) - 1) + ([True] if inp[-1].isspace() else [False])
            return Doc(self.vocab, words=words, spaces=spaces)
        elif isinstance(inp, list):
            return Doc(self.vocab, words=inp)
        else:
            raise ValueError("Unexpected input format. Expected string to be split on whitespace, or list of tokens.")
# Normally load spacy NLP

nlp = English()
nlp.tokenizer = SpacyPretokenizedTokenizer(nlp.vocab)
sents = [['I', 'like', 'cookies', '.'], ['Do', 'you', '?']]

doc = nlp(sents)
print(list(doc.sents).__len__())
print(doc)
