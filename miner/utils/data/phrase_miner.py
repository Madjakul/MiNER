# miner/utils/data/phrase_miner.py

import re
import logging
from typing import Literal,List, Dict
from collections import defaultdict

from spacy.lang.fr import French
from spacy.lang.en import English
from spacy.tokens.span import Span
from spacy.tokens.token import Token

import miner.utils.data.preprocessing as pp

class PhraseMiner():
    """Abstract class wrapping **SpaCy** functions for tokenization and entity
    ruling. This class also holds the methods to mine potential important
    propostions.

    Notes
    -----
    The **SpaCy** tokennizer only takes care of english or french languages.

    Parameters
    ----------
    lang: ``str``, {"en", "fr"}
        Language of the corpus. `fr` for french or `en` for english.

    Attributes
    ----------
    nlp: ``spacy.lang.en.English``, ``spacy.lang.fr.French``
        **SpaCy** tokenizer to use.
    ruler: ``spacy.pipeline.entity_ruler.EntityRuler``
        **SpaCy**'s entity ruler object to perform string matching.
    n_grams: ``defaultdict(lambda: defaultdict(int))``
        Defaultdict to perform principal propositions mining. The first set of
        keys represent the length of the propositon. For each length, there are
        keys representing their frequency in a given corpus. `{1: {"Hello": 2,
        "world": 2}, 2: {"Hello World": 1}}`.
    """

    def __init__(self, lang: Literal["en", "fr"]):
        self.nlp = English() if lang == "en" else French()
        self.ruler = self.nlp.add_pipe("entity_ruler")
        self.n_grams = defaultdict(lambda: defaultdict(int))

    def __is_beginning(self, token: Token):
        if token.ent_iob_ == "O" and token.text.lower() not in pp.STOP_WORDS_EN \
                and token.text.lower() not in pp.STOP_WORDS_EN \
                and not token.is_punct and not token.is_digit:
            return True
        return False

    def __is_ending(self, token: Token):
        if token.ent_iob_ != "O" or token.text.lower() in pp.STOP_WORDS_EN \
                or token.text.lower() in pp.STOP_WORDS_FR or token.is_punct \
                or token.is_digit:
            return True
        return False

    def _get_ngrams(self, span: Span):
        for n in range(1, len(span)):
            for b_idx in range(len(span) - n + 1):
                if span[b_idx:b_idx + n].text != " ":
                    self.n_grams[n][span[b_idx:b_idx + n].text] += 1

    def _remove_infrequent_ngrams(self):
        for n in self.n_grams:
            self.n_grams[n] = { # type: ignore
                span: freq \
                    for span, freq in self.n_grams[n].items() if freq >= 3
            }

    def _update_frequencies(self, n: int, span: str, freq: int):
        for lower_n in self.n_grams:
            if lower_n >= n:
                continue
            else:
                for new_span in self.n_grams[lower_n]:
                    if len(new_span.split()) < 1: continue
                    b = new_span.split()[0]
                    e = new_span.split()[-1]
                    if bool(set(span.split()) & set([b, e])):
                        self.n_grams[lower_n][new_span] -= freq

    def _remove_redundant_ngrams(self):
        for n in self.n_grams:
            for span, freq in self.n_grams[n].items():
                self._update_frequencies(n, span, freq)

    def get_unk_gazetteers(self, corpus: List[str]):
        """Unsupervised principal propositions mining [1]_.

        Parameters
        ----------
        corpus: ``list``
            List of entries in natural language.

        Returns
        -------
        patterns: ``list``
            List of principal propositions for the given corpus.

        References
        ----------
        ..  [1] Ellie Small and Javier Cabrera. 2022. Principal phrase mining.
            (October 2022). Retrieved January 31, 2023 from
            https://arxiv.org/abs/2206.13748
        """
        for text in corpus:
            escaped_text = pp.escape(text).lower()
            doc = self.nlp(escaped_text)
            # doc = pp.tokenize(self.nlp, text)
            b_idx = -1
            for idx, token in enumerate(doc):
                if b_idx == -1 and self.__is_beginning(token):
                    b_idx = idx
                elif b_idx != -1 and self.__is_ending(token):
                    span = doc[b_idx:idx]
                    self._get_ngrams(span)
                    b_idx = -1
        logging.info("Removing infrequent words...")
        self._remove_infrequent_ngrams()
        logging.info("Removing redundants words...")
        self._remove_redundant_ngrams()
        logging.info("Removing infrequent words...")
        self._remove_infrequent_ngrams()
        patterns = []
        for n_grams in self.n_grams.values():
            patterns.extend([n_gram for n_gram in n_grams.keys()])
        patterns = list(filter(lambda x: not x.isdigit(), patterns))
        patterns = list(
            filter(lambda x: not re.match(r"^[_\W]+$", x), patterns)
        )
        return patterns

    def compute_patterns(self, gazetteers: Dict[str, List[str]]):
        """Adds each gazetteer from a given list to the **SpaCy**'s entity
        ruler in order to retrieve them from any given text in an IOB format
        [2]_.

        Parameters
        ----------
        gazetteers: ``dict``
            Dictionary. `{"name_of_file": ["list", "of", "entries"]}`.

        References
        ----------
        .. [2] Lance A. Ramshaw and Mitchell P. Marcus. 1995. Text chunking
            using transformation-based learning. (May 1995). Retrieved November
            7, 2022 from https://arxiv.org/abs/cmp-lg/9505040
        """
        patterns = []
        for label, entries in gazetteers.items():
            logging.info(f"Adding {len(entries)} entries to {label}")
            for entry in entries:
                escaped_text = pp.escape(entry).lower()
                # pattern = self.nlp(escaped_text)
                # if len(pattern) > 1: # If there are more than one token in the entry
                #     patterns.append({
                #         "label": label,
                #         "pattern": [{"LOWER": token.text} for token in pattern]
                #     })
                # else:
                #     print(pattern.text)
                patterns.append({
                    "label": label,
                    "pattern": escaped_text
                })
        self.ruler.add_patterns(patterns)

    def dump(self, corpus: List[str], path: str):
        """Dumps weakly annotated data in a `conll` file.

        Parameters
        ----------
        corpus: ``list``
            List of text.
        path: ``str``
            Path to the dumped file.
        """
        with open(path, "w", encoding="utf-8") as f:
            for text in corpus:
                doc = self.nlp(pp.escape(text).lower())
                for token in doc:
                    if token.ent_iob_ == "O":
                        f.write(
                            f"{token.text}\t{token.ent_iob_}\n"
                        )
                    else:
                        f.write(
                            f"{token.text}\t{token.ent_iob_}-{token.ent_type_}\n"
                        )
                f.write("\n")

