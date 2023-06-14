# experiments/benchmark_kb.py

import logging

import wandb
from seqeval.scheme import IOB2
from seqeval.metrics import classification_report
from seqeval.metrics import f1_score, precision_score, recall_score

from miner.utils.data import PhraseMiner
from miner.utils.data import preprocessing as pp


CORPUS_LIST = [
    "./data/bc5cdr/cdr_test_corpus.txt",
    "./data/ncbi_disease/ncbi_test_corpus.txt",
    "./data/wikigold/wiki_test_corpus.txt"
]
GOLD_CONLL_LIST = [
    "./data/bc5cdr/gold/cdr_test.conll",
    "./data/ncbi_disease/gold/ncbi_test.conll",
    "./data/wikigold/gold/wiki_test.conll"
]
DISTANT_CONLL_LIST = [
    "./data/bc5cdr/distant/cdr_test.conll",
    "./data/ncbi_disease/distant/ncbi_test.conll",
    "./data/wikigold/distant/wiki_test.conll"
]
GAZET = [
    "./data/bc5cdr/gazetteers/",
    "./data/ncbi_disease/gazetteers/",
    "./data/wikigold/gazetteers/"
]
COLUMNS = ["dataset", "precision", "recall", "f1"]
DATA = []


def benchmark_kb(wandb_: bool=False):
    logging.info("=== KB Matching ===")
    for corpus, gold_conll, distant_conll, gazet \
        in list(zip(CORPUS_LIST, GOLD_CONLL_LIST, DISTANT_CONLL_LIST, GAZET)):
        logging.info(f"Reading training data from {corpus}")
        with open(corpus, "r", encoding="utf-8") as f:
            corpus_ = f.read().splitlines()
        corpus_ = [text.lower() for text in corpus_]
        test_corpus, y_true = pp.read_conll(gold_conll)

        logging.info(f"Loading gazetteers from {gazet}")
        phrase_miner = PhraseMiner("en")
        gazetteers = pp.load_gazetteers(gazet)
        phrase_miner.compute_patterns(gazetteers)
        y_pred = []
        with open(distant_conll, "w") as f:
            for gold_text, dist_text in list(zip(test_corpus, corpus_)):
                doc = phrase_miner.nlp(dist_text)
                ents = [
                    (ent.text, ent.start_char, ent.end_char, ent.label_) \
                    for ent in doc.ents
                ]
                text_idx = 0
                token_idx = 0
                local_y_pred = []
                for ent in ents:
                    while token_idx < ent[1]:
                        f.write(f"{gold_text[text_idx]}\tO\n")
                        local_y_pred.append("O")
                        token_idx += len(gold_text[text_idx]) + 1
                        text_idx += 1
                    if token_idx == ent[1]:
                        f.write(f"{gold_text[text_idx]}\tB-{ent[-1]}\n")
                        local_y_pred.append(f"B-{ent[-1]}")
                        token_idx += len(gold_text[text_idx]) + 1
                        text_idx += 1
                    while ent[1] < token_idx < ent[2]:
                        f.write(f"{gold_text[text_idx]}\tI-{ent[-1]}\n")
                        local_y_pred.append(f"I-{ent[-1]}")
                        token_idx += len(gold_text[text_idx]) + 1
                        text_idx += 1
                while len(local_y_pred) < len(gold_text):
                    local_y_pred.append("O")
                y_pred.append(local_y_pred)
        logging.warning(classification_report(y_true, y_pred, mode="strict", scheme=IOB2))
        if wandb_:
            f1 = f1_score(y_true, y_pred, mode="strict", scheme=IOB2)
            precision = precision_score(y_true, y_pred, mode="strict", scheme=IOB2)
            recall = recall_score(y_true, y_pred, mode="strict", scheme=IOB2)
            DATA.append([gold_conll, precision, recall, f1])
    if wandb_:
        with wandb.init(project="miner", entity="madjakul", name="benchmark_kb"):
            table = wandb.Table(data=DATA, columns=COLUMNS)
            wandb.log({"kb_benchmark": table})
    logging.info("--- Done ---\n\n")

