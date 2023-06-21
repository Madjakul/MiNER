# experiments/benchmark_kb.py

import logging

import wandb
from spacy.lang.en import English
from seqeval.scheme import IOB2
from seqeval.metrics import classification_report
from seqeval.metrics import f1_score, precision_score, recall_score

from miner.utils.data import preprocessing as pp


GOLD_CONLL_LIST = [
    "./data/bc5cdr/gold/cdr_test.conll",
    # "./data/ncbi_disease/gold/ncbi_test.conll",
    # "./data/wikigold/gold/wiki_test.conll"
]
DISTANT_CONLL_LIST = [
    "./data/bc5cdr/distant/cdr_test_autoner.conll",
    # "./data/ncbi_disease/distant/ncbi_test.conll",
    # "./data/wikigold/distant/wiki_test.conll"
]
COLUMNS = ["dataset", "precision", "recall", "f1"]
DATA = []


def benchmark_kb(wandb_: bool=False):
    logging.info("=== KB Matching ===")
    nlp = English()
    for gold_conll, distant_conll \
        in list(zip(GOLD_CONLL_LIST, DISTANT_CONLL_LIST)):
        logging.info(f"Reading test data from {gold_conll}")
        gold_corpus, y_true_ = pp.read_conll(gold_conll)
        logging.info(f"Reading distant test data from {distant_conll}")
        _, y_pred_ = pp.read_conll(distant_conll)
        print(len(y_true_))
        print(len(y_pred_))

        y_true = []
        y_pred = []
        i = 0
        for tokens, local_y_true_ in list(zip(gold_corpus, y_true_)):
            local_y_true = []
            for token, y_ in list(zip(tokens, local_y_true_)):
                doc = nlp(token)
                local_y_true.extend([y_] * len(doc))
            if len(local_y_true) == len(y_pred_[i]):
                y_true.append(local_y_true)
                y_pred.append(y_pred_[i])
            i += 1
        logging.warning(
            classification_report(y_true, y_pred, mode='strict', scheme=IOB2)
        )
        if wandb_:
            f1 = f1_score(
                y_true, y_pred, mode='strict', scheme=IOB2
            )
            precision = precision_score(
                y_true, y_pred, mode='strict', scheme=IOB2
            )
            recall = recall_score(y_true, y_pred, mode='strict', scheme=IOB2
            )
            DATA.append([gold_conll, precision, recall, f1])
    if wandb_:
        with wandb.init(
            project="miner", entity="madjakul", name="benchmark_kb"
        ):
            table = wandb.Table(data=DATA, columns=COLUMNS)
            wandb.log({"kb_benchmark": table})
    logging.info("--- Done ---\n\n")

