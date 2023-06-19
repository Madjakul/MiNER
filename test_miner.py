# test_miner.py

import logging
import argparse

import torch
from torch.utils.data import DataLoader
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score
from seqeval.scheme import IOB2

from miner.utils import logging_config
from miner.modules import NER
from miner.utils.data import NER_Dataset
from miner.utils.data import preprocessing as pp


logging_config()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cuda":
    torch.cuda.empty_cache()


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", type=str)
    parser.add_argument(
        "--test_corpus_path",
        type=str,
    )
    parser.add_argument(
        "--labels_path",
        type=str,
    )
    parser.add_argument("--lm_path", type=str)
    parser.add_argument("--max_length", type=int)
    parser.add_argument("--ner_batch_size", type=int)
    parser.add_argument("--ner_path", type=str)
    parser.add_argument("--corrected_loss", type=int)
    parser.add_argument("--gamma", type=float)
    args = parser.parse_args()

    logging.info("=== Testing ===")

    logging.info(f"Reading labels from {args.labels_path}")
    with open(args.labels_path, "r", encoding="utf-8") as f:
        labels = f.read().splitlines()
    logging.info(f"Loading test data from {args.test_corpus_path}")
    test_corpus, test_labels = pp.read_conll(args.test_corpus_path)
    test_dataset = NER_Dataset(
        lang=args.lang,
        device=DEVICE,
        max_length=args.max_length,
        iterable_corpus=test_corpus,
        labels=labels,
        iterable_labels=test_labels
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=4,
        shuffle=False
    )
    logging.info("Building the test dataloader...")
    idx2label = {v: k for k, v in test_dataset.label2idx.items()}

    logging.info(f"Loading the NER from {args.ner_path}")
    ner = NER(
        lang=args.lang,
        max_length=args.max_length,
        lm_path=args.lm_path,
        num_labels=len(labels) + 1,
        padding_idx=test_dataset.label2idx["PAD"],
        device=DEVICE,
        partial=True,
        dropout=0.1,
        corrected_loss=bool(args.corrected_loss),
        gamma=args.gamma
    ).to(DEVICE)
    ner.load_state_dict(torch.load(args.ner_path)["model_state_dict"])
    ner.eval()

    logging.info("Predicting...")
    y_true = []
    y_pred = []
    with torch.no_grad():
        for x, y in test_dataloader:
            result = ner.viterbi_decode(x)
            y_pred.extend(result)
            y_true.extend(y.tolist())
        for i, y in enumerate(y_pred):
            for j, _ in enumerate(y):
                y_pred[i][j] = idx2label[y_pred[i][j]]
                y_true[i][j] = idx2label[y_true[i][j]]
            y_true[i] = y_true[i][:len(y_pred[i])]
            y_true[i][-1] = idx2label[test_dataset.label2idx["O"]] # Replace the </s> tagged with PAD by an O for proper alignement
    logging.warning(classification_report(y_true, y_pred, mode="strict", scheme=IOB2))
    logging.warning(precision_score(y_true, y_pred, mode="strict", scheme=IOB2))
    logging.warning(recall_score(y_true, y_pred, mode="strict", scheme=IOB2))
    logging.warning(f1_score(y_true, y_pred, mode="strict", scheme=IOB2))

