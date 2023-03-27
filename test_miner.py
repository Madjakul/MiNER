# test_miner.py

import logging
import argparse
from itertools import chain

import torch
from torch.utils.data import DataLoader
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report

from miner.utils import logging_config
from miner.modules import NER
from miner.utils.data import NER_Dataset
from miner.utils.data import preprocessing as pp


logging_config()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cuda":
    torch.cuda.empty_cache()


def bio_classification_report(y_true: list, y_pred: list):
    """Classification report for a list of BIO-encoded sequences. It computes
    token-level metrics.

    Warnings
    --------
    It requires scikit-learn 0.15+ (or a version from github master) to
    calculate averages properly!

    Parameters
    ----------
    y_true: ``list``
        True labels.
    y_pred: ``list``
        Predicted labels.

    Returns
    -------
    classification_report: ``str``
        Scikit-learn classification report.
    """
    lb = LabelBinarizer()
    y_true_combined = lb.fit_transform(list(chain.from_iterable(y_true)))
    y_pred_combined = lb.transform(list(chain.from_iterable(y_pred)))

    tagset = set(lb.classes_) - {"PAD"}
    tagset = sorted(tagset, key=lambda tag: tag.split("-", 1)[::-1])
    class_indices = {cls: idx for idx, cls in enumerate(lb.classes_)}

    return classification_report(
        y_true_combined,
        y_pred_combined,
        labels = [class_indices[cls] for cls in tagset],
        target_names = tagset,
    )


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", choices=["en", "fr"], default="en")
    parser.add_argument(
        "--test_corpus_path",
        type=str,
        default="./data/bc5cdr/cdr_test.conll"
    )
    parser.add_argument(
        "--labels_path",
        type=str,
        default="./data/labels.txt"
    )
    parser.add_argument("--lm_path", type=str, default="./tmp/lm")
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--ner_batch_size", type=int, default=4)
    parser.add_argument("--ner_path", type=str, default="./tmp/ner.pt")
    args = parser.parse_args()

    logging.info("=== Testing ===")

    logging.info(f"Reading labels from {args.labels_path}")
    with open(args.labels_path, "r", encoding="utf-8") as f:
        labels = f.read().splitlines()
    logging.info(f"Loading test data from {args.test_corpus_path}")
    test_corpus, test_labels = pp.read_conll(args.test_corpus_path)
    logging.info("Building the test dataloader...")
    test_dataset = NER_Dataset(
        lang=args.lang,
        device=DEVICE,
        iterable_corpus=test_corpus,
        iterable_labels=test_labels,
        labels=labels,
        max_length=args.max_length,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.ner_batch_size,
        shuffle=True
    )
    idx2label = {v: k for k, v in test_dataset.label2idx.items()}

    logging.info(f"Loading the NER from {args.ner_path}")
    ner = NER(
        lang=args.lang,
        max_length=args.max_length,
        lm_path=args.lm_path,
        num_labels=len(labels) + 1,
        padding_idx=test_dataset.label2idx["PAD"],
        device=DEVICE
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
        for j, label in enumerate(y):
            y_pred[i][j] = idx2label[y_pred[i][j]]
            y_true[i][j] = idx2label[y_true[i][j]]
        y_true[i] = y_true[i][:len(y_pred[i])]

    logging.info("Testing...")
    logging.warning(bio_classification_report(y_true, y_pred))

