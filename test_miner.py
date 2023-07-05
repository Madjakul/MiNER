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
        batch_size=1,
        shuffle=False
    )
    logging.info("Building the test dataloader...")
    idx2label = {v: k for k, v in test_dataset.label2idx.items()}

    logging.info(f"Loading the NER from {args.ner_path}")
    ner = NER(
        lang=args.lang,
        max_length=args.max_length,
        lm_path=args.lm_path,
        num_labels=len(labels),
        device=DEVICE,
        dropout=0.1,
        corrected_loss=bool(args.corrected_loss),
    ).to(DEVICE)
    ner.load_state_dict(torch.load(args.ner_path)["model_state_dict"])
    ner.eval()

    logging.info("Predicting...")
    y_true = []
    y_pred = []
    with torch.no_grad():
        for x, y, word_ids in test_dataloader:
            result = ner.viterbi_decode(x)[0]
            y_ = y.tolist()[0]
            local_y_true = []
            local_y_pred = []
            last = -100
            for idx, word_id in enumerate(word_ids[0]):
                if idx == 0:
                    local_y_true.append(idx2label[y_[idx]])
                    local_y_pred.append(idx2label[result[idx]])
                    continue
                elif last != word_id:
                    local_y_true.append(idx2label[y_[idx]])
                    local_y_pred.append(idx2label[result[idx]])
                last = word_id
                if last == -1:
                    break
            y_pred.append(local_y_pred)
            y_true.append(local_y_true)
    logging.warning(classification_report(y_true, y_pred, mode="strict", scheme=IOB2))
    logging.warning(precision_score(y_true, y_pred, mode="strict", scheme=IOB2))
    logging.warning(recall_score(y_true, y_pred, mode="strict", scheme=IOB2))
    logging.warning(f1_score(y_true, y_pred, mode="strict", scheme=IOB2))

