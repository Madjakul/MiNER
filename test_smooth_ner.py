# test_smooth_ner

import logging

import torch
from transformers import AutoTokenizer
from tqdm import tqdm
from seqeval.scheme import IOB2
from seqeval.metrics import (
    classification_report, f1_score, precision_score, recall_score
)

from miner.modules import SmoothNER
from miner.utils import TestSmoothArgParse, logging_config, align_labels
from miner.utils.data import preprocessing as pp


logging_config()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cuda":
    torch.cuda.empty_cache()


if __name__=="__main__":
    args = TestSmoothArgParse.parse_known_args()

    logging.info("\n\n=== Testing Smooth NER===")

    logging.info(f"Reading labels from {args.labels_path}")
    with open(args.labels_path, "r", encoding="utf-8") as f:
        labels = f.read().splitlines()
    label2idx = {label: idx for idx, label in enumerate(labels)}
    idx2label = {v: k for k, v in label2idx.items()}
    test_corpus, test_labels = pp.read_conll(args.test_corpus_path)

    logging.info(f"Loading the NER from {args.ner_path}")
    ner = SmoothNER(
        lm_path=args.lm_path,
        num_labels=len(labels),
        device=DEVICE,
        dropout=0.1
    ).to(DEVICE)
    ner.load_state_dict(torch.load(args.ner_path)["model_state_dict"])
    ner.eval()

    tokenizer = AutoTokenizer.from_pretrained(
        "roberta-base", add_prefix_space=True
    )

    logging.info("Predicting...")
    y_pred = []
    with torch.no_grad():
        for sequence, tags in tqdm(list(zip(test_corpus, test_labels))):
            inputs = tokenizer(
                sequence,
                is_split_into_words=True,
                max_length=256,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            ).to(DEVICE)
            local_y_pred = ner.decode(inputs)[0]
            local_y_pred = align_labels(inputs, local_y_pred, idx2label)
            y_pred.append(local_y_pred)
    logging.warning(
        classification_report(test_labels, y_pred, mode="strict", scheme=IOB2)
    )
    logging.warning(
        precision_score(test_labels, y_pred, mode="strict", scheme=IOB2)
    )
    logging.warning(
        recall_score(test_labels, y_pred, mode="strict", scheme=IOB2)
    )
    logging.warning(
        f1_score(test_labels, y_pred, mode="strict", scheme=IOB2)
    )
    logging.info("== Done ==")

