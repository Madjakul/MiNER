# test_miner.py

import logging
import argparse

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
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
    parser.add_argument("--ner_path", type=str)
    parser.add_argument("--q", type=float, default=0.1)
    parser.add_argument("--dropout", type=float, default=0.1)
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
        dropout=args.dropout,
        q=args.q
    ).to(DEVICE)
    ner.load_state_dict(torch.load(args.ner_path)["model_state_dict"])
    ner.eval()

    logging.info("Predicting...")
    y_true = []
    y_pred = []
    with torch.no_grad():
        for x, y, mask in tqdm(test_dataloader):
            x = {key: val.squeeze(1) for key, val in x.items()}
            local_y_pred_ = ner.viterbi_decode(x)[0]
            local_y_true_ = y.tolist()[0][:len(local_y_pred_)]
            local_mask_ = mask.tolist()[0][:(len(local_y_pred_))]
            local_y_pred, local_y_true, local_mask = [], [], []
            for idx, mask_value in enumerate(local_mask_):
                if mask_value == 0:
                    continue
                local_y_pred.append(idx2label[local_y_pred_[idx]])
                local_y_true.append(idx2label[local_y_true_[idx]])
            y_pred.append(local_y_pred)
            y_true.append(local_y_true)
    logging.warning(classification_report(y_true, y_pred, mode="strict", scheme=IOB2))
    logging.warning(precision_score(y_true, y_pred, mode="strict", scheme=IOB2))
    logging.warning(recall_score(y_true, y_pred, mode="strict", scheme=IOB2))
    logging.warning(f1_score(y_true, y_pred, mode="strict", scheme=IOB2))

