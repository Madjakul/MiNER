# train_miner.py

import logging
import argparse

import torch
from torch.utils.data import DataLoader

from miner.utils import logging_config
from miner.modules import NER
from miner.trainers import NER_Trainer
from miner.utils.data import NER_Dataset
from miner.utils.data import preprocessing as pp


logging_config()
if torch.cuda.is_available():
    DEVICE = "cuda"
    torch.cuda.empty_cache()
else: DEVICE = "cpu"


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", type=str, default="en")
    parser.add_argument(
        "--train_data_path",
        type=str,
        default="./data/bc5cdr/cdr_train.conll"
    )
    parser.add_argument(
        "--gazetteers_path",
        type=str,
        default="./data/gazetteers/"
    )
    parser.add_argument(
        "--labels_path",
        type=str,
        default="./data/labels.txt"
    )
    parser.add_argument("--lm_path", type=str, default="./tmp/lm")
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--ner_batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--momentum", type=float, default=0.72)
    parser.add_argument("--clip", type=float, default=3.2)
    parser.add_argument("--corrected_loss", type=int, default=0)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--ner_epochs", type=int, default=100)
    parser.add_argument("--ner_accumulation_steps", type=int, default=4)
    parser.add_argument("--ner_path", type=str, default="./tmp/ner.pt")
    parser.add_argument("--min_delta", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=8)
    args = parser.parse_args()

    logging.info("=== Training ===")

    torch.manual_seed(args.seed)

    logging.info(f"Loading labels from {args.labels_path}")
    with open(args.labels_path, "r", encoding="utf-8") as f:
        labels = f.read().splitlines()
    logging.info(f"Loading training data from {args.train_data_path}")
    train_corpus, train_labels = pp.read_conll(args.train_data_path)
    logging.info(f"Loading validation data from {args.val_corpus_path}")
    val_corpus, val_labels = pp.read_conll(args.val_corpus_path)
    logging.info("Building the training dataloader...")
    train_dataset = NER_Dataset(
        lang=args.lang,
        device=DEVICE,
        max_length=args.max_length,
        iterable_corpus=train_corpus,
        labels=labels,
        iterable_labels=train_labels
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.ner_batch_size,
        shuffle=True
    )

    logging.info(f"Building the NER with {train_dataset.label2idx}")
    ner = NER(
        lang=args.lang,
        max_length=args.max_length,
        lm_path=args.lm_path,
        num_labels=len(labels) + 1,
        padding_idx=len(labels),
        device=DEVICE,
        partial=True,
        corrected_loss=bool(args.corrected_loss),
        gamma=args.gamma
    ).to(DEVICE)

    logging.info("Training...")
    trainer = NER_Trainer(
        ner=ner,
        lr=args.lr,
        momentum=args.momentum,
        patience=args.patience,
        min_delta=args.min_delta,
        epochs=args.ner_epochs,
        max_length=args.max_length,
        device=DEVICE,
        accumulation_steps=args.ner_accumulation_steps,
        ner_path=args.ner_path,
        clip=args.clip
    )
    trainer.train(train_dataloader)
    logging.info("--- Done ---\n\n")

