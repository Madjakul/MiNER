# selftrain_miner.py

import logging
import argparse

import torch
from torch.utils.data import DataLoader

from miner.utils import logging_config
from miner.modules import NER
from miner.trainers import SelfTrainer
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
    )
    parser.add_argument(
        "--val_data_path",
        type=str
    )
    parser.add_argument(
        "--gazetteers_path",
        type=str,
    )
    parser.add_argument(
        "--labels_path",
        type=str,
    )
    parser.add_argument("--lm_path", type=str)
    parser.add_argument("--sam", type=int)
    parser.add_argument("--max_length", type=int)
    parser.add_argument("--ner_batch_size", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--corrected_loss", type=int)
    parser.add_argument("--gamma", type=float)
    parser.add_argument("--ner_epochs", type=int)
    parser.add_argument("--ner_accumulation_steps", type=int)
    parser.add_argument("--ner_path", type=str)
    parser.add_argument("--dropout", type=float)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()

    logging.info("=== Training ===")

    torch.manual_seed(args.seed)

    logging.info(f"Loading labels from {args.labels_path}")
    with open(args.labels_path, "r", encoding="utf-8") as f:
        labels = f.read().splitlines()
    logging.info(f"Loading training data from {args.train_data_path}")
    train_corpus, train_labels = pp.read_conll(args.train_data_path)
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
    logging.info(f"Loading validation data from {args.val_data_path}")
    val_corpus, val_labels = pp.read_conll(args.val_data_path)
    val_dataset = NER_Dataset(
        lang=args.lang,
        device=DEVICE,
        max_length=args.max_length,
        iterable_corpus=val_corpus,
        labels=labels,
        iterable_labels=val_labels
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.ner_batch_size,
        shuffle=False
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
        dropout=args.dropout,
        gamma=args.gamma
    ).to(DEVICE)
    ner.load_state_dict(torch.load(args.ner_path)["model_state_dict"])

    logging.info("Training...")
    trainer = SelfTrainer(
        ner=ner,
        lr=args.lr,
        train_dataloader=train_dataloader,
        batch_size=args.ner_batch_size,
        epochs=args.ner_epochs,
        max_length=args.max_length,
        device=DEVICE,
        accumulation_steps=args.ner_accumulation_steps,
        ner_path=args.ner_path,
        sam=bool(args.sam),
        idx2label = {v: k for k, v in train_dataset.label2idx.items()}
    )
    trainer.train(train_dataloader, val_dataloader)
    logging.info("--- Done ---\n\n")

