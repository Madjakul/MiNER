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
    logging.info("=== Training ===")
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", choices=["en", "fr"], default="en")
    parser.add_argument(
        "--train_corpus_path",
        type=str,
        default="./data/bc5cdr/cdr_train.conll"
    )
    parser.add_argument(
        "--valid_corpus_path",
        type=str,
        default="./data/bc5cdr/cdr_val.conll"
    )
    parser.add_argument(
        "--gazetteers_path",
        type=str,
        default="./data/gazetteers/"
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        default=["O", "B-CHEMICAL", "I-CHEMICAL", "B-DISEASE", "I-DISEASE"]
    )
    parser.add_argument("--lm_path", type=str, default="./tmp/lm")
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--ner_batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--ner_epochs", type=int, default=50)
    parser.add_argument("--ner_accumulation_steps", type=int, default=4)
    parser.add_argument("--ner_path", type=str, default="./tmp/ner.pt")
    parser.add_argument("--min_delta", type=int, default=0.1)
    parser.add_argument("--seed", type=int, default=8)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    train_corpus, train_labels = pp.read_conll(args.train_corpus_path)
    val_corpus, val_labels = pp.read_conll(args.valid_corpus_path)
    train_dataset = NER_Dataset(
        lang=args.lang,
        device=DEVICE,
        max_length=args.max_length,
        iterable_corpus=train_corpus,
        labels=args.labels,
        iterable_labels=train_labels
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.ner_batch_size,
        shuffle=True
    )
    val_dataset = NER_Dataset(
        lang=args.lang,
        device=DEVICE,
        max_length=args.max_length,
        iterable_corpus=val_corpus,
        labels=args.labels,
        iterable_labels=val_labels
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.ner_batch_size,
        shuffle=True
    )
    ner = NER(
        lang=args.lang,
        max_length=args.max_length,
        lm_path=args.lm_path,
        num_labels=len(args.labels) + 1,
        padding_idx=len(args.labels),
        device=DEVICE
    ).to(DEVICE)
    trainer = NER_Trainer(
        ner=ner,
        lr=args.lr,
        momentum=args.momentum,
        patience=args.patience,
        min_delta=args.min_delta,
        epochs=args.ner_epochs,
        max_length=args.max_length,
        label2idx=train_dataset.label2idx,
        device=DEVICE,
        accumulation_steps=args.ner_accumulation_steps,
        ner_path=args.ner_path
    )
    trainer.train(train_dataloader, val_dataloader)
    logging.info("--- Done ---\n\n")

