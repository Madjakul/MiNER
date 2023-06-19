# wandb_train.py

import logging

import wandb
import torch
from torch.utils.data import DataLoader

from miner.utils import logging_config
from miner.modules import NER
from miner.trainers import NER_Trainer
from miner.utils.data import NER_Dataset
from miner.utils.data import preprocessing as pp


GAMMA = [
    0.5, 0.55, 0.6, 0.65, 0.7,
    0.75, 0.8, 0.85, 0.9, 0.95, 1.0
]
COLUMNS = ["gamma", "precision", "recall", "f1"]
DATA = []


logging_config()
if torch.cuda.is_available():
    DEVICE = "cuda"
    torch.cuda.empty_cache()
else: DEVICE = "cpu"


def benchmark_gamma():
    logging.info("=== Training ===")

    lang = "en"
    batch_size = 8
    max_length = 256
    epochs = 50
    ner_path = "./tmp/cdr-dev_gamma-ner.pt"
    lm_path = "./tmp/cdr_lm"
    labels_path = "./data/bc5cdr/labels.txt"
    train_corpus_path = "./data/bc5cdr/distant/cdr_dev.conll"
    val_corpus_path = "./data/bc5cdr/gold/cdr_test.conll"

    for g in GAMMA:
        with wandb.init(
            project="miner",
            entity="madjakul",
            name=f"bc5cdr-benchmark_gamma-{g}"
        ):
            config = wandb.config
            logging.info(f"Loading labels from {labels_path}")
            with open(labels_path, "r", encoding="utf-8") as f:
                labels = f.read().splitlines()
            logging.info(f"Loading training data from {train_corpus_path}")
            train_corpus, train_labels = pp.read_conll(train_corpus_path)
            logging.info(f"Loading validation data from {val_corpus_path}")
            val_corpus, val_labels = pp.read_conll(val_corpus_path)
            logging.info("Building the training dataloader...")
            train_dataset = NER_Dataset(
                lang=lang,
                device=DEVICE,
                max_length=max_length,
                iterable_corpus=train_corpus,
                labels=labels,
                iterable_labels=train_labels
            )
            train_dataloader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=False
            )
            logging.info("Building the validation dataloader...")
            val_dataset = NER_Dataset(
                lang=lang,
                device=DEVICE,
                max_length=max_length,
                iterable_corpus=val_corpus,
                labels=labels,
                iterable_labels=val_labels
            )
            val_dataloader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False
            )

            logging.info(f"Building the NER with {train_dataset.label2idx}")
            ner = NER(
                lang=lang,
                max_length=max_length,
                lm_path=lm_path,
                num_labels=len(labels) + 1,
                padding_idx=len(labels),
                device=DEVICE,
                dropout=0.1,
                partial=True,
                corrected_loss=True,
                gamma=g
            ).to(DEVICE)

            logging.info("Training...")
            trainer = NER_Trainer(
                ner=ner,
                lr=0.05,
                patience=5,
                min_delta=0.001,
                epochs=epochs,
                max_length=max_length,
                device=DEVICE,
                accumulation_steps=16 // batch_size,
                ner_path=ner_path,
                momentum=0.7,
                clip=4.0,
                optimizer="SGD",
                sam=True,
                idx2label = {v: k for k, v in val_dataset.label2idx.items()}
            )
            trainer.train(
                train_dataloader=train_dataloader,
                val_dataloader=val_dataloader,
                wandb_=True
            )
        logging.info("--- Done ---\n\n")

