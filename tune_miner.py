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


logging_config()
if torch.cuda.is_available():
    DEVICE = "cuda"
    torch.cuda.empty_cache()
else: DEVICE = "cpu"


if __name__=="__main__":
    logging.info("=== Training ===")

    lang = "en"
    max_length = 512
    epochs = 100
    ner_path = "./tmp/ner.pt"
    lm_path = "./tmp/lm"
    labels_path = "./data/bc5cdr/labels.txt"
    train_corpus_path = "./data/bc5cdr/distant/cdr_dev.conll"
    val_corpus_path = "./data/bc5cdr/gold/cdr_test.conll"

    with wandb.init(project="miner", entity="madjakul", name="bc5cdr_tuning"):
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
            batch_size=4,
            shuffle=True
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
            batch_size=4,
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
            dropout=config.dropout,
            partial=True,
            corrected_loss=True,
            gamma=config.gamma
        ).to(DEVICE)

        logging.info("Training...")
        trainer = NER_Trainer(
            ner=ner,
            lr=config.lr,
            patience=5,
            min_delta=1.0,
            epochs=epochs,
            max_length=max_length,
            device=DEVICE,
            accumulation_steps=4,
            ner_path=ner_path,
            momentum=config.momentum,
            clip=config.clip,
            optimizer=config.optimizer,
            sam=bool(config.sam),
            idx2label = {v: k for k, v in val_dataset.label2idx.items()}
        )
        trainer.train(
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            wandb_=True
        )
        logging.info("--- Done ---\n\n")

