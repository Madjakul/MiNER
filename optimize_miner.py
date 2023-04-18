# optimize_miner.py

import logging

import wandb
import torch
from torch.utils.data import DataLoader

from miner.utils import logging_config, get_batch_size
from miner.modules import NER
from miner.trainers import NER_Trainer
from miner.utils.data import NER_Dataset
from miner.utils.data import preprocessing as pp


WANDB_PROJECT_NAME = "miner_bc5cdr_hyperparameter-optimization"
logging_config()
if torch.cuda.is_available():
    DEVICE = "cuda"
    torch.cuda.empty_cache()
else: DEVICE = "cpu"


if __name__=="__main__":
    logging.info("=== Training ===")

    lang = "en"
    max_length = 256
    epochs = 100
    ner_path = "./tmp/ner.pt"
    lm_path = "roberta-base"
    labels_path = "./data/labels.txt"
    train_corpus_path = "./data/bc5cdr/cdr_val.conll"
    val_corpus_path = "./data/bc5cdr/cdr_train.conll"
    test_corpus_path ="./data/bc5cdr/cdr_test.conll"
    dummy_ner = NER(
        lang="en",
        max_length=max_length,
        lm_path=lm_path,
        num_labels=5,
        padding_idx=0,
        device=DEVICE
    ).to(DEVICE)
    batch_size = get_batch_size(
        model=dummy_ner,
        max_length=max_length,
        dataset_size=5000,
        device=DEVICE,
    ) // 2
    logging.info(f"Maximum supported batch size: {batch_size}")
    # batch_size = 1

    with wandb.init(project=WANDB_PROJECT_NAME):    # type: ignore
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
            batch_size=batch_size,
            shuffle=True
        )

        logging.info(f"Building the NER with {train_dataset.label2idx}")
        ner = NER(
            lang=lang,
            max_length=max_length,
            lm_path=lm_path,
            num_labels=len(labels) + 1,
            padding_idx=len(labels),
            device=DEVICE,
            dropout=config.dropout
        ).to(DEVICE)

        logging.info("Training...")
        trainer = NER_Trainer(
            ner=ner,
            lr=config.lr,
            optim=config.optim,
            patience=config.patience,
            min_delta=config.min_delta,
            epochs=epochs,
            max_length=max_length,
            device=DEVICE,
            accumulation_steps=config.accumulation_steps,
            ner_path=ner_path
        )
        trainer.train(train_dataloader, val_dataloader)
        logging.info("--- Done ---\n\n")

