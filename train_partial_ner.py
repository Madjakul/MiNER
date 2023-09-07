# train_partial_ner.py

import logging

import wandb
import torch
import torch.backends.cudnn
from torch.utils.data import DataLoader

from miner.utils import TrainPartialArgParse, logging_config
from miner.modules import PartialNER
from miner.trainers import PartialNERTrainer
from miner.utils.data import PartialNERDataset
from miner.utils.data import preprocessing as pp


logging_config()
if torch.cuda.is_available():
    DEVICE = "cuda"
    torch.cuda.empty_cache()
else: DEVICE = "cpu"
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


if __name__=="__main__":
    args = TrainPartialArgParse.parse_known_args()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    if args.wandb:
        wandb.init(
            project=args.project,
            entity=args.entity,
            name=f"""train{args.ner_path}_bs{args.ner_batch_size}-""" \
                f"""lr{args.lr}-m{args.momentum}-c{args.clip}"""
        )

    logging.info("=== Training ===")

    logging.info(f"Loading labels from {args.labels_path}")
    with open(args.labels_path, "r", encoding="utf-8") as f:
        labels = f.read().splitlines()
    logging.info(f"Loading training data from {args.train_data_path}")
    train_corpus, train_labels = pp.read_conll(args.train_data_path)
    logging.info("Building the training dataloader...")
    train_dataset = PartialNERDataset(
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
    if args.val_data_path is not None:
        logging.info(f"Loading validation data from {args.val_data_path}")
        val_corpus, val_labels = pp.read_conll(args.val_data_path)

        logging.info("Building the validation dataloader...")
        val_dataset = PartialNERDataset(
            lang=args.lang,
            device=DEVICE,
            max_length=args.max_length,
            iterable_corpus=val_corpus,
            labels=labels,
            iterable_labels=val_labels
        )
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=args.ner_batch_size * 2,
            shuffle=True
        )

    logging.info(f"Building the NER with {train_dataset.label2idx}")
    ner = PartialNER(
        lang=args.lang,
        max_length=args.max_length,
        lm_path=args.lm_path,
        num_labels=len(labels),
        device=DEVICE,
        dropout=args.dropout,
        q=args.q
    ).to(DEVICE)
    logging.info("*** Training ***")
    trainer = PartialNERTrainer(
        ner=ner,
        lr=args.lr,
        momentum=args.momentum,
        patience=args.patience,
        epochs=args.ner_epochs,
        max_length=args.max_length,
        device=DEVICE,
        ner_path=args.ner_path,
        clip=args.clip,
        sam=args.sam,
        idx2label = {v: k for k, v in train_dataset.label2idx.items()},
        loss_fn=args.loss_fn
    )
    trainer.train(
        train_dataloader,
        val_dataloader=val_dataloader if "val_dataloader" in locals() else None,
        wandb_=args.wandb
    )
    logging.info("--- Done ---\n\n")

