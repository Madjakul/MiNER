# train_smooth_ner.py

import logging

import wandb
import torch
import torch.backends.cudnn
from torch.utils.data import DataLoader

from miner.utils import TrainSmoothArgParse, logging_config
from miner.modules import PartialNER, SmoothNER
from miner.trainers import SmoothNERTrainer
from miner.utils.data import SmoothNERDataset
from miner.utils.data import preprocessing as pp


logging_config()
if torch.cuda.is_available():
    DEVICE = "cuda"
    torch.cuda.empty_cache()
else: DEVICE = "cpu"
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


if __name__=="__main__":
    args = TrainSmoothArgParse.parse_known_args()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    if args.wandb:
        wandb.init(
            project=args.project,
            entity=args.entity,
            name=f"""train{args.ner_path}_bs{args.ner_batch_size}-""" \
                f"""lr{args.lr}-m{args.momentum}-c{args.clip}"""
        )

    logging.info("\n\n=== Training Smooth NER ===")
    logging.info(f"Loading labels from {args.labels_path}")
    with open(args.labels_path, "r", encoding="utf-8") as f:
        labels = f.read().splitlines()
    label2idx = {label: idx for idx, label in enumerate(labels)}
    idx2label = {v: k for k, v in label2idx.items()}
    logging.info(f"Loading the partial NER from {args.partial_ner_path}")
    partial_ner = PartialNER(
        lm_path=args.lm_path,
        num_labels=len(labels),
        device=DEVICE,
        dropout=args.dropout,
    ).to(DEVICE)
    partial_ner.load_state_dict(
        torch.load(args.partial_ner_path)["model_state_dict"]
    )
    partial_ner.eval()

    logging.info(f"Loading training data from {args.train_data_path}")
    train_corpus, train_labels = pp.read_conll(args.train_data_path)
    logging.info("Building the training dataloader...")
    train_dataset = SmoothNERDataset(
        partial_ner=partial_ner,
        device=DEVICE,
        max_length=args.max_length,
        corpus=train_corpus,
        lm_path=args.lm_path
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.ner_batch_size,
        shuffle=True
    )

    val_dataloader = None
    if args.val_data_path is not None:
        logging.info(f"Loading validation data from {args.val_data_path}")
        val_corpus, val_labels = pp.read_conll(args.val_data_path)
        logging.info("Building the validation dataloader...")
        val_dataset = SmoothNERDataset(
            partial_ner=partial_ner,
            device=DEVICE,
            max_length=args.max_length,
            corpus=val_corpus,
            lm_path=args.lm_path,
            labels=val_labels
        )

    smooth_ner = SmoothNER(
        lm_path=args.lm_path,
        num_labels=len(labels),
        device=DEVICE,
        dropout=args.dropout,
    ).to(DEVICE)

    logging.info("*** Training ***")
    trainer = SmoothNERTrainer(
        smooth_ner=smooth_ner,
        lr=args.lr,
        epochs=args.ner_epochs,
        accumulation_steps=args.accumulation_steps,
        max_length=args.max_length,
        device=DEVICE,
        ner_path=args.smooth_ner_path,
        idx2label=idx2label
    )
    trainer(
        train_dataloader,
        val_dataset=val_dataset,
        wandb_=args.wandb
    )
    logging.info("=== Done ===")

