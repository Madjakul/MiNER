# pretrain_miner.py

import logging
import argparse

import torch

from miner.utils import logging_config
from miner.trainers import TransformerTrainer
from miner.utils.data import TransformerDataset
from miner.modules import RoBERTa, CamemBERT, Longformer
from miner.utils.data import preprocessing as pp


logging_config()
if torch.cuda.is_available():
    DEVICE = "cuda"
    torch.cuda.empty_cache()
else: DEVICE = "cpu"


if __name__=="__main__":
    logging.info("=== Pretraining ===")
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", choices=["fr", "en"], default="en")
    parser.add_argument("--max_length", type=int, default=256)
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
    parser.add_argument("--lm_path", type=str, default="./tmp/lm")
    parser.add_argument("--seed", type=int, default=8)
    parser.add_argument("--mlm_probability", type=float, default=0.15)
    parser.add_argument("--lm_train_batch_size", type=int, default=4)
    parser.add_argument("--lm_epochs", type=int, default=60)
    parser.add_argument("--lm_accumulation_steps", type=int, default=16)
    args = parser.parse_args()
    # Setting up a torch seed for replicability
    torch.manual_seed(args.seed)
    # Loading the training and the validation corpus
    logging.info(f"Loading training data from {args.train_corpus_path}")
    train_corpus, train_labels = pp.read_conll(args.train_corpus_path)
    logging.info(f"Loading validation data from {args.valid_corpus_path}")
    val_corpus, val_labels = pp.read_conll(args.valid_corpus_path)
    if args.lang == "fr":
        lm = CamemBERT(DEVICE)
    elif args.max_length > 512:
        lm = Longformer(DEVICE)
    else:
        lm = RoBERTa(DEVICE)
    # Building the custom dataset to further pretrain the LLM
    logging.info("Building the dataset...")
    lm_dataset = TransformerDataset(
        lang=args.lang,
        train_corpus=train_corpus,
        valid_corpus=val_corpus,
        max_length=args.max_length,
        mlm_probability=args.mlm_probability,
    )
    lm_dataset.add_vocab(train_corpus, lm)
    # Building the language model's trainer
    lm_trainer = TransformerTrainer(
        lm=lm,
        lm_path=args.lm_path,
        lm_dataset=lm_dataset,
        per_device_train_batch_size=args.lm_train_batch_size,
        seed=args.seed,
        per_device_eval_batch_size=args.lm_train_batch_size,
        num_train_epochs=args.lm_epochs,
        gradient_accumulation_steps=args.lm_accumulation_steps,
    )
    # Pretraining of the LLM on domain-specifi data
    logging.info("Training...")
    lm_trainer.train()
    logging.info("--- Done ---\n\n")

