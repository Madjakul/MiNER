# pretrain_miner.py

import logging

import torch
import torch.backends.cudnn

from miner.utils import PretrainArgParse, logging_config
from miner.trainers import TransformerTrainer
from miner.utils.data import TransformerDataset
from miner.modules import RoBERTa, CamemBERT, Longformer


logging_config()
if torch.cuda.is_available():
    DEVICE = "cuda"
    torch.cuda.empty_cache()
else: DEVICE = "cpu"
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


if __name__=="__main__":
    args = PretrainArgParse.parse_known_args()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    logging.info("=== Pretraining LM for MiNER ===")


    logging.info(f"Loading training data from {args.train_corpus_path}")
    with open(args.train_corpus_path, "r", encoding="utf-8") as f:
        train_corpus = f.read().splitlines()
    logging.info(f"Loading validation data from {args.val_corpus_path}")
    with open(args.val_corpus_path, "r", encoding="utf-8") as f:
        val_corpus = f.read().splitlines()

    if args.lang == "fr":
        logging.info("Using CamemBERT checkpoint as language model")
        lm = CamemBERT(DEVICE)
    elif args.max_length > 512:
        logging.info("Using Longformer checkpoint as language model")
        lm = Longformer(DEVICE)
    else:
        logging.info("Using  RoBERTa checkpoint as language model")
        lm = RoBERTa(DEVICE)

    logging.info("Building the dataset...")
    lm_dataset = TransformerDataset(
        lang=args.lang,
        train_corpus=train_corpus,
        valid_corpus=val_corpus,
        max_length=args.max_length,
        mlm_probability=args.mlm_probability,
    )
    # lm_dataset.add_vocab(train_corpus, lm)

    logging.info("*** Training ***")
    lm_trainer = TransformerTrainer(
        lm=lm,
        lm_path=args.lm_path,
        lm_dataset=lm_dataset,
        per_device_train_batch_size=args.lm_train_batch_size,
        seed=args.seed,
        per_device_eval_batch_size=args.lm_train_batch_size,
        num_train_epochs=args.lm_epochs,
        gradient_accumulation_steps=args.lm_accumulation_steps,
        wandb=args.wandb
    )
    lm_trainer.train()
    logging.info("=== Done ===\n\n")

