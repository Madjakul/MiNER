# pretrain_miner.py

import logging

import torch
import torch.backends.cudnn

from miner.utils import PretrainArgParse, logging_config
from miner.trainers import TransformerTrainer
from miner.utils.data import TransformerDataset
from miner.modules import RoBERTa


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

    logging.info("\n\n=== Pretraining LM for MiNER ===")


    logging.info(f"Loading training data from {args.train_corpus_path}")
    with open(args.train_corpus_path, "r", encoding="utf-8") as f:
        train_corpus = f.read().splitlines()
    logging.info(f"Loading validation data from {args.val_corpus_path}")
    with open(args.val_corpus_path, "r", encoding="utf-8") as f:
        val_corpus = f.read().splitlines()

    logging.info("Using  RoBERTa checkpoint as language model")
    lm = RoBERTa(DEVICE)

    logging.info("Building the dataset...")
    lm_dataset = TransformerDataset(
        train_corpus=train_corpus,
        valid_corpus=val_corpus,
        max_length=args.max_length,
        mlm_probability=args.mlm_probability,
    )

    # There is a silent bug on HuggingFace modifying the beavior of the
    # tokenizers, upon using the ``add_vocab`` function.
    # see: https://github.com/huggingface/transformers/pull/23909
    # I don't know how token encoding impact the performances on a downstream
    # task, therefor, won't use it for further pre-training.
    # lm_dataset.add_vocab(train_corpus, lm)

    logging.info("*** Training ***")
    lm_trainer = TransformerTrainer(
        lm=lm,
        lm_path=args.lm_path,
        lm_dataset=lm_dataset,
        per_device_train_batch_size=args.lm_train_batch_size,
        seed=args.seed,
        per_device_eval_batch_size=args.lm_train_batch_size,
        max_steps=args.max_steps,
        gradient_accumulation_steps=args.lm_accumulation_steps,
        wandb=args.wandb
    )
    lm_trainer.train()
    logging.info("=== Done ===")

