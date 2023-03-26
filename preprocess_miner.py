# phrase_mining.py

import logging
import argparse

from miner.utils import logging_config
from miner.utils.data import PhraseMiner
from miner.utils.data import preprocessing as pp


logging_config()


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", choices=["en", "fr"], default="en")
    parser.add_argument(
        "--train_corpus_path",
        type=str,
        default="./data/bc5cdr/cdr_train_corpus.txt"
    )
    parser.add_argument(
        "--val_corpus_path",
        type=str,
        default="./data/bc5cdr/cdr_valid_corpus.txt"
    )
    parser.add_argument(
        "--train_conll_path", type=str, default="./data/bc5cdr/cdr_train.conll"
    )
    parser.add_argument(
        "--val_conll_path", type=str, default="./data/bc5cdr/cdr_val.conll"
    )
    parser.add_argument(
        "--gazetteers_path", type=str, default="./data/gazetteers/"
    )
    parser.add_argument(
        "--unk_gazetteers_path", type=str, default="./data/gazetteers/UNK.txt"
    )
    args = parser.parse_args()
    logging.info("=== Mining Quality Phrases ===")

    logging.info(f"Reading training data from {args.train_corpus_path}")
    with open(args.train_corpus_path, "r", encoding="utf-8") as f:
        train_corpus = f.read().splitlines()
    logging.info(f"Reading validation data from {args.val_corpus_path}")
    with open(args.val_corpus_path, "r", encoding="utf-8") as f:
        val_corpus = f.read().splitlines()

    phrase_miner = PhraseMiner(lang=args.lang)
    logging.info(f"Loading gazetteers from {args.gazetteers_path}")
    gazetteers = pp.load_gazetteers(args.gazetteers_path)
    phrase_miner.compute_patterns(gazetteers)
    logging.info("Mining phrases...")
    unk_gazetteers = phrase_miner.get_unk_gazetteers(
        corpus=train_corpus
    )

    logging.info(f"Dumping quality tokens to {args.unk_gazetteers_path}")
    with open(args.unk_gazetteers_path, "w", encoding="utf-8") as f:
        [f.write(f"{gazet}\n") for gazet in unk_gazetteers]
    phrase_miner.compute_patterns({"UNK": unk_gazetteers})
    logging.info(f"Dumping training data to {args.train_conll_path}")
    phrase_miner.dump(train_corpus, args.train_conll_path)
    logging.info(f"Dumping validation data to {args.val_conll_path}")
    phrase_miner.dump(val_corpus, args.val_conll_path)
    logging.info("--- Done ---\n\n")

