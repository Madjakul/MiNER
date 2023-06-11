# preprocess_miner.py

import logging
import argparse

from miner.utils import logging_config
from miner.utils.data import PhraseMiner
from miner.utils.data import preprocessing as pp


logging_config()


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lang",
        type=str,
        help="Language of the corpus {'en', 'fr'}"
    )
    parser.add_argument(
        "--corpus_path",
        type=str,
        help="Path to the raw corpus."
    )
    parser.add_argument(
        "--conll_path",
        type=str,
        help="Path and name of the conll distant training data."
    )
    parser.add_argument(
        "--gazetteers_path",
        type=str,
        help="Path to the dictionaries"
    )
    parser.add_argument(
        "--unk_gazetteers_path",
        type=str,
        help="Path and name to the file containing the mined entities."
    )
    args = parser.parse_args()

    logging.info("=== Mining Phrases ===")

    logging.info(f"Reading training data from {args.corpus_path}")
    with open(args.corpus_path, "r", encoding="utf-8") as f:
        corpus = f.read().splitlines()
    corpus = [text.lower() for text in corpus]

    logging.info(f"Loading gazetteers from {args.gazetteers_path}")
    phrase_miner = PhraseMiner(lang=args.lang)
    gazetteers = pp.load_gazetteers(args.gazetteers_path)
    phrase_miner.compute_patterns(gazetteers)
    logging.info("Mining phrases...")
    unk_gazetteers = phrase_miner.get_unk_gazetteers(
        corpus=corpus
    )

    logging.info(f"Dumping quality tokens to {args.unk_gazetteers_path}")
    with open(args.unk_gazetteers_path, "w", encoding="utf-8") as f:
        [f.write(f"{gazet}\n") for gazet in unk_gazetteers]
    phrase_miner.compute_patterns({"UNK": unk_gazetteers})
    logging.info(f"Dumping training data to {args.conll_path}")
    phrase_miner.dump(corpus, args.conll_path)
    logging.info("--- Done ---\n\n")

