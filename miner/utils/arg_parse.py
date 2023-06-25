# miner/utils/arg_parse.py

import argparse


class PreprocessArgParse():
    """Arguments for preprocessing.
    """

    @classmethod
    def parse_known_args(cls):
        parser = argparse.ArgumentParser(
            description="Arguments to preprocess MiNER's data."
        )
        parser.add_argument(
            "--lang",
            type=str,
            required=True,
            help="Language of the corpus {'en', 'fr'}."
        )
        parser.add_argument(
            "--corpus_path",
            type=str,
            required=True,
            help="Path to the raw corpus."
        )
        parser.add_argument(
            "--conll_path",
            type=str,
            required=True,
            help="Path and name of the generated conll training data."
        )
        parser.add_argument(
            "--gazetteers_path",
            type=str,
            required=True,
            help="Path to the dictionaries"
        )
        parser.add_argument(
            "--unk_gazetteers_path",
            type=str,
            nargs="?",
            const=None,
            help="Path and name to the file containing the mined entities."
        )
        parser.add_argument(
            "--label_completion",
            action="store_true",
            help="Weak label completion to generate unknown tokens."
        )
        args, _ = parser.parse_known_args()
        return args


class PretrainArgParse():
    """Arguments to pretrain the language model.
    """

    @classmethod
    def parse_known_args(cls):
        parser = argparse.ArgumentParser(
            description="Arguments to preprocess MiNER's data."
        )
        parser.add_argument(
            "--lang",
            type=str,
            help="Language of the corpus {'en', 'fr'}"
        )
        parser.add_argument(
            "--train_corpus_path",
            type=str,
        )
        parser.add_argument(
            "--val_corpus_path",
            type=str,
        )
        parser.add_argument(
            "--max_length",
            type=int,
            help="Maximum sequence length."
        )
        parser.add_argument(
            "--lm_path",
            type=str,
            help="Path to language model."
        )
        parser.add_argument("--seed", type=int, help="Torch seed.")
        parser.add_argument(
            "--mlm_probability",
            type=float,
            help="Token masking probability"
        )
        parser.add_argument(
            "--lm_train_batch_size",
            type=int,
            help="Train and eval batch size."
        )
        parser.add_argument(
            "--lm_epochs",
            type=int,
            help="Number of epochs to train."
        )
        parser.add_argument(
            "--lm_accumulation_steps",
            type=int,
            help="Gradient and eval prediction accumulation."
        )
        args, _ = parser.parse_known_args()
        return args

