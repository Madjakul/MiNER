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
            "--max_steps",
            type=int,
            help="Number of steps to train."
        )
        parser.add_argument(
            "--lm_accumulation_steps",
            type=int,
            help="Gradient and eval prediction accumulation."
        )
        parser.add_argument(
            "--wandb",
            action="store_true",
            help="Wether or not to use Weight and Biases."
        )
        parser.add_argument(
            "--project",
            type=str,
            nargs="?",
            const=None,
            help="Name of the wandb project."
        )
        parser.add_argument(
            "--entity",
            type=str,
            nargs="?",
            const=None,
            help="Wandb nickname."
        )
        parser.add_argument("--seed", type=int, help="Torch seed.")
        args, _ = parser.parse_known_args()
        return args


class TrainPartialArgParse():
    """Arguments to train the partial named entity recognizer.
    """

    @classmethod
    def parse_known_args(cls):
        parser = argparse.ArgumentParser()
        parser.add_argument("--lang", type=str, default="en")
        parser.add_argument(
            "--train_data_path",
            type=str,
            default="./data/bc5cdr/cdr_train.conll"
        )
        parser.add_argument(
            "--labels_path",
            type=str,
            default="./data/labels.txt"
        )
        parser.add_argument(
            "--lm_path",
            type=str,
            help="Local path/huggingface checkpoint to a language model."
        )
        parser.add_argument(
            "--sam",
            action="store_true",
            help="Wether to use or not Sharpness Aware Minimization."
        )
        parser.add_argument(
            "--max_length",
            type=int,
            help="Maximum sequence length."
        )
        parser.add_argument(
            "--ner_batch_size",
            type=int,
            help="Train and eval batch size."
        )
        parser.add_argument(
            "--lr",
            type=float,
            help="Initial learning rate."
        )
        parser.add_argument(
            "--clip",
            type=float,
            help="Gradient clipping norm."
        )
        parser.add_argument("--ner_epochs", type=int)
        parser.add_argument(
            "--ner_path",
            type=str,
            help="Path to the trained named entity recognizer."
        )
        parser.add_argument(
            "--dropout",
            type=float,
            help="Dropout between pretrained LM and partial CRF."
        )
        parser.add_argument(
            "--loss_fn",
            type=str,
            nargs="?",
            const=None,
            help="nll, c_nll or gce."
        )
        parser.add_argument(
            "--q",
            type=float,
            nargs="?",
            const=None,
            help="q hyperparameter for GCE loss."
        )
        parser.add_argument(
            "--val_data_path",
            type=str,
            nargs="?",
            const=None,
            help="Path to validation conll file."
        )
        parser.add_argument(
            "--wandb",
            action="store_true",
            help="Wether or not to use Weight and Biases."
        )
        parser.add_argument(
            "--project",
            type=str,
            nargs="?",
            const=None,
            help="Name of the wandb project."
        )
        parser.add_argument(
            "--entity",
            type=str,
            nargs="?",
            const=None,
            help="Wandb nickname."
        )
        parser.add_argument("--seed", type=int)
        parser.add_argument("--momentum", type=float)
        args, _ = parser.parse_known_args()
        return args


class TrainSmoothArgParse():
    """Arguments to train the smooth named entity recognizer.
    """

    @classmethod
    def parse_known_args(cls):
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--train_data_path",
            type=str,
        )
        parser.add_argument(
            "--labels_path",
            type=str,
            help="Path to the list of labels for the given dataset."
        )
        parser.add_argument(
            "--lm_path",
            type=str,
            help="Local path/huggingface checkpoint to a language model."
        )
        parser.add_argument(
            "--max_length",
            type=int,
            help="Maximum sequence length."
        )
        parser.add_argument(
            "--ner_batch_size",
            type=int,
            help="Train and eval batch size."
        )
        parser.add_argument(
            "--accumulation_steps",
            type=int,
            help="Gradient accumulation steps before an optimizer step."
        )
        parser.add_argument(
            "--lr",
            type=float,
            help="Initial learning rate."
        )
        parser.add_argument("--ner_epochs", type=int)
        parser.add_argument(
            "--partial_ner_path",
            type=str,
            help="Path to the trained partial named entity recognizer."
        )
        parser.add_argument(
            "--smooth_ner_path",
            type=str,
            help="Path to the trained smooth named entity recognizer."
        )
        parser.add_argument(
            "--dropout",
            type=float,
            help="Dropout between pretrained LM and partial CRF."
        )
        parser.add_argument(
            "--val_data_path",
            type=str,
            nargs="?",
            const=None,
            help="Path to validation conll file."
        )
        parser.add_argument(
            "--wandb",
            action="store_true",
            help="Wether or not to use Weight and Biases."
        )
        parser.add_argument(
            "--project",
            type=str,
            nargs="?",
            const=None,
            help="Name of the wandb project."
        )
        parser.add_argument(
            "--entity",
            type=str,
            nargs="?",
            const=None,
            help="Wandb nickname."
        )
        parser.add_argument("--seed", type=int)
        args, _ = parser.parse_known_args()
        return args


class TestPartialArgParse():
    """Arguments to test the partial named entity recognizer.
    """

    @classmethod
    def parse_known_args(cls):
        parser = argparse.ArgumentParser()
        parser.add_argument("--lang", type=str, default="en")
        parser.add_argument(
            "--test_data_path",
            type=str,
        )
        parser.add_argument(
            "--labels_path",
            type=str,
        )
        parser.add_argument(
            "--lm_path",
            type=str,
            help="Local path/huggingface checkpoint to a language model."
        )
        parser.add_argument(
            "--max_length",
            type=int,
            help="Maximum sequence length."
        )
        parser.add_argument(
            "--ner_path",
            type=str,
            help="Path to the trained named entity recognizer."
        )
        args, _ = parser.parse_known_args()
        return args


class TestSmoothArgParse():
    """Arguments to test the smooth named entity recognizer.
    """

    @classmethod
    def parse_known_args(cls):
        parser = argparse.ArgumentParser()
        parser.add_argument("--lang", type=str, default="en")
        parser.add_argument(
            "--test_corpus_path",
            type=str,
        )
        parser.add_argument(
            "--labels_path",
            type=str,
        )
        parser.add_argument(
            "--lm_path",
            type=str,
            help="Local path/huggingface checkpoint to a language model."
        )
        parser.add_argument(
            "--max_length",
            type=int,
            help="Maximum sequence length."
        )
        parser.add_argument(
            "--ner_path",
            type=str,
            help="Path to the trained named entity recognizer."
        )
        args, _ = parser.parse_known_args()
        return args

