# miner/trainers/transformer_trainer.py

from typing import Union
from transformers import TrainingArguments, Trainer

from miner.modules import RoBERTa, CamemBERT, Longformer
from miner.utils.data import TransformerDataset


class TransformerTrainer():
    """Wrapper for the transformers ``Trainer`` class to perform domain-
    specific MLM _[1] before adding the NER head _[2].

    Parameters
    ----------
    lm: ``miner.modules.RoBERTa``, ``miner.modules.CamemBERT`` or ``miner.modules.Longformer``
        Language model checkpoint from **HuggingFace**.

    Attributes
    ----------
    training_args: ``transformers.TrainingArguments``
        Stores the hyperparameters to pretrain the large language model.
    trainer: ``transformers.Trainer``
        Stores the datasets to perform MLM and feed the large language model
        with.

    References
    ----------
    ..  [1] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova.
        2019. Bert: Pre-training of deep bidirectional Transformers for
        language understanding. (May 2019). Retrieved January 31, 2023 from
        https://arxiv.org/abs/1810.04805v2
    ..  [2] Suchin Gururangan et al. 2020. Don't stop pretraining: Adapt
        language models to domains and tasks. (May 2020). Retrieved January 31,
        2023 from https://arxiv.org/abs/2004.10964
    """

    def __init__(
        self, lm: Union[RoBERTa, CamemBERT, Longformer], lm_path: str,
        lm_dataset: TransformerDataset, per_device_train_batch_size: int,
        seed: int, per_device_eval_batch_size: int, num_train_epochs: int,
        gradient_accumulation_steps: int
    ):
        self.lm_path = lm_path
        self.training_args = TrainingArguments(
            output_dir=lm_path,
            do_eval=True,
            evaluation_strategy="epoch",
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=per_device_eval_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            num_train_epochs=num_train_epochs,
            logging_strategy="epoch",
            save_strategy="epoch",
            seed=seed,
            load_best_model_at_end=True,
            greater_is_better=False,
            log_level="error",
            report_to=None,
            save_total_limit=2
        )
        self.trainer = Trainer(
            model=lm.model,
            tokenizer=lm_dataset.tokenizer,
            args=self.training_args,
            data_collator=lm_dataset.data_collator,
            train_dataset=lm_dataset.mlm_ds["train"],   # type: ignore
            eval_dataset=lm_dataset.mlm_ds["valid"]     # type: ignore
        )

    def train(self):
        """Perform MLM to further pretrain a large language model.
        """
        self.trainer.train()
        self.trainer.save_model(self.lm_path)

