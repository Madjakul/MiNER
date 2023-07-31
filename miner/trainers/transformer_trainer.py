# miner/trainers/transformer_trainer.py

from typing import Union
from transformers import TrainingArguments, Trainer

from miner.modules import RoBERTa, CamemBERT, Longformer
from miner.utils.data import TransformerDataset


class TransformerTrainer():
    """Wrapper for the transformers ``Trainer`` class to perform domain-
    specific MLM [1]_ before adding the NER head [2]_.

    Parameters
    ----------
    lm: ``miner.modules.RoBERTa``, ``miner.modules.CamemBERT``, ``miner.modules.Longformer``
        Language model checkpoint from **HuggingFace**.
    lm_path: ``str``
        Path to the local file that will contained the trained language model.
    lm_dataset: ``miner.utils.data.TransformerDataset``
        Iterable object containing the training and validation data.
    per_device_train_batch_size: ``int``
        Training batch size.
    seed: ``int``
        Integers used to initialized the weight of the LLM. Used for
        replicability.
    per_device_eval_batch_size: ``int``
        Validation batch size.
    num_train_epochs: ``int``
        Maximum number of training epochs.
    gradient_accumulation_steps: ``int``
        For how manys steps the gradient is accumulated.

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
        gradient_accumulation_steps: int, wandb: bool
    ):
        self.lm_path = lm_path
        self.training_args = TrainingArguments(
            output_dir=lm_path,
            # overwrite_output_dir=True,
            # do_train=True,
            do_eval=True,
            evaluation_strategy="epoch",
            # warmup_ratio=0.06,
            # learning_rate=5e-10,
            # eval_accumulation_steps=gradient_accumulation_steps,
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=per_device_eval_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            num_train_epochs=num_train_epochs,
            logging_strategy="epoch",
            save_strategy="no",
            seed=seed,
            # data_seed=seed,
            log_level="error",
            report_to="wandb" if wandb else "none",
            # fp16_full_eval=True,
            save_total_limit=1
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
        """Performs MLM to further pretrain a large language model.
        """
        self.trainer.train()
        self.trainer.save_model(self.lm_path)

