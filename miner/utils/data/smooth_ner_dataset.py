# miner/utils/data/smooth_ner_dataset.py

from typing import Literal, Optional, List

import transformers
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from miner.modules import PartialNER


class SmoothNERDataset(Dataset):
    """Dataset to feed the smooth NER with.

    Parameters
    ----------

    Attributes
    ----------
    """

    def __init__(
        self, partial_ner: PartialNER, corpus: List[List[str]],
        max_length: int, lang: Literal["en", "fr"],
        device: Optional[Literal["cpu", "cuda"]]="cpu"
    ):
        self.partial_ner = partial_ner
        self.corpus = corpus
        self.max_length = max_length
        self.device = device
        if lang == "fr" and max_length <= 512:
            self.tokenizer = AutoTokenizer.from_pretrained(
                "camembert-base", add_prefix_space=True
            )
        elif max_length > 512 and lang == "en":
            self.tokenizer = AutoTokenizer.from_pretrained(
                "allenai/longformer-base-4096", add_prefix_space=True
            )
        elif lang == "en" and max_length <= 512:
            self.tokenizer = AutoTokenizer.from_pretrained(
                "roberta-base", add_prefix_space=True
            )
        else:
            raise ValueError(
                f"Wrong combination of language ({lang}) and maximum sequence"
                + f" length ({max_length})."
            )


    def __len__(self):
        return len(self.corpus)

    def __getitem__(self, idx: int):
        x = self._tokenize(idx)
        y = self.enhanced_marginal_probabilities(x)
        return x, y

    def _tokenize(self, idx: int):
        inputs = self.tokenizer(
            self.corpus[idx],
            is_split_into_words=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return inputs.to(self.device)

    def enhanced_marginal_probabilities(
        self, inputs: transformers.BatchEncoding
    ):
        """Enhances high-confidence predictions while demote low-confidence
        ones via squaring and normalizing the current predictions on the
        original sequence [1]_.

        Parameters
        ----------

        Returns
        -------

        References
        ----------
        ..  [1] Xie, Junyuan, Ross Girshick, and Ali Farhadi. "Unsupervised
                deep embedding for clustering analysis." International
                conference on machine learning. PMLR, 2016.
        """
        tmp_p = self.partial_ner.marginal_probabilities(inputs) # * mask
        mask = inputs["attention_mask"].unsqueeze(2).expand(tmp_p.shape[0], tmp_p.shape[1], tmp_p.shape[2])
        p = tmp_p * mask
        squared_p = p ** 2
        denominator = squared_p.sum(dim=1)
        enhanced_p = squared_p / denominator.unsqueeze(1)
        return enhanced_p

