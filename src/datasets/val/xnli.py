__author__ = "Richard Diehl Martinez"
""" Dataset class for iterating over XNLI data"""

import abc
import logging
import os
import torch
import numpy as np
import random

from collections import defaultdict

from transformers import XLMRobertaTokenizer

from torch.utils.data import TensorDataset

# Base class for NLU Tasks
from .nlutask import NLUTaskGenerator

# data processing imports
from lib.processors.xnli import XnliProcessor
from lib.processors.utils import convert_examples_to_features

logger = logging.getLogger(__name__)

# We always use the XLM sentencepiece tokenizer
tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-base")

# imports for type hints
from configparser import ConfigParser
from typing import Dict, Tuple, Union, List
from torch import Tensor

# to stop the huggingface tokenizer from giving the sequence longer than 512 warning
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)


class XNLIGenerator(NLUTaskGenerator):

    """
    A base class that stores parameters associated with the XNLI task and implements a generator
    that yields a dictionary of datasets for finetune, development and evaluation for each language
    in the XNLI corpus.
    """

    MAX_SEQ_LENGTH = 128

    def __init__(self, config: ConfigParser, eval_type: str) -> None:
        """
        We assume that the data for the xnli task has been downloaded as part of the
        XTREME cross-lingual benchmark (https://github.com/google-research/xtreme).
        """

        super().__init__(config, eval_type)

        # location of folder containing xnli data
        self.data_dir = config.get(self.config_name, "data_dir")

        if self.eval_type == "few_shot":
            self.K = config.getint(self.config_name, "k")
            self.N = self.num_classes  # alias for number of classes

    ### General properties of XNLI ###

    @property
    def num_classes(self) -> int:
        # contradiction, entailment, neutral
        return 3

    @property
    def task_type(self) -> str:
        return "classification"

    ### Helper functions for metric computation ###

    @property
    def metric_name(self) -> str:
        return "accuracy"

    def compute_metric(self, logits: Tensor, labels: Tensor) -> float:
        """Computes the accuracy of the model predictions"""
        predictions = torch.argmax(logits, dim=-1).tolist()
        labels = labels.tolist()
        accuracy = (np.array(predictions) == np.array(labels)).sum() / len(labels)
        return accuracy

    def metric_is_better(self, curr_metric: float, best_metric: float) -> bool:
        """Returns True if metric is better than best_metric"""
        return curr_metric > best_metric

    ### Helper functions for processing data to pass into model ###

    def process_batch(self, batch: List[Tensor]) -> Dict[str, Tensor]:
        """Processes a batch of data to be passed into the model"""

        model_inputs = {
            "input_ids": batch[0],
            "attention_mask": batch[1],
            "label_ids": batch[3],
        }

        return model_inputs

    ### Main data iterator ###

    def __iter__(self) -> Dict[str, Dict[str, Union[str, TensorDataset]]]:
        """
        Loads in the data for each languages, tokenizes it, and caches it to disk. If the data
        has already been cached, then it is loaded from disk. Using the data, creates a dictionary
        structure that contains the finetune, dev and eval datasets for each language; this is
        then yielded.

        Yields:
            A dictionary structure storing the finetune, dev and eval datasets for a single
            evaluation language.

                ex. for cross-lingual evaluation english -> french
                [
                    {
                        "finetune": {
                            "language": "en",
                            "dataset": torch.TensorDataset(...)
                        }
                        "dev": {
                            "language": "fr",
                            "dataset": torch.TensorDataset(...)
                        }
                        "eval": {
                            "language": "fr",
                            "dataset": torch.TensorDataset(...)
                        }
                    },
                    (... for other languages similar to above)
                ]

        """

        processor = XnliProcessor()

        for eval_language in self.eval_languages:

            data_dict = dict()

            # when using the standard evaluation type, the finetune, dev and eval languages are
            # the same
            for split in ["finetune", "dev", "eval"]:

                if self.eval_type == "few_shot" and split == "dev":
                    continue

                if self.eval_type == "cross_lingual" and split == "finetune":
                    split_language = "en"
                else:
                    split_language = eval_language

                cached_features_file = os.path.join(
                    self.data_dir,
                    "cache_{}_{}_{}_{}_{}{}".format(
                        split,
                        "xlm-roberta-base",
                        self.MAX_SEQ_LENGTH,
                        "xnli",
                        split_language,
                        "",
                    ),
                )

                # checking if the data has already been cached
                if os.path.exists(cached_features_file):
                    features = torch.load(cached_features_file)
                else:
                    # NOTE: the data has not been cached so we need to load it in

                    label_list = processor.get_labels()

                    if split == "finetune":
                        # if the language is english then we are not translating the data
                        if split_language == "en":
                            examples = processor.get_train_examples(
                                self.data_dir, language=split_language
                            )
                        else:
                            examples = processor.get_translate_train_examples(
                                self.data_dir, language=split_language
                            )
                    elif split == "dev":
                        examples = processor.get_dev_examples(
                            self.data_dir, language=split_language
                        )
                    elif split == "eval":
                        examples = processor.get_test_examples(
                            self.data_dir, language=split_language
                        )

                    features = convert_examples_to_features(
                        examples,
                        tokenizer,
                        label_list=label_list,
                        max_length=self.MAX_SEQ_LENGTH,
                        output_mode="classification",
                        pad_on_left=False,
                        pad_token=tokenizer.pad_token_id,
                        pad_token_segment_id=0,
                        lang2id=None,
                    )

                    torch.save(features, cached_features_file)

                # Building datasets

                if self.eval_type == "few_shot" and split == "finetune":
                    random.shuffle(features)

                    support_set_labels_count = defaultdict(int)
                    support_set_features = []
                    support_set_size = 0

                    for feature in features:
                        label_id = feature.label
                        if support_set_labels_count[label_id] < self.K:
                            support_set_labels_count[label_id] += 1
                            support_set_size += 1
                            support_set_features.append(feature)

                        if support_set_size == self.N * self.K:
                            break

                    features = support_set_features

                # TODO: If the dataset is fewshot we have to select just the fewshot examples
                # load into random sampler and randomly generate an N-way K-shot batch wrapped
                # into a dataset

                # Convert to Tensors and build dataset
                all_input_ids = torch.tensor(
                    [f.input_ids for f in features], dtype=torch.long
                )
                all_attention_mask = torch.tensor(
                    [f.attention_mask for f in features], dtype=torch.long
                )
                all_token_type_ids = torch.tensor(
                    [f.token_type_ids for f in features], dtype=torch.long
                )
                all_labels = torch.tensor([f.label for f in features], dtype=torch.long)

                dataset = TensorDataset(
                    all_input_ids, all_attention_mask, all_token_type_ids, all_labels
                )

                data_dict[split] = {"language": split_language, "dataset": dataset}

            yield data_dict


def main():
    config = ConfigParser()
    config.add_section("XNLI_FEW_SHOT")
    config.set(
        "XNLI_FEW_SHOT",
        "data_dir",
        "../../rds-personal-3CBQLhZjXbU/data/xtreme/download/xnli",
    )

    # evaluation languages
    config.set("XNLI_FEW_SHOT", "eval_languages", "en")

    # adding required training parameters
    config.set("XNLI_FEW_SHOT", "batch_size", "16")

    config.set("XNLI_FEW_SHOT", "max_epochs", "16")

    config.set("XNLI_FEW_SHOT", "lr", "1e-5")

    config.set("XNLI_FEW_SHOT", "task_head_init_method", "random")

    config.set("XNLI_FEW_SHOT", "eval_type", "few_shot")

    config.set("XNLI_FEW_SHOT", "k", "8")

    xnli_generator = XNLIGenerator(config, "few_shot")

    print("iterating over the generator")
    for data_dict in xnli_generator:
        pass


if __name__ == "__main__":
    main()
