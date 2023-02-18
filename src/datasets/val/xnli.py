__author__ = "Richard Diehl Martinez"
""" Task class for iterating over XNLI data"""

import logging
import os
import random
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
from transformers import XLMRobertaTokenizer

import wandb
from lib.processors.utils import convert_examples_to_features

# data processing imports
from lib.processors.xnli import XnliProcessor

from ...utils import move_to_device

# Base class for NLU Tasks
from .nlutask import NLUTaskGenerator

logger = logging.getLogger(__name__)

# We always use the XLM sentencepiece tokenizer
tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-base")

# imports for type hints
from typing import Dict, List, Union

from torch import Tensor

from ...metalearners import BaseLearner
from ...models import BaseModel

# to stop the huggingface tokenizer from giving the sequence longer than 512 warning
logging.getLogger("transformers.tokenization_utils_base").setLevel(
    logging.ERROR
)


class XNLIGenerator(NLUTaskGenerator):

    """
    A base class that stores parameters associated with the XNLI task and implements a generator
    that yields a dictionary of datasets for finetune, development and evaluation for each language
    in the XNLI corpus.
    """

    MAX_SEQ_LENGTH = 128

    def __init__(self, eval_type: str) -> None:
        """
        We assume that the data for the xnli task has been downloaded as part of the
        XTREME cross-lingual benchmark (https://github.com/google-research/xtreme).
        """

        super().__init__(eval_type)

        # location of folder containing xnli data
        self.data_dir = wandb.config[self.config_name]["data_dir"]

        if self.eval_type == "few_shot":
            self.K = wandb.config[self.config_name]["k"]
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

    ### Helper functions for evaluating the model on the task ###

    def run_evaluation(
        self,
        finetune_model: BaseModel,
        finetune_task_head_weights: Dict[str, torch.nn.Parameter],
        learner: BaseLearner,
        split_dataset: TensorDataset,
        device: torch.device = None,
        **kwargs
    ) -> Dict[str, float]:
        """
        After finetuning on the finetuning set, runs this evaluation hook to evaluate the model
        on a given vaal or eval set.
        """

        all_logits = []
        all_labels = []

        total_loss = 0.0
        total_samples = 0

        # create a dataloader for the split dataset
        split_dataloader = DataLoader(
            split_dataset,
            sampler=SequentialSampler(split_dataset),
            batch_size=self.batch_size,
        )

        for split_batch in split_dataloader:
            with torch.no_grad():
                split_batch = self.process_batch(split_batch)
                split_batch = move_to_device(split_batch, device)

                split_outputs = finetune_model(
                    input_ids=split_batch["input_ids"],
                    attention_mask=split_batch["attention_mask"],
                )

                split_logits, split_loss = learner._compute_task_loss(
                    split_outputs,
                    split_batch,
                    finetune_task_head_weights,
                    task_type=self.task_type,
                )

            split_labels = split_batch["label_ids"]

            all_logits.append(split_logits)
            all_labels.append(split_labels)

            batch_size = split_logits.size(0)
            total_loss += (
                split_loss.detach().item() * batch_size
            )  # loss avg across batch
            total_samples += batch_size

        total_loss /= total_samples

        all_logits = torch.cat(all_logits, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        metric = self._compute_accuracy(all_logits, all_labels)

        return {
            "loss": total_loss,
            "metric": metric,
        }

    def _compute_accuracy(self, logits: Tensor, labels: Tensor) -> float:
        """Computes the accuracy of the model predictions"""
        predictions = torch.argmax(logits, dim=-1).tolist()
        labels = labels.tolist()
        accuracy = (np.array(predictions) == np.array(labels)).sum() / len(
            labels
        )
        return accuracy

    def metric_is_better(self, curr_metric: float, best_metric: float) -> bool:
        # NOTE: This must be implemented
        """Returns True if metric is better than best_metric"""
        return curr_metric > best_metric

    ### Helper functions for processing data to pass into model ###

    def process_batch(
        self, batch: List[Tensor], **kwargs
    ) -> Dict[str, Tensor]:
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
                all_labels = torch.tensor(
                    [f.label for f in features], dtype=torch.long
                )

                dataset = TensorDataset(
                    all_input_ids,
                    all_attention_mask,
                    all_token_type_ids,
                    all_labels,
                )

                data_dict[split] = {
                    "language": split_language,
                    "dataset": dataset,
                }

            yield data_dict
