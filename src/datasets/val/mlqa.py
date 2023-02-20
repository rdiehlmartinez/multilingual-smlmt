__author__ = "Richard Diehl Martinez"
""" Task class for itearting over MLQA data"""

import logging
import os

import torch
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
from transformers import XLMRobertaTokenizer
from transformers.data.metrics.squad_metrics import (
    compute_predictions_logits,
    squad_evaluate,
)

import wandb

# data processing imports
from lib.processors.squad import (
    SquadResult,
    SquadV1Processor,
    squad_convert_examples_to_features,
)

from ...utils import move_to_device

# Base class for NLU tasks
from .nlutask import NLUTaskGenerator

logger = logging.getLogger(__name__)

# We always use the XLM sentencepiece tokenizer
tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-base")

# imports for type hints
from typing import Dict, List, Union

from torch import Tensor

from ...metalearners import BaseLearner
from ...models import BaseModel


class MLQAGenerator(NLUTaskGenerator):
    DOC_STRIDE = 128
    MAX_SEQ_LENGTH = 384
    MAX_QUERY_LENGTH = 64

    def __init__(self, eval_type: str) -> None:
        """
        We assume that the data for the MLQA task has been downloaded as part of the
        XTREME cross-lingual benchmark (https://github.com/google-research/xtreme).

        NOTE: The MLQA dataset cannot be used for few shot evaluation.
        """
        super().__init__(eval_type)

        self.train_data_dir = wandb.config[self.config_name]["train_data_dir"]
        self.dev_data_dir = wandb.config[self.config_name]["dev_data_dir"]
        self.eval_data_dir = wandb.config[self.config_name]["eval_data_dir"]

    @property
    def task_type(self) -> str:
        return "qa"

    ### Helper functions for metric computation ###
    @property
    def metric_name(self) -> str:
        return "f1"

    ### Helper functions for evaluating the model on the task ###

    def run_evaluation(
        self,
        finetune_model: BaseModel,
        finetune_task_head_weights: Dict[str, torch.nn.Parameter],
        learner: BaseLearner,
        split_dataset: TensorDataset,
        device: torch.device,
        split_language: str,
        split: str,
        **kwargs,
    ) -> Dict[str, float]:
        """
        After finetuning on the finetuning set, runs this evaluation hook to evaluate the model
        on a given val or eval set.
        """

        def to_list(tensor):
            return tensor.detach().cpu().tolist()

        ### Loading in the examples and features ###
        if split == "dev":
            input_dir = self.dev_data_dir
            filename = (
                f"dev-context-{split_language}-question-{split_language}.json"
            )
        elif split == "eval":
            input_dir = self.eval_data_dir
            filename = (
                f"test-context-{split_language}-question-{split_language}.json"
            )

        # NOTE: The cache exists because the data has already been created in the __iter__ loop
        cached_features_file = os.path.join(
            input_dir,
            "cached_{}_{}_{}_{}".format(
                split, "xlm-roberta-base", self.MAX_SEQ_LENGTH, split_language
            ),
        )
        features_and_dataset = torch.load(cached_features_file)
        features, _ = (
            features_and_dataset["features"],
            features_and_dataset["dataset"],
        )

        processor = SquadV1Processor()

        examples = processor.get_examples(
            input_dir, filename=filename, language=split_language
        )

        ### Cerating the dataloader and iterating over the data ###
        split_dataloader = DataLoader(
            split_dataset,
            sampler=SequentialSampler(split_dataset),
            batch_size=self.batch_size,
        )
        all_results = []

        total_loss = 0.0
        total_samples = 0

        for split_batch in split_dataloader:
            split_example_indices = split_batch[5]
            split_batch = self.process_batch(split_batch)

            with torch.no_grad():
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

                # Accumulating the loss
                batch_size = split_batch["input_ids"].size(0)
                total_loss += (
                    split_loss.item() * batch_size
                )  # loss averaged across the batch
                total_samples += batch_size

            start_logits, end_logits = split_logits.split(1, dim=-1)
            start_logits, end_logits = start_logits.squeeze(
                -1
            ), end_logits.squeeze(-1)

            # Iterating over the examples in the batch and adding the results to all_results
            for i, example_index in enumerate(split_example_indices):
                example_start_logits = to_list(start_logits[i])
                example_end_logits = to_list(end_logits[i])

                eval_feature = features[example_index.item()]
                unique_id = int(eval_feature.unique_id)
                result = SquadResult(
                    unique_id, example_start_logits, example_end_logits
                )

                all_results.append(result)

        predictions = compute_predictions_logits(
            examples,
            features,
            all_results,
            20,  # args.n_best_size,
            30,  # args.max_answer_length,
            False,  # args.do_lower_case,
            None,  # output_prediction_file,
            None,  # output_nbest_file,
            None,  # output_null_log_odds_file,
            False,  # args.verbose_logging,
            False,  # args.version_2_with_negative,
            0.0,  # args.null_score_diff_threshold,
            tokenizer,
        )

        # Compute the F1 and exact scores.
        results = squad_evaluate(examples, predictions)
        total_loss /= total_samples

        return {
            "metric": results["f1"]
            / 100,  # F1 is returned as a percentage; we want a decimal
            "loss": total_loss,
        }

    def metric_is_better(self, curr_metric: float, best_metric: float) -> bool:
        """Returns True if metric is better than best_metric"""
        return curr_metric > best_metric

    ### Helper function for processing data to pass into model ###
    def process_batch(self, batch: List[Tensor]) -> Dict[str, Tensor]:
        """Processes a batch of data to be passed into the model"""

        model_inputs = {
            "input_ids": batch[0],
            "attention_mask": batch[1],
            "start_positions": batch[3],
            "end_positions": batch[4],
        }

        return model_inputs

    ### Main iterator ###

    def __iter__(self) -> Dict[str, Dict[str, Union[str, TensorDataset]]]:
        # iterate over the specified languages

        processor = SquadV1Processor()

        for eval_language in self.eval_languages:
            data_dict = dict()

            for split in ["finetune", "dev", "eval"]:
                if split == "finetune":
                    input_dir = self.train_data_dir
                elif split == "dev":
                    input_dir = self.dev_data_dir
                elif split == "eval":
                    input_dir = self.eval_data_dir

                if self.eval_type == "cross_lingual" and split == "finetune":
                    split_language = "en"
                else:
                    split_language = eval_language

                # NEED TO GET EXAMPLES

                if split == "finetune" and split_language != "en":
                    input_dir = os.path.join(input_dir, "translate-train")

                cached_features_file = os.path.join(
                    input_dir,
                    "cached_{}_{}_{}_{}".format(
                        split,
                        "xlm-roberta-base",
                        self.MAX_SEQ_LENGTH,
                        split_language,
                    ),
                )

                if os.path.exists(cached_features_file):
                    features_and_dataset = torch.load(cached_features_file)
                    features, dataset = (
                        features_and_dataset["features"],
                        features_and_dataset["dataset"],
                    )
                else:
                    if split == "finetune":
                        if split_language == "en":
                            filename = "train-v1.1.json"
                        else:
                            # loading in the translated training data
                            filename = f"squad.translate.train.en-{split_language}.json"

                        examples = processor.get_examples(
                            input_dir,
                            filename=filename,
                            language=split_language,
                        )
                    else:
                        if split == "dev":
                            filename = f"dev-context-{split_language}-question-{split_language}.json"
                        else:
                            filename = f"test-context-{split_language}-question-{split_language}.json"

                        examples = processor.get_examples(
                            input_dir,
                            filename=filename,
                            language=split_language,
                        )

                    features, dataset = squad_convert_examples_to_features(
                        examples=examples,
                        tokenizer=tokenizer,
                        max_seq_length=self.MAX_SEQ_LENGTH,
                        doc_stride=self.DOC_STRIDE,
                        max_query_length=self.MAX_QUERY_LENGTH,
                        return_dataset="pt",
                        threads=12,
                        lang2id=None,
                    )

                    torch.save(
                        {"features": features, "dataset": dataset},
                        cached_features_file,
                    )

                data_dict[split] = {
                    "language": split_language,
                    "dataset": dataset,
                }

            yield data_dict
