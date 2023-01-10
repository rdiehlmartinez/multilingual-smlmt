__author__ = "Richard Diehl Martinez"
""" Deals with the orchestration of model evaluation on a variety of NLU tasks """

import abc
import logging
import os
import torch
import wandb
import numpy as np

from collections import defaultdict
from ..datasets import (
    NLU_STANDARD_TASK_DATA_GENERATOR_MAPPING,
    NLU_FEW_SHOT_TASK_DATA_GENERATOR_MAPPING,
    NLU_CROSS_LINGUAL_TASK_DATA_GENERATOR_MAPPING,
)
from ..utils import num_gpus
from ..utils.data import ShuffledIterableDataset
from ..utils.evaluation import AccuracyMetric

from wandb.errors import UsageError
from multiprocessing import Pool, Manager

logger = logging.getLogger(__name__)

# Importing type hints
from multiprocessing import Lock, Queue
from typing import List, Callable, Union, Dict, Any
from configparser import ConfigParser
from ..metalearners import BaseLearner
from ..datasets import NLUDataset
from ..utils.evaluation import Metric


"""
Main evaluator class; orchestrates the evaluation of a model on a variety of NLU tasks
"""


class Evaluator(object):
    def __init__(self, config: ConfigParser, use_multiple_gpus: bool) -> None:
        """
        Sets up dataset generators for each eval task provided in the config. The
        Evaluator class is a general framework for calling the inference procedures
        of a learner, and computing relevant metrics for each task that is meant to
        be evaluated on.

        Args:
            * config: ConfigParser object containing the configuration for the model
                use_multiple_gpus: Whether or not to use multiple GPUs for evaluation
            * use_multiple_gpus: Whether or not to use multiple GPUs for evaluation
        """

        ### read in and initialize dataset_generators
        standard_tasks_str = config.get("EVALUATION", "standard_tasks", fallback="")
        few_shot_tasks_str = config.get("EVALUATION", "few_shot_tasks", fallback="")
        cross_lingual_tasks_str = config.get(
            "EVALUATION", "cross_lingual_tasks", fallback=""
        )

        if (
            standard_tasks_str == ""
            and few_shot_tasks_str == ""
            and cross_lingual_tasks_str == ""
        ):
            raise ValueError("No tasks provided for evaluation")

        standard_tasks = [
            task_str for task_str in standard_tasks_str.split(",") if task_str != ""
        ]
        few_shot_tasks = [
            task_str for task_str in few_shot_tasks_str.split(",") if task_str != ""
        ]
        cross_lingual_tasks = [
            task_str
            for task_str in cross_lingual_tasks_str.split(",")
            if task_str != ""
        ]

        # The data generators map a specific task string ('xnli') to a generator that yields
        # a tuple of finetune, eval datasets in different languages for that task

        self.standard_task_data_generators = {
            task: NLU_STANDARD_TASK_DATA_GENERATOR_MAPPING[task](config)
            for task in standard_tasks
        }

        self.few_shot_task_data_generators = {
            task: NLU_FEW_SHOT_TASK_DATA_GENERATOR_MAPPING[task](config)
            for task in few_shot_tasks
        }

        self.cross_lingual_task_data_generators = {
            task: NLU_CROSS_LINGUAL_TASK_DATA_GENERATOR_MAPPING[task](config)
            for task in cross_lingual_tasks
        }

        self.save_checkpoints = config.getboolean(
            "EXPERIMENT", "save_checkpoints", fallback=True
        )

        if self.save_checkpoints:
            # NOTE: checkpoints are saved by the pipeline when the evaluation is completed -
            # however if we are saving checkpoints the evaluator needs to mark when a new
            # best model is found so that the pipeline can save the model
            self.eval_run_tracker = defaultdict(list)

        self.use_wandb = config.getboolean("EXPERIMENT", "use_wandb", fallback=True)

        self.use_multiple_gpus = use_multiple_gpus

        if self.use_wandb:
            """
            The evaluation results for each task are stored in tables in wandb; for each task we
            store two tables: an overview table that maps training step number to the evaluation
            results at that training step and a second that for a given evaluation run, shows the
            finetuning process of the model i.e. maps finetuning steps to the evaluation results.
            """

            standard_overview_columns = [
                "run_id",
                "meta_train_step",
                "eval_language",
                "eval_loss",
                "eval_metric",
                "num_finetune_steps",
            ]

            standard_finetune_columns = [
                "run_id",
                "meta_train_step",
                "finetune_step",
                "eval_language",
                "finetune_loss",
                "dev_loss",
                "dev_metric",
            ]

            few_shot_columns = [
                "run_id",
                "meta_train_step",
                "eval_language",
                "eval_loss",
                "eval_metric",
                "num_finetune_steps",
                "k",  # FEW SHOT SPECIFIC
                "n",  # FEW SHOT SPECIFIC
            ]

            cross_lingual_columns = [
                "run_id",
                "meta_train_step",
                "eval_language",
                "eval_loss",
                "eval_metric",
                "num_finetune_steps",
                "finetune_language",  # CROSS LINGUAL SPECIFIC
            ]

            self.eval_tables = {}

            for task in standard_tasks:
                self.eval_tables[task + "_standard_overview"] = wandb.Table(
                    columns=standard_overview_columns
                )
                self.eval_tables[task + "_standard_finetune"] = wandb.Table(
                    columns=standard_finetune_columns
                )

            for task in few_shot_tasks:
                self.eval_tables[task + "_few_shot"] = wandb.Table(
                    columns=few_shot_columns
                )

            for task in cross_lingual_tasks:
                self.eval_tables[task + "_cross_lingual"] = wandb.Table(
                    columns=cross_lingual_columns
                )

    ### Entry point to running evaluation

    @staticmethod
    def _run_single_task(
        finetune_dataset: NLUDataset,
        eval_dataset: NLUDataset,
        learner: BaseLearner,
        num_task_batches: int,
        eval_type: str,
        metric: Metric,
        task_name: str,
        task_type: str,
        task_head_init_method: str,
        num_classes: int,
        batch_size: int,
        max_epochs: int,
        lr: float,
        device_queue: Union[Queue, None] = None,
    ) -> None:
        """
        Runs evaluation on a single task

        Args:
            * finetune_dataset: Dataset to finetune on
            * eval_dataset: Dataset to evaluate on
            * learner: Learner to use for evaluation
            * num_task_batches: Number of training batches we've seen so far
            * eval_type: Type of task to evaluate on (standard, few_shot, cross_lingual)
            * metric: Metric to use for evaluation
            * task_name: Name of task to evaluate on
            * task_type: Type of task to evaluate on
            * task_head_init_method: Method to use for initializing the task head
            * num_classes: Number of classes for the task
            * batch_size: Batch size to use for evaluation
            * max_epochs: Maximum number of epochs to finetune for
            * lr: Learning rate to use for finetuning
            * device_queue: Queue to use for getting the next available GPU device (only
                required if running the function via multiprocessing)

        Returns (if eval_type == "standard"):
            * eval_metric: The evaluation metric for the given task
            * total_eval_loss: A float indicating the loss of the model on the evaluation set
            * total_eval_loss: A float indicating the loss of the model on the evaluation set
        Returns (if eval_type != "standard", i.e. either fewshot or crosslingual ):
            * eval_metric: The evaluation metric for the given
            * total_eval_loss: A float indicating the loss of the model on the evaluation set
            * finetune_info: A list of dictionaries containing the following information:
                * finetune_loss: The loss of the model on the finetuning set
                * dev_loss: The loss of the model on the development set
                * dev_metric: The metric of the model on the development set
                * step_num: The number of steps the model has been finetuned for

        """

        # get the next available GPU device if running via multiprocessing
        if device_queue is not None:
            override_gpu_device = device_queue.get()
        else:
            override_gpu_device = None

        # NOTE: If we're not doing few_shot evaluation, then we are finetuning training on a
        # larger dataset and we might want to hold out a dev set to monitor finetuning; if
        # this is the case we wrap finetune_dataset in a ShuffledIterableDataset

        if eval_type != "few_shot":
            finetune_dataset = ShuffledIterableDataset(
                finetune_dataset,
                buffer_size=5000,
                hold_out_dev_batch=True,
                batch_size=batch_size,
            )

        eval_results = learner.run_evaluation(
            finetune_dataset,
            eval_dataset,
            eval_type,
            metric,
            task_type,
            task_head_init_method,
            num_classes,
            batch_size,
            max_epochs,
            lr,
            device=override_gpu_device,
            return_finetune_info=True if eval_type == "standard" else False,
        )

        if device_queue is not None:
            device_queue.put(override_gpu_device)

        return eval_results

    @staticmethod
    def zip_longest_pad_last(*lists):
        """
        Zips together lists of different lengths by padding the shorter lists with the last item
        in the list.

        modified from: https://stackoverflow.com/a/44250949
        """

        def g(l):
            for item in l:
                yield item
            while True:
                # continues to return the last item in the list
                yield item

        gens = [g(l) for l in lists]
        for _ in range(max(map(len, lists))):
            yield tuple(next(g) for g in gens)

    def log_results(
        self,
        task_name: str,
        eval_type: str,
        metric: Metric,
        num_task_batches: int,
        task_eval_results: List[Dict[str, Any]],
    ):
        """
        Logs out the results of the evaluation to the logger and wandb (if applicable)

        Args:
            * task_name: Name of the task
            * eval_type: Type of evaluation
            * metric: Metric used for evaluation
            * num_task_batches: Number of task batches that have been trained on so far
            * task_eval_results: A list of dictionaries containing the following information:
                * eval_language: The language of the evaluation set
                * eval_loss: A float indicating the loss of the model on the evaluation set
                * eval_metric: The evaluation metric for the given task
                * num_finetune_steps: The number of steps the model has been finetuned for

                Depending on the eval_type, each dictionaru might also contain the following:
                * finetune_info: A list of dictionaries containing the following information:
                    * finetune_loss: The loss of the model on the finetuning set
                    * dev_loss: The loss of the model on the development set
                    * dev_metric: The metric of the model on the development set
                    * step_num: The number of steps the model has been finetuned for
                    [NOTE that this is only included if eval_type == "standard"]
                * finetune_language: The language of the finetuning set
                [NOTE that this is only included if eval_type == "cross_lingual"]
                * k (int): The number of examples per class used for evaluation
                [NOTE that this is only included if eval_type == "few_shot"]
                * n (int): The number of classes used for evaluation
                [NOTE that this is only included if eval_type == "few_shot"]
        """

        eval_loss_sum = 0.0
        eval_metric_sum = 0.0
        num_finetune_steps_sum = 0.0

        for eval_result in task_eval_results:
            eval_loss_sum += eval_result["eval_loss"]
            eval_metric_sum += eval_result["eval_metric"]
            num_finetune_steps_sum += eval_result["num_finetune_steps"]

        eval_loss_mean = eval_loss_sum / len(task_eval_results)
        eval_metric_mean = eval_metric_sum / len(task_eval_results)
        num_finetune_steps_mean = num_finetune_steps_sum / len(task_eval_results)

        logger.info(f"\t\t ({task_name}) Avg. {metric.name}: {eval_metric_mean:.4f}")
        logger.info(f"\t\t ({task_name}) Avg. Loss: {eval_loss_mean:.4f}")
        logger.info(
            f"\t\t ({task_name}) Avg. Finetune Steps: {num_finetune_steps_mean:.4f}"
        )

        if self.use_wandb:

            # ==========================
            # 1) logging out the average eval metric and loss over all eval languages at the
            # curent num_task_batches training steps

            average_data_log = [
                wandb.run.id,
                num_task_batches,
                "average",  # averaged over all eval languages
                eval_loss_mean,
                eval_metric_mean,
                num_finetune_steps_mean,
            ]

            if eval_type == "standard":
                average_data_table_name = f"{task_name}_standard_overview"
            elif eval_type == "few_shot":
                average_data_table_name = f"{task_name}_few_shot"

                # Few shot evaluation expects a k and n value to be logged
                k = task_eval_results[0]["k"]
                n = task_eval_results[0]["n"]

                assert all(
                    [k == result["k"] for result in task_eval_results]
                ), "k values are not the same for all eval languages"
                assert all(
                    [n == result["n"] for result in task_eval_results]
                ), "n values are not the same for all eval languages"

                average_data_log.extend([k, n])
            elif eval_type == "cross_lingual":
                average_data_table_name = f"{task_name}_cross_lingual"

                # Cross lingual evaluation expects a finetune language to be logged
                finetune_lng = task_eval_results[0]["finetune_language"]

                assert all(
                    [
                        finetune_lng == result["finetune_language"]
                        for result in task_eval_results
                    ]
                ), "finetune languages are not the same for all eval languages"

                average_data_log.append(finetune_lng)

            self.eval_tables[average_data_table_name].add_data(*average_data_log)
            # ==========================

            # ==========================
            # 2) For each individual eval language, logging out the eval metrics to wandb
            for eval_results in task_eval_results:

                data_log = [
                    wandb.run.id,
                    num_task_batches,
                    eval_results["eval_language"],
                    eval_results["eval_loss"],
                    eval_results["eval_metric"],
                    eval_results["num_finetune_steps"],
                ]

                if eval_type == "standard":
                    data_table_name = f"{task_name}_standard_overview"
                elif eval_type == "few_shot":
                    data_table_name = f"{task_name}_few_shot"
                    data_log.extend([eval_results["k"], eval_results["n"]])
                elif eval_type == "cross_lingual":
                    data_table_name = f"{task_name}_cross_lingual"
                    data_log.append(eval_results["finetune_language"])

                self.eval_tables[data_table_name].add_data(*data_log)

                if eval_type == "standard":
                    # When evaluating in standard mode, we also log out the finetuning process
                    finetune_info = eval_results["finetune_info"]
                    eval_lng = eval_results["eval_language"]
                    for finetune_step_info in finetune_info:
                        # each finetune_step is a dictionary of metrics
                        self.eval_tables[task_name + "_standard_finetune"].add_data(
                            wandb.run.id,
                            num_task_batches,
                            finetune_step_info["step_num"],
                            eval_lng,
                            finetune_step_info["finetune_loss"],
                            finetune_step_info["dev_loss"],
                            finetune_step_info["dev_metric"],
                        )
            # ==========================

            # ==========================
            # 3) [NOTE: Optional] If we are evaluating in standard mode, we also log out
            # the averaged metrics at each finetune step of the finetuning process
            if eval_type == "standard":
                finetune_infos = [
                    eval_results["finetune_info"] for eval_results in task_eval_results
                ]
                for all_finetune_step_info in self.zip_longest_pad_last(
                    *finetune_infos
                ):
                    # zipping together the finetune info at each step for all the languages so that
                    # we can report the average metrics at each step across all languages

                    # NOTE: zip_longest_pad_last will pad the shorter lists with the last item in the
                    # list, as a result the current finetune_step_num is the max step_num across all
                    # languages (because some languages may have converged before others)

                    finetune_step_num = max(
                        [f_step["step_num"] for f_step in all_finetune_step_info]
                    )

                    finetune_step_loss_mean = sum(
                        [f_step["finetune_loss"] for f_step in all_finetune_step_info]
                    ) / len(all_finetune_step_info)

                    finetune_step_dev_loss_mean = sum(
                        [f_step["dev_loss"] for f_step in all_finetune_step_info]
                    ) / len(all_finetune_step_info)

                    finetune_step_dev_metric_mean = sum(
                        [f_step["dev_metric"] for f_step in all_finetune_step_info]
                    ) / len(all_finetune_step_info)

                    self.eval_tables[task_name + "_standard_finetune"].add_data(
                        wandb.run.id,
                        num_task_batches,
                        finetune_step_num,
                        "average",
                        finetune_step_loss_mean,
                        finetune_step_dev_loss_mean,
                        finetune_step_dev_metric_mean,
                    )

            # ==========================

            # 4) Push up tables to wandb
            if eval_type == "standard":
                wandb_overview_table_name = task_name + "_standard_overview_table"
                wandb_finetune_table_name = task_name + "_standard_finetune_table"

                # Reinitialization required due to ongoing bug, see:
                # https://github.com/wandb/wandb/issues/2981

                curr_wandb_overview_table = wandb.Table(
                    columns=self.eval_tables[task_name + "_standard_overview"].columns,
                    data=self.eval_tables[task_name + "_standard_overview"].data,
                )
                curr_wandb_finetune_table = wandb.Table(
                    columns=self.eval_tables[task_name + "_standard_finetune"].columns,
                    data=self.eval_tables[task_name + "_standard_finetune"].data,
                )

                wandb.log(
                    {
                        wandb_overview_table_name: curr_wandb_overview_table,
                        wandb_finetune_table_name: curr_wandb_finetune_table,
                    }
                )
            elif eval_type == "few_shot":
                wandb_table_name = task_name + "_few_shot_table"
                curr_wandb_table = wandb.Table(
                    columns=self.eval_tables[task_name + "_few_shot"].columns,
                    data=self.eval_tables[task_name + "_few_shot"].data,
                )
                wandb.log({wandb_table_name: curr_wandb_table})
            elif eval_type == "cross_lingual":
                wandb_table_name = task_name + "_cross_lingual_table"
                curr_wandb_table = wandb.Table(
                    columns=self.eval_tables[task_name + "_cross_lingual"].columns,
                    data=self.eval_tables[task_name + "_cross_lingual"].data,
                )
                wandb.log({wandb_table_name: curr_wandb_table})

    def run(self, learner: BaseLearner, num_task_batches: int = 0) -> bool:
        """
        Runs evaluation of the passed in learner on the self.tasks evaluation tasks.
        Loops over each of the evaluation tasks in self.tasks and for each task
        runs the learner's finetuning procedure and inference procedure. The inference
        procedure returns some predictions which are then used to compute metrics for each
        of the tasks.

        Args:
            * learner (subclass of BaseLearner): learning procedure that was used to train the model
            * num_task_batches (int): optional value of the current task batch number at which
                we are evaluating

        Returns:
            * new_best (bool): True if evaluation results are better than previous best results
        """

        logger.info("")
        logger.info("-" * 30)
        logger.info("Running evaluator")

        new_best = False

        eval_task_data_generators = {
            "standard": self.standard_task_data_generators,
            "few_shot": self.few_shot_task_data_generators,
            "cross_lingual": self.cross_lingual_task_data_generators,
        }

        for eval_type, eval_task_data_generators in eval_task_data_generators.items():
            logger.info("*" * 20)
            logger.info(f"Evaluation type: {eval_type}")

            for task_idx, (task_name, task_data_generator) in enumerate(
                eval_task_data_generators.items()
            ):
                logger.info(f"\t(Task {task_idx}) {task_name}")

                # Getting task-specific parameters
                task_type = task_data_generator.task_type
                task_head_init_method = task_data_generator.task_head_init_method
                num_classes = task_data_generator.num_classes

                batch_size = task_data_generator.batch_size
                max_epochs = task_data_generator.max_epochs
                lr = task_data_generator.lr

                if task_type == "classification":
                    metric = AccuracyMetric()
                else:
                    logger.exception(
                        f"Invalid task type: {task_type} for task: {task_name}"
                    )
                    raise Exception(
                        f"Invalid task type: {task_type} for task: {task_name}"
                    )

                # Arguments to pass into the finetuning-evaluation routine; expressed as a
                # tuple so that we can pass it into a process pool
                task_args = (
                    learner,
                    num_task_batches,
                    eval_type,
                    metric,
                    task_name,
                    task_type,
                    task_head_init_method,
                    num_classes,
                    batch_size,
                    max_epochs,
                    lr,
                )

                # If possible, we want to run finetuning-evaluation for each of the languages of
                # the given task in parallel. The evaluation results for each language are returned
                # in the dictionary structure; and the results for the task is correspondingly a
                # list of dictionaries.

                if self.use_multiple_gpus:

                    # If using multiple GPUs, we need to use a process pool with a queue to
                    # manage which GPU each process gets assigned to

                    with Manager() as manager:

                        device_queue = manager.Queue()
                        for i in range(num_gpus):
                            device_queue.put(torch.device(f"cuda:{i}"))

                        with Pool(num_gpus) as pool:

                            task_eval_results = pool.starmap(
                                self._run_single_task,
                                [
                                    (
                                        finetune_dataset,
                                        eval_dataset,
                                        *task_args,
                                        device_queue,
                                    )
                                    for finetune_dataset, eval_dataset in task_data_generator
                                ],
                            )

                else:
                    # If using a single GPU, we can just run the tasks sequentially
                    task_eval_results = []

                    for finetune_dataset, eval_dataset in task_data_generator:
                        eval_results = self._run_single_task(
                            finetune_dataset,
                            eval_dataset,
                            *task_args,
                            None,  # device_queue
                        )
                        task_eval_results.append(eval_results)

                # Logging out results
                self.log_results(
                    task_name, eval_type, metric, num_task_batches, task_eval_results
                )

                # If we are saving eval checkpoints, then do some book-keeping to keep track of
                # the best model
                if self.save_checkpoints:
                    eval_metrics = [
                        result["eval_metric"] for result in task_eval_results
                    ]
                    eval_metric_mean = sum(eval_metrics) / len(eval_metrics)
                    self.eval_run_tracker[
                        f"{task_name}.{eval_type}.{metric.name}"
                    ].append(eval_metric_mean)

                    if (
                        metric.summary(
                            self.eval_run_tracker[
                                f"{task_name}.{eval_type}.{metric.name}"
                            ]
                        )
                        == eval_metric_mean
                    ):
                        new_best = True

        ### If specified, possibly saving out checkpoint
        logger.info("*" * 20)
        logger.info("Finished evaluator")
        logger.info("-" * 30)
        logger.info("")

        return new_best
