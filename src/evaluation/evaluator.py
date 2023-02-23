__author__ = "Richard Diehl Martinez"
""" Deals with the orchestration of model evaluation on a variety of NLU tasks """

import logging
from multiprocessing import Manager, Pool

import torch

import wandb

from ..datasets import NLU_TASK_GENERATOR_MAP
from ..utils import num_gpus

logger = logging.getLogger(__name__)

from multiprocessing import Queue
from typing import Any, Dict, List, Tuple, Union

# Importing type hints
from torch.utils.data import Dataset

from ..datasets import NLUTaskGenerator
from ..metalearners import BaseLearner

"""
Main evaluator class; orchestrates the evaluation of a model on a variety of NLU tasks
"""


class Evaluator(object):
    def __init__(self, use_multiple_gpus: bool) -> None:
        """
        Sets up dataset generators for each eval task provided in the config. The
        Evaluator class is a general framework for calling the inference procedures
        of a learner, and computing relevant metrics for each task that is meant to
        be evaluated on.

        Args:
            * use_multiple_gpus: Whether or not to use multiple GPUs for evaluation
        """

        wandb.define_metric("avg_eval_metric", step_metric="num_task_batches")
        wandb.define_metric(
            "avg_eval_finetune_steps", step_metric="num_task_batches"
        )

        ### read in and initialize dataset_generators
        if "standard_tasks" not in wandb.config["EVALUATION"]:
            wandb.config["EVALUATION"]["standard_tasks"] = ""
        if "few_shot_tasks" not in wandb.config["EVALUATION"]:
            wandb.config["EVALUATION"]["few_shot_tasks"] = ""
        if "cross_lingual_tasks" not in wandb.config["EVALUATION"]:
            wandb.config["EVALUATION"]["cross_lingual_tasks"] = ""

        standard_tasks_str = wandb.config["EVALUATION"]["standard_tasks"]
        few_shot_tasks_str = wandb.config["EVALUATION"]["few_shot_tasks"]
        cross_lingual_tasks_str = wandb.config["EVALUATION"][
            "cross_lingual_tasks"
        ]

        if (
            standard_tasks_str == ""
            and few_shot_tasks_str == ""
            and cross_lingual_tasks_str == ""
        ):
            raise ValueError("No tasks provided for evaluation")

        standard_tasks = [
            task_str
            for task_str in standard_tasks_str.split(",")
            if task_str != ""
        ]
        few_shot_tasks = [
            task_str
            for task_str in few_shot_tasks_str.split(",")
            if task_str != ""
        ]
        cross_lingual_tasks = [
            task_str
            for task_str in cross_lingual_tasks_str.split(",")
            if task_str != ""
        ]

        # The data generators map a specific task string ('xnli') to a generator that yields
        # a tuple of finetune, dev and eval datasets in different languages for that task

        self.standard_task_generators = {
            task: NLU_TASK_GENERATOR_MAP[task]("standard")
            for task in standard_tasks
        }

        self.few_shot_task_generators = {
            task: NLU_TASK_GENERATOR_MAP[task]("few_shot")
            for task in few_shot_tasks
        }

        self.cross_lingual_task_generators = {
            task: NLU_TASK_GENERATOR_MAP[task]("cross_lingual")
            for task in cross_lingual_tasks
        }

        self.save_best_checkpoints = wandb.config["EXPERIMENT"]["save_best_checkpoints"]

        if self.save_best_checkpoints:
            # NOTE: checkpoints are saved by the pipeline when the evaluation is completed -
            # however if we are saving best so-far checkpoints, the evaluator needs to mark when
            # a new best model is found so that the pipeline can save that model later
            self.best_eval_tracker = {}

        self.use_multiple_gpus = use_multiple_gpus

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
        learner: BaseLearner,
        task_generator: NLUTaskGenerator,
        task_data: Dict[str, Dict[str, Union[str, Dataset]]],
        device_queue: Union[Queue, None] = None,
    ) -> None:
        """
        Runs evaluation on a single task

        Args:
            * task_generator: Task generator class that encapsulates the task (the finetune,
                dev and eval datasets are yielded from this class)
            * finetune_dataset: Dataset to finetune on
            * dev_dataset: Dataset to use for development
            * eval_dataset: Dataset to evaluate on
            * device_queue: Queue to use for getting the next available GPU device (only
                required if running the function via multiprocessing)

        Returns:
            * eval_results: A dictionary containing the evaluation results for the task, varies
                depending on the evaluation type (standard, few_shot or cross_lingual)
        """

        # get the next available GPU device if running via multiprocessing
        if device_queue is not None:
            override_gpu_device = device_queue.get()
        else:
            override_gpu_device = None

        eval_results = task_generator.run_finetune_evaluation(
            learner,
            task_data,
            device=override_gpu_device,
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

    def _average_task_eval_results(
        self,
        task_eval_results: List[Dict[str, Any]],
    ) -> Tuple[float, float, float]:
        """
        Averages the evaluation results for a task across all languages.

        Returns:
            * eval_loss_mena: The average loss across all languages
            * eval_metric_mean: The average metric across all languages
            * num_finetune_steps_mean: The average number of finetuning steps across all languages
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
        num_finetune_steps_mean = num_finetune_steps_sum / len(
            task_eval_results
        )

        return (eval_loss_mean, eval_metric_mean, num_finetune_steps_mean)

    def log_results(
        self,
        task_name: str,
        eval_type: str,
        metric_name: str,
        num_task_batches: int,
        task_eval_results: List[Dict[str, Any]],
    ):
        """
        Logs out the results of the evaluation to the logger and wandb

        Args:
            * task_name: Name of the task
            * eval_type: Type of evaluation
            * metric: Name of the evaluation metric (e.g. "accuracy")
            * num_task_batches: Number of task batches that have been trained on so far
            * task_eval_results: A list of dictionaries containing the following information:
                * eval_language: The language of the evaluation set
                * eval_loss: A float indicating the loss of the model on the evaluation set
                * eval_metric: The evaluation metric for the given task
                * num_finetune_steps: The number of steps the model has been finetuned for

                Depending on the eval_type, each dictionary might also contain the following:
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

        (
            eval_loss_mean,
            eval_metric_mean,
            num_finetune_steps_mean,
        ) = self._average_task_eval_results(task_eval_results)

        logger.info(
            f"\t\t ({task_name}) Avg. {metric_name}: {eval_metric_mean:.4f}"
        )
        logger.info(f"\t\t ({task_name}) Avg. Loss: {eval_loss_mean:.4f}")
        logger.info(
            f"\t\t ({task_name}) Avg. Finetune Steps: {num_finetune_steps_mean:.4f}"
        )

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
                    self.eval_tables[
                        task_name + "_standard_finetune"
                    ].add_data(
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
                eval_results["finetune_info"]
                for eval_results in task_eval_results
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
                    [
                        f_step["finetune_loss"]
                        for f_step in all_finetune_step_info
                    ]
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
                columns=self.eval_tables[
                    task_name + "_standard_overview"
                ].columns,
                data=self.eval_tables[task_name + "_standard_overview"].data,
            )
            curr_wandb_finetune_table = wandb.Table(
                columns=self.eval_tables[
                    task_name + "_standard_finetune"
                ].columns,
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

        eval_type_task_generators_map = {
            "few_shot": self.few_shot_task_generators,
            "standard": self.standard_task_generators,
            "cross_lingual": self.cross_lingual_task_generators,
        }

        # across all types of evaluation, all tasks and all languages
        all_eval_metric_means = []
        all_num_finetune_steps_mean = []

        for (
            eval_type,
            task_generators,
        ) in eval_type_task_generators_map.items():
            logger.info("*" * 20)
            logger.info(f"Evaluation type: {eval_type}")

            for task_idx, (task_name, task_generator) in enumerate(
                task_generators.items()
            ):
                # Make each type of task run its own evaluation
                logger.info("-" * 20)

                logger.info(f"\t(Task {task_idx}) {task_name}")

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
                                        learner,
                                        task_generator,
                                        task_data,
                                        device_queue,
                                    )
                                    for task_data in task_generator
                                ],
                            )

                else:
                    # If using a single GPU, we can just run the tasks sequentially
                    task_eval_results = []

                    for task_data in task_generator:
                        eval_results = self._run_single_task(
                            learner,
                            task_generator,
                            task_data,
                            None,  # device_queue
                        )
                        task_eval_results.append(eval_results)

                # Logging out results
                self.log_results(
                    task_name,
                    eval_type,
                    task_generator.metric_name,
                    num_task_batches,
                    task_eval_results,
                )

                # The eval metric mean and mean number of finetune steps for this given task
                # (i.e. averaged across all languages in the task)
                (
                    _,
                    task_eval_metric_mean,
                    task_num_finetune_steps_mean,
                ) = self._average_task_eval_results(task_eval_results)

                all_eval_metric_means.append(task_eval_metric_mean)
                all_num_finetune_steps_mean.append(
                    task_num_finetune_steps_mean
                )

                # If we are saving eval checkpoints, then do some book-keeping to keep track of
                # the best model so-far
                if self.save_best_checkpoints:
                    eval_metrics = [
                        result["eval_metric"] for result in task_eval_results
                    ]
                    eval_metric_mean = sum(eval_metrics) / len(eval_metrics)

                    eval_key = (
                        f"{task_name}.{eval_type}.{task_generator.metric_name}"
                    )

                    if eval_key not in self.best_eval_tracker:
                        self.best_eval_tracker[eval_key] = eval_metric_mean
                        new_best = True
                    elif task_generator.metric_is_better(
                        eval_metric_mean, self.best_eval_tracker[eval_key]
                    ):
                        self.best_eval_tracker[eval_key] = eval_metric_mean
                        new_best = True

        # averaging together the eval metric means and mean number of finetune steps across all tasks
        all_eval_metric_mean = sum(all_eval_metric_means) / len(
            all_eval_metric_means
        )
        all_num_finetune_steps_mean = sum(all_num_finetune_steps_mean) / len(
            all_num_finetune_steps_mean
        )
        # logging these out to wandb

        wandb.log(
            {
                "avg_eval_metric": all_eval_metric_mean,
                "avg_eval_finetune_steps": all_num_finetune_steps_mean,
            }
        )

        ### If specified, possibly saving out checkpoint
        logger.info("*" * 20)
        logger.info("Finished evaluator")
        logger.info("-" * 30)
        logger.info("")

        return new_best
