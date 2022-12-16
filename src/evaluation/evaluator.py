__author__ = 'Richard Diehl Martinez'
""" Deals with the orchestration of model evaluation on a variety of NLU tasks """

import abc
import logging
import os
import torch
import wandb
import numpy as np

from collections import defaultdict
from ..datasets import NLUDataLoader, NLU_TASK_DATA_GENERATOR_MAPPING
from ..utils import num_gpus
from ..utils.data import ShuffledIterableDataset
from ..utils.evaluation import Metric

from wandb.errors import UsageError
from multiprocessing import Pool, Manager

logger = logging.getLogger(__name__)

# Importing type hints
from multiprocessing import Lock, Queue
from typing import List, Callable, Union, Dict
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
        tasks_str = config.get("EVALUATION", "tasks", fallback="")
        if tasks_str == "":
            logger.warning("Initializing evaluator with no tasks for evaluation")

        self.tasks = tasks_str.split(',')

        self.task_data_generators = {
            task: NLU_TASK_DATA_GENERATOR_MAPPING[task](config) for task in self.tasks
        }

        self.batch_size = config.getint("EVALUATION", "batch_size", fallback=256)

        self.max_epochs = config.getint("EVALUATION", "max_epochs", fallback=5)
 

        self.save_checkpoints = config.getboolean("EXPERIMENT", "save_checkpoints", fallback=True)

        if self.save_checkpoints:
            # NOTE: checkpoints are saved by the pipeline when the evaluation is completed - 
            # however if we are saving checkpoints the evaluator needs to mark when a new 
            # best model is found so that the pipeline can save the model 
            self.eval_run_tracker = defaultdict(list)

        self.use_wandb = config.getboolean('EXPERIMENT', 'use_wandb', fallback=True)

        self.use_multiple_gpus = use_multiple_gpus
        
    ### Entry point to running evaluation

    @staticmethod
    def _run_single_task(
        finetune_dataset: ShuffledIterableDataset,
        eval_dataset: NLUDataset,
        learner: BaseLearner,
        num_task_batches: int,
        metric: Metric,
        task: str,
        task_type: str,
        task_head_init_method: str,
        num_classes: int,
        batch_size: int,
        max_epochs: int,
        device_queue: Union[Queue, None] = None,
    ) -> None:
        """
        Runs evaluation on a single task

        Args: 
            finetune_dataset: Dataset to finetune on
            eval_dataset: Dataset to evaluate on
            learner: Learner to use for evaluation
            num_task_batches: Number of training batches we've seen so far
            metric: Metric to use for evaluation
            task: Name of task to evaluate on
            task_type: Type of task to evaluate on
            task_head_init_method: Method to use for initializing the task head
            num_classes: Number of classes for the task
            batch_size: Batch size to use for evaluation
            max_epochs: Maximum number of epochs to finetune for
            device_queue: Queue to use for getting the next available GPU device (only
                required if running the function via multiprocessing)
        """        

        # get the next available GPU device if running via multiprocessing
        if device_queue is not None:
            override_gpu_device = device_queue.get()
        else:
            override_gpu_device = None

        # Wrapping finetune dataset in a ShuffledIterableDataset (to prevent overfitting 
        # issues when finetuning on a small dataset); ShuffledIterableDataset also enables 
        # us to hold out a validation set from the finetune dataset to monitor finetuning
        finetune_dataset = ShuffledIterableDataset(
            finetune_dataset,
            buffer_size=5000,
            hold_out_dev_batch=True,
            batch_size=batch_size,
        )

        eval_metric, eval_loss, finetune_info = learner.run_evaluation(
            finetune_dataset,
            eval_dataset,
            metric,
            task_type,
            task_head_init_method,
            num_classes,
            max_epochs=max_epochs,
            batch_size=batch_size,
            device=override_gpu_device,
            return_finetune_info=True,
        )

        if device_queue is not None:
            device_queue.put(override_gpu_device)

        return (eval_dataset.language, eval_metric, eval_loss, finetune_info)


    def log_results(
        self,
        eval_lngs: List[str],
        eval_metrics: List[float],
        eval_losses: List[float],
        finetune_infos: List[Dict[str, List[float]]],
        task: str,
        task_idx: int,
        metric: Metric,
        num_task_batches: int,
    ): 
        """
        Logs out the results of the evaluation to the logger and wandb (if applicable)

        Args: 
            * eval_lngs: List of languages that were evaluated
            * eval_metrics: List of evaluation metrics for each language
            * eval_losses: List of evaluation losses for each language
            * finetune_infos: List of finetuning info for each language
            * task: Name of the task
            * task_idx: Index of the task being evaluated
            * metric: Metric used for evaluation
            * num_task_batches: Number of task batches that have been trained on so far
        """

        eval_metric_mean = sum(eval_metrics)/len(eval_metrics)
        eval_loss_mean = sum(eval_losses)/len(eval_losses)

        logger.info(f"\t (Task {task_idx}) Avg. {metric.name}: {eval_metric_mean:.4f}")
        logger.info(f"\t (Task {task_idx}) Avg. Loss: {eval_loss_mean:.4f}")

        if self.use_wandb:

            # Logging out task-wide metrics and loss vs. meta training steps
            wandb.define_metric(
                f"{task}.{metric.name}",
                step_metric="num_task_batches",
                summary=metric.summary.__name__
            )
            wandb.define_metric(
                f"{task}.loss",
                step_metric="num_task_batches",
                summary='min'
            )

            wandb.log({
                task: {
                    "loss": eval_loss_mean,
                    metric.name: eval_metric_mean,      
                },
                "num_task_batches": num_task_batches
            })
        
            # Logging out language-specific metrics
            
            for eval_lng, eval_metric, eval_loss, finetune_info in zip(
                eval_lngs, eval_metrics, eval_losses, finetune_infos
            ):
                # Defining metrics 

                wandb.define_metric(
                    f"{task}.{eval_lng}.{metric.name}",
                    step_metric="num_task_batches",
                    summary=metric.summary.__name__
                )
                wandb.define_metric(
                    f"{task}.{eval_lng}.loss",
                    step_metric="num_task_batches",
                    summary='min'
                )

                # Logging out metrics and loss for a given language vs. meta training steps
                wandb.log({
                    task: {
                        eval_lng: {
                            "loss": eval_loss,
                            metric.name: eval_metric,
                        },
                    },
                    "num_task_batches": num_task_batches   
                })


                # Logging out finetuning process metrics and loss vs. finetuning steps for 
                # a given language 
                step_metric_name = f"{task}.{eval_lng}.step_{num_task_batches}.finetune_step"

                wandb.define_metric(step_metric_name)
                    
                for key in finetune_info[0].keys():

                    if key == "finetune_step":
                        continue

                    wandb.define_metric( 
                        f"{task}.{eval_lng}.step_{num_task_batches}.{key}",
                        step_metric=step_metric_name,
                    )

                # Logging out finetuning process metrics and loss vs. finetuning steps

                for finetune_step_info in finetune_info:
                    # each finetune_step is a dictionary of metrics
                    for key, value in finetune_step_info.items():
                        if key == "finetune_step":
                            continue

                        wandb.log({
                            task: {
                                eval_lng: {
                                    f"step_{num_task_batches}": {
                                        "finetune_step": finetune_step_info["step_num"],
                                        key: value
                                    }
                                }  
                            }
                        })


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
        logger.info("-"*30)
        logger.info("Running evaluator")

        new_best = False

        for task_idx, task in enumerate(self.tasks):
            logger.info("*"*20)
            logger.info(f"(Task {task_idx}) Running evaluation task: {task}")

            ## Setting up params for the given task
            task_data_generator = self.task_data_generators[task]

            task_type = task_data_generator.task_type
            task_head_init_method = task_data_generator.task_head_init_method
            num_classes = task_data_generator.num_classes

            if task_type == "classification": 
                metric = AccuracyMetric()
            else: 
                logger.exception(f"Invalid task type: {task_params['task_type']} for task: {task}")
                raise Exception(f"Invalid task type: {task_params['task_type']} for task: {task}")

            task_args = (
                learner,
                num_task_batches,
                metric,
                task,
                task_type,
                task_head_init_method,
                num_classes,
                self.batch_size,
                self.max_epochs,
            )

            if self.use_multiple_gpus:

                # If using multiple GPUs, we need to use a process pool with a queue to
                # manage which GPU each process gets assigned to

                with Manager() as manager:

                    device_queue = manager.Queue()
                    for i in range(num_gpus):
                        device_queue.put(torch.device(f"cuda:{i}"))

                    with Pool(num_gpus) as pool:

                        task_results = pool.starmap(
                            self._run_single_task,
                            [
                                (
                                    finetune_dataset,
                                    eval_dataset,
                                    *task_args,
                                    device_queue,
                                )
                                for finetune_dataset, eval_dataset in task_data_generator
                            ]
                        )
                        
                        # unpack task results
                        eval_lngs, eval_metrics, eval_losses, finetune_infos = zip(*task_results)
            else: 
                # If using a single GPU, we can just run the tasks sequentially

                eval_lngs, eval_metrics, eval_losses, finetune_infos = [], []

                for finetune_dataset, eval_dataset in task_data_generator:
                    eval_lng, eval_metric, eval_loss, finetune_info = self._run_single_task(
                        finetune_dataset,
                        eval_dataset,
                        *task_args,
                        None, 

                    )
                    eval_lngs.append(eval_lng)
                    eval_metrics.append(eval_metric)
                    eval_losses.append(eval_loss)
                    finetune_infos.append(finetune_info)
                    
            # Logging out results 
            self.log_results(
                eval_lngs,
                eval_metrics,
                eval_losses,
                finetune_infos,
                task,
                task_idx, 
                metric,
                num_task_batches,
            )

            # If we are saving eval checkpoints, then do some book-keeping to keep track of
            # the best model 
            if self.save_checkpoints:
                eval_metric_mean = sum(eval_metrics)/len(eval_metrics)
                self.eval_run_tracker[f'{task}.{metric.name}'].append(eval_metric_mean)

                if metric.summary(self.eval_run_tracker[f'{task}.{metric.name}']) \
                        == eval_metric_mean:
                    new_best = True

        ### If specified, possibly saving out checkpoint 
        logger.info("*"*20)
        logger.info("Finished evaluator")
        logger.info("-"*30)
        logger.info("")

        return new_best
