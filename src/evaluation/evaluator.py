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
from ..utils.data import ShuffledIterableDataset

logger = logging.getLogger(__name__)

# Importing type hints
from typing import List, Callable
from configparser import ConfigParser
from ..metalearners import BaseLearner

"""
Helper classes for the evaluator; define interface for different types of evaluation metrics i.e. 
accuracy, F1, etc.
"""

class Metric(object, metaclass=abc.ABCMeta):

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Name of the metric"""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def summary(self) -> Callable:
        """Summary function to use for the metric"""
        raise NotImplementedError

    @staticmethod
    @abc.abstractmethod
    def __call__(self, predictions: List[int], labels: List[int]) -> float:
        """ 
        Computes the metric given the predictions and labels

        Args: 
            predictions: List of predictions
            labels: List of labels

        Returns: 
            metric: The metric value
            
        """
        raise NotImplementedError

class AccuracyMetric(Metric): 

    @property
    def name(self):
        return "accuracy"

    @property
    def summary(self):
        return max

    @staticmethod
    def __call__(
        predictions: List[int],
        labels: List[int]
    ) -> float:
        """ 
        Computes accuracy of predictions for the data of the eval_dataloader
        """        
        accuracy = (np.array(predictions) == np.array(labels)).sum()/len(labels)
        return accuracy

"""
Main evaluator class; orchestrates the evaluation of a model on a variety of NLU tasks
"""

class Evaluator(object): 
    def __init__(self, config: ConfigParser) -> None: 
        """ 
        Sets up dataset generators for each eval task provided in the config. The 
        Evaluator class is a general framework for calling the inference procedures
        of a learner, and computing relevant metrics for each task that is meant to 
        be evaluated on.
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
            # possibly keep track of previous runs of the evaluator for checkpoint purposes
            self.eval_run_tracker = defaultdict(list)

        self.use_wandb = config.getboolean('EXPERIMENT', 'use_wandb', fallback=True)
        
    ### Entry point to running evaluation

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

        for idx, task in enumerate(self.tasks):
            logger.info("*"*20)
            logger.info(f"(Task {idx}) Running evaluation task: {task}")

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

            task_metrics, task_losses = [], []

            for subtask_idx, (finetune_dataset, eval_dataset) in enumerate(task_data_generator):
                finetune_lng = finetune_dataset.language
                eval_lng = eval_dataset.language

                logger.info(f"\t Finetuning on: {finetune_lng} - Evaluating on: {eval_lng}")

                # Wrapping finetune dataset in a ShuffledIterableDataset (to prevent overfitting 
                # issues when finetuning on a small dataset); ShuffledIterableDataset also enables 
                # us to hold out a validation set from the finetune dataset to monitor finetuning
                finetune_dataset = ShuffledIterableDataset(
                    finetune_dataset,
                    buffer_size=5000,
                    hold_out_dev_batch=True,
                    batch_size=self.batch_size,
                )

                eval_metric, eval_loss = learner.run_evaluation(
                    finetune_dataset,
                    eval_dataset,
                    metric,
                    task_type,
                    task_head_init_method,
                    num_classes,
                    max_epochs=self.max_epochs,
                    batch_size=self.batch_size,
                )

                ### Logging out metrics
                if self.use_wandb:
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


                logger.info(f"{metric.name}: {eval_metric:.4f} - Eval Loss: {eval_loss:.4f}")
                if self.use_wandb:
                    wandb.log({
                        task: {
                            eval_lng: {
                                "loss": eval_loss,
                                metric.name: eval_metric,
                            },
                        },
                        "num_task_batches": num_task_batches   
                    })

                task_metrics.append(eval_metric)
                task_losses.append(eval_loss)
            
            task_metric_mean = sum(task_metrics)/len(task_metrics)
            task_loss_mean = sum(task_losses)/len(task_losses)
            
            if self.use_wandb:
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
                        "loss": task_loss_mean,
                        metric.name: task_metric_mean,      
                    },
                    "num_task_batches": num_task_batches
                })

                logger.info(f"\t (Task {idx}) Avg. {metric.name}: {task_metric_mean:.4f}")
                logger.info(f"\t (Task {idx}) Avg. Loss: {task_loss_mean:.4f}")

                # If we are saving eval checkpoints, then do some book-keeping to keep track of
                # the best model
                if self.save_checkpoints:
                    self.eval_run_tracker[f'{task}.{metric.name}'].append(task_metric_mean)

                    if metric.summary(self.eval_run_tracker[f'{task}.{metric.name}']) \
                            == task_metric_mean:
                        new_best = True

        ### If specified, possibly saving out checkpoint 
        logger.info("*"*20)
        logger.info("Finished evaluator")
        logger.info("-"*30)
        logger.info("")

        return new_best
