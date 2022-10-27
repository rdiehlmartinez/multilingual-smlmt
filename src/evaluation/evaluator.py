__author__ = 'Richard Diehl Martinez'
""" Deals with the orchestration of model evaluation on a variety of NLU tasks """

import logging
import os
import torch
import wandb
import numpy as np

from collections import defaultdict
from ..datasets import NLUDataLoader, NLU_TASK_DATA_GENERATOR_MAPPING

logger = logging.getLogger(__name__)

# Importing type hints
from typing import List
from configparser import ConfigParser
from ..metalearners import BaseLearner

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

        self.batch_size = config.getint("EVALUATION", "batch_size", fallback=32)

        self.save_eval_checkpoints = config.getboolean(
            "EVALUATION",
            "save_eval_checkpoints",
            fallback=False
        )

        if self.save_eval_checkpoints:
            # possibly keep track of previous runs of the evaluator for checkpoint purposes
            self.eval_run_tracker = defaultdict(list)

        self.save_latest_checkpoint = config.getboolean(
            "EVALUATION",
            "save_latest_checkpoint",
            fallback=True
        )

        self.use_wandb = config.getboolean('EXPERIMENT', 'use_wandb', fallback=True)
        
        self.learner_method = config.get("LEARNER", "method")

    ### Helper methods for computing evaluation metrics

    @staticmethod
    def compute_accuracy(
        predictions: List[int],
        evaluation_dataloader: NLUDataLoader
    ) -> float:
        """ 
        Computes accuracy of predictions for the data of the evaluation_dataloader
        """
        labels = []
        for data_batch in evaluation_dataloader:
            labels.extend(data_batch['label_ids'].tolist())
        
        accuracy = (np.array(predictions) == np.array(labels)).sum()/len(labels)
        return accuracy

    ### Entry point to running evaluation

    def run(self, learner: BaseLearner, num_task_batches: int = 0) -> None:
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
        """

        logger.info("")
        logger.info("-"*30)
        logger.info("Running evaluator")

        mark_best_ckpt = False

        for idx, task in enumerate(self.tasks):
            logger.info("*"*20)
            logger.info(f"(Task {idx}) Running evaluation task: {task}")

            ## Setting up params for the given task
            task_data_generator = self.task_data_generators[task]

            task_type = task_data_generator.task_type
            n_labels = task_data_generator.n_labels

            if task_type == "classification": 
                compute_metric = self.compute_accuracy
                metric_name = "acc"
                metric_summary = 'max'
            else: 
                logger.exception(f"Invalid task type: {task_params['task_type']} for task: {task}")
                raise Exception(f"Invalid task type: {task_params['task_type']} for task: {task}")

            for subtask_idx, (support_batch, evaluation_dataset) in enumerate(task_data_generator):
                evaluation_lng = evaluation_dataset.language
                logger.info(f"\t Evaluating on: {evaluation_lng}")

                evaluation_dataloader = NLUDataLoader(
                    evaluation_dataset,
                    batch_size=self.batch_size
                )

                ### Running Finetuning
                inference_params = learner.run_finetuning(
                    support_batch, 
                    task_type,
                    n_labels,
                )

                ### Running Inference 
                predictions, eval_loss = learner.run_inference(
                    evaluation_dataloader,
                    task_type,
                    **inference_params,
                )

                ### Logging out metrics
                if self.use_wandb:
                    wandb.define_metric(f"{task}.{evaluation_lng}.{metric_name}",
                                        step_metric="num_task_batches", summary=metric_summary)
                    wandb.define_metric(f"{task}.{evaluation_lng}.loss",
                                        step_metric="num_task_batches", summary='min')

                # compute metrics using predictions 
                metric = compute_metric(predictions, evaluation_dataloader)
                logger.info(f"{metric_name}: {metric:.4f} - Eval Loss: {eval_loss:.4f}")
                if self.use_wandb:
                    wandb.log({task: {
                                    evaluation_lng: {
                                        "loss": eval_loss,
                                        metric_name: metric,
                                    },
                                },
                            "num_task_batches": num_task_batches
                            })


        ### If specified, possibly saving out checkpoint 
        logger.info("*"*20)
        logger.info("Finished evaluator")
        

        if self.save_latest_checkpoint or self.save_eval_checkpoints:
            if not self.use_wandb:
                logger.error("Cannot save model checkpoint because use_wandb set to False")
            else:
                checkpoint = {
                    'learner_state_dict': learner.state_dict(),
                    'optimizer_state_dict': learner.optimizer.state_dict(),
                }

                torch.save(checkpoint, os.path.join(wandb.run.dir, "latest-checkpoint.pt"))
                wandb.save("latest-checkpoint.pt")

                if mark_best_ckpt:
                    logger.info(f"Saving new best model checkpoint at step: {num_task_batches}")
                    torch.save(checkpoint, os.path.join(wandb.run.dir,\
                                            f"checkpoint-{num_task_batches}.pt"))
                    wandb.save(f"checkpoint-{num_task_batches}.pt")


        logger.info("-"*30)
        logger.info("")

