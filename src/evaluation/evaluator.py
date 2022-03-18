__author__ = 'Richard Diehl Martinez'
''' Orchestration of model evaluation '''

import logging
import wandb
import numpy as np

from ..datasets import NLUDataLoader, NLU_DATASET_GENERATOR_MAPPING

logger = logging.getLogger(__name__)

TASK_EVALUATION_PARAMS = {
    "xnli": {
        "task_type": "classification",
        "n_classes": 3, 
    }
}

class Evaluator(object): 
    def __init__(self, config): 
        """ 
        Sets up dataset generators for each eval task provided in the config. The 
        Evaluator class is a general framework for calling the inference procedures
        of a learner, and computing relevant metrics for each task that is meant to 
        be evaluated on.
        """

        eval_tasks_str = config.get("EVALUATION", "tasks", fallback="")
        if eval_tasks_str == "":
            logger.warning("Initializing evaluator with no eval tasks")

        self.eval_tasks = eval_tasks_str.split(',')

        self.dataset_generators = {task: NLU_DATASET_GENERATOR_MAPPING[task](config)
                                    for task in self.eval_tasks}

        self.batch_size = config.getint("EVALUATION", "batch_size", fallback=32)

        # setting up metrics for logging of evaluation 
        for eval_task in self.eval_tasks:
            wandb.define_metric(f"{eval_task}*", step_metric="num_task_batches")

    @staticmethod
    def compute_accuracy(predictions, evaluation_dataloader):
        """ Computes accuracy of predictions extraction from data of the evaluation_dataloader """
        labels = []
        for data_batch in evaluation_dataloader:
            labels.extend(data_batch['label_ids'].tolist())
        
        accuracy = (np.array(predictions) == np.array(labels)).sum()/len(labels)
        return accuracy

    def run(self, learner, num_task_batches=0):
        """ 
        Runs evaluation of the passed in learner on the self.eval_tasks evaluation tasks. 
        Loops over each of the evaluation tasks in self.eval_tasks and for each eval_tasks 
        runs the learner's finetuning procedure and inference procedure. The inference 
        procedure returns some predictions which are then used to compute metrics for each
        of the tasks. 

        Args:
            * learner (subclass of BaseLearner): learning procedure 
            * num_task_batches (int): optional value of the current task batch number 
                at which we are evaluating
        """

        logger.info("#"*30)
        logger.info("Running evaluator")

        for idx, eval_task in enumerate(self.eval_tasks):
            logger.info("*"*20)
            logger.info(f"(Task {idx}) Running evaluation task: {eval_task}")

            eval_task_params = TASK_EVALUATION_PARAMS[eval_task]
            eval_task_type = eval_task_params['task_type']

            dataset_generator = self.dataset_generators[eval_task]

            if eval_task_type == "classification": 
                finetune_method = learner.run_finetuning_classification
                inference_method = learner.run_inference_classification
                compute_metric = self.compute_accuracy
                metric_name = "acc"
            else: 
                raise Exception(f"Invalid task type: {eval_task_type} for task: {eval_task}")

            eval_task_metrics = []
            eval_task_losses = []

            for subtask_idx, (finetune_dataset, evaluation_dataset) in enumerate(dataset_generator):
                finetune_language = finetune_dataset.language
                evaluation_language = evaluation_dataset.language
                logger.info(f"\t Finetuning on language: {finetune_language} - evaluating on language: {evaluation_language}")

                finetune_dataloader = NLUDataLoader(finetune_dataset, batch_size=self.batch_size)
                evaluation_dataloader = NLUDataLoader(evaluation_dataset, batch_size=self.batch_size)

                if not dataset_generator.use_few_shot_adaptation:
                    # we are doing zero-shot adaptation so the initial finetuning is always the same
                    if subtask_idx == 0:
                        inference_params = finetune_method(finetune_dataloader, **eval_task_params)
                else:
                    inference_params = finetune_method(finetune_dataloader, **eval_task_params)

                adaptation_batch = None
                if dataset_generator.adapt_on_eval:
                    # adapt on the first batch of the evaluation datalaoder
                    adaptation_batch = next(iter(evaluation_dataloader))

                predictions, eval_loss = inference_method(evaluation_dataloader, **inference_params, adaptation_batch=adaptation_batch)

                # compute metrics using predictions 
                metric = compute_metric(predictions, evaluation_dataloader)
                logger.info(f"\t \t {metric_name}: {metric:.4f} - Eval Loss: {eval_loss:.4f}")
                wandb.log({eval_task: {
                                evaluation_language: {
                                    "loss": eval_loss,
                                    metric_name: metric,
                                },
                            },
                           "num_task_batches": num_task_batches
                        })
            
                eval_task_metrics.append(metric)
                eval_task_losses.append(eval_loss)
                
            eval_task_metrics_mean = sum(eval_task_metrics)/len(eval_task_metrics)
            eval_task_loss_mean = sum(eval_task_losses)/len(eval_task_losses)
            logger.info(f"\t (Task {idx}) Avg. {metric_name}: {eval_task_metrics_mean:.4f} Avg Loss: {eval_task_loss_mean:.4f}")

        logger.info("*"*20)
        logger.info("Finished evaluator")
        logger.info("#"*30)