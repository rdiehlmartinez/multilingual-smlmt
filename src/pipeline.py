__author__ = "Richard Diehl Martinez"
""" Wrapper class for training and evaluating a model using a given meta learning technique """

import sys

sys.path.insert(0, "../lib")

import logging
import os

# Importing type hints
from typing import Union

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from transformers import AdamW, get_linear_schedule_with_warmup

import wandb

from .datasets import MetaDataLoader, MetaDataset

# used to get vocab size for model
from .datasets.train.metadataset import VOCAB_SIZE as XLMR_VOCAB_SIZE
from .evaluation import Evaluator
from .metalearners import MAML, BaselineLearner
from .models import XLMR
from .utils import device as BASE_DEVICE
from .utils import num_gpus

logger = logging.getLogger(__name__)


class Pipeline(object):
    """
    Orchestrates model loading, training and evaluation using a specific type of (meta-)learner.
    """

    def __init__(self, run_id: int, resume_num_task_batches: int = 0) -> None:
        """Initialize base model and meta learning method based on a config

        NOTE: The optional keyword argument (resume_num_task_batches) should never be manually set,
        rather it is passed in automatically by the program if it encounters a time expiration
        error and thus spawns a new job to continue running the program.

        Args:
            * run_id: id of the run (our internal id, not wandb's)
            * resume_num_task_batches (int): number of task batches to resume training from
        """

        self.run_id = run_id

        self.mode = wandb.config["EXPERIMENT"]["mode"]
        if self.mode not in ["train", "inference"]:
            raise Exception(
                f"Invalid pipeline run mode: {self.mode} - must be 'train' or 'inference'"
            )

        # whether to save intermediary checkpoints; either way, we always save the latest checkpoint
        self.save_best_checkpoints = wandb.config["EXPERIMENT"]["save_best_checkpoints"]

        # Setting device
        self.base_device = BASE_DEVICE
        self.use_multiple_gpus = (
            self.base_device == torch.device("cuda") and num_gpus > 1
        )
        # setting num_task_batches before learner, to inform learner if we are resuming training
        # or starting fresh
        self.num_task_batches = (
            resume_num_task_batches if resume_num_task_batches else 0
        )

        # setting up metrics for logging to wandb
        # counter tracks number of batches of tasks seen by metalearner
        wandb.define_metric("num_task_batches")

        # Possibly loading in a checkpoint file
        self.checkpoint = self.load_checkpoint()

        if self.mode == "train":
            # Setting up metadataset and meta dataloading (only for training)
            self.meta_dataset = MetaDataset()

            self.use_smlmt_labels = wandb.config["LANGUAGE_TASK"][
                "use_smlmt_labels"
            ]

            self.meta_dataloader = MetaDataLoader(
                self.meta_dataset, use_smlmt_labels=self.use_smlmt_labels
            )

        # setting base model
        self.base_model_name = wandb.config["BASE_MODEL"]["name"]
        self.base_model = self.load_model(self.base_model_name)

        # setting learner
        self.learner_method = wandb.config["LEARNER"]["method"]
        self.learner = self.load_learner(self.learner_method)

        if self.mode == "train":
            # setting meta optimization and training hyper-parameters
            self.num_tasks_per_iteration = wandb.config["PIPELINE"][
                "num_tasks_per_iteration"
            ]

            self.eval_every_n_iteration = wandb.config["PIPELINE"][
                "eval_every_n_iteration"
            ]
            self.max_task_batch_steps = wandb.config["PIPELINE"][
                "max_task_batch_steps"
            ]

            # setting meta learning rate
            self.meta_lr = wandb.config["PIPELINE"]["meta_lr"]

            # setting up the optimizer and learning rate scheduler for meta learning
            self.meta_optimizer = self.setup_meta_optimizer()

            self.meta_lr_scheduler_method = wandb.config["PIPELINE"][
                "meta_lr_scheduler_method"
            ]

            self.meta_lr_scheduler = self.setup_meta_lr_scheduler(
                self.meta_lr_scheduler_method
            )

        # setting evaluator
        if "EVALUATION" in wandb.config:
            self.evaluator = Evaluator(
                use_multiple_gpus=self.use_multiple_gpus
            )

    ### -- Initialization helper functions -- ###

    def _log_parameters(self):
        """
        Helper functionality for logging out parameters and hyperparameters of the pipeline
        """
        logger.debug("")
        logger.debug("*" * 40)
        logger.debug("PIPELINE PARAMETERS")
        logger.debug("")
        for section, section_dict in wandb.config.items():
            if not isinstance(section_dict, dict):
                # if section is not a dictionary, then it has been added in by wandb and is not
                # a parameter we set (all our parameters are nested in dictionaries)
                continue

            logger.debug(f"\t {section} PARAMETERS: ")
            for key, value in section_dict.items():
                logger.debug(f"\t\t * {key}: {value}")
        logger.debug("*" * 40)
        logger.debug("")

    def load_checkpoint(self):
        """
        Posibly loads in a checkpoint file if either provided in config or if we are resuming
        training from a previous job.
        """

        # if num_task_batches is 0 at the start of training, then we are resuming training
        if self.num_task_batches > 0:
            checkpoint_file = "latest-checkpoint.pt"
            checkpoint_run = None
        else:
            if "checkpoint_file" in wandb.config["LEARNER"]:
                checkpoint_file = wandb.config["LEARNER"]["checkpoint_file"]
                checkpoint_run = wandb.config["LEARNER"]["checkpoint_run"]
            else:
                checkpoint_file = None
                checkpoint_run = None

        if checkpoint_file:
            logger.info(f"Loading in checkpoint file: {checkpoint_file}")
            wandb_checkpoint = wandb.restore(
                checkpoint_file, run_path=checkpoint_run
            )
            checkpoint = torch.load(wandb_checkpoint.name)
            os.rename(
                os.path.join(wandb.run.dir, checkpoint_file),
                os.path.join(wandb.run.dir, "loaded_checkpoint.pt"),
            )
            return checkpoint

        return None

    def load_model(self, base_model_name: str) -> torch.nn.Module:
        """
        Helper function for reading in base model, should be intialized with the
        from_kwargs() class method. NOTE: We curently only support XLM-R models. To change this
        requires adding a new model class and slightly reworking the metadataset class to use
        a different type of tokenizer.

        Args:
            * base_model_name (str): name of base model to load

        Returns:
            * model (torch.nn.Module): base model to be used for meta learning
        """
        model_kwargs = wandb.config["BASE_MODEL"]

        if base_model_name == "xlm_r":
            model_cls = XLMR
        else:
            logger.exception(f"Invalid base model type: {base_model_name}")
            raise Exception(f"Invalid base model type: {base_model_name}")

        model = model_cls.from_kwargs(**model_kwargs)

        return model

    def load_learner(
        self, learner_method: str
    ) -> Union[MAML, BaselineLearner]:
        """
        Helper function for reading in (meta) learning procedure

        Args:
            * learner_method (str): name of learner to load

        Returns:
            * learner (either MAML or BaselineLearner): learner to be used for meta learning
        """

        learner_kwargs = wandb.config["LEARNER"]
        del learner_kwargs["method"]

        if self.mode == "train":
            if self.use_smlmt_labels:
                learner_kwargs["lm_head_n"] = wandb.config["LANGUAGE_TASK"][
                    "n"
                ]
            else:
                # size of the tokenizer vocab; NOTE that we currently only support XLM-R
                learner_kwargs["lm_head_n"] = XLMR_VOCAB_SIZE

        if learner_method == "maml":
            learner_cls = MAML
        elif learner_method == "baseline":
            learner_cls = BaselineLearner
        else:
            logger.exception(f"Invalid learner method: {learner_method}")
            raise Exception(f"Invalid learner method: {learner_method}")

        learner = learner_cls(
            base_model=self.base_model,
            base_device=self.base_device,
            seed=wandb.config["EXPERIMENT"]["seed"],
            **learner_kwargs,
        )

        if self.checkpoint is not None:
            learner.load_state_dict(
                self.checkpoint["learner_state_dict"], strict=False
            )

        return learner

    def setup_meta_optimizer(self) -> Optimizer:
        """
        Helper function for setting up meta optimizer and optionally an associated learning
        rate scheduler.

        Returns:
            * meta_optimizer (Optimizer): meta optimizer to be used for meta learning
        """
        meta_optimizer = AdamW(
            self.learner.outerloop_optimizer_param_groups(), lr=self.meta_lr
        )
        meta_optimizer.zero_grad()

        if self.checkpoint is not None:
            meta_optimizer.load_state_dict(
                self.checkpoint["optimizer_state_dict"]
            )
        return meta_optimizer

    def setup_meta_lr_scheduler(
        self, meta_lr_scheduler_method
    ) -> Union[LambdaLR, None]:
        """
        Helper function for setting up meta scheduler and optionally an associated learning
        rate scheduler.

        Args:
            * meta_lr_scheduler_method (str): name of meta learning rate scheduler to use

        Returns:
            * meta_lr_scheduler (_LRScheduler or None): meta learning rate scheduler
        """

        if meta_lr_scheduler_method is not None:
            if meta_lr_scheduler_method == "linear":
                meta_lr_scheduler = get_linear_schedule_with_warmup(
                    self.meta_optimizer,
                    num_warmup_steps=0.1 * self.max_task_batch_steps,
                    num_training_steps=self.max_task_batch_steps,
                )
            else:
                raise Exception(
                    f"Invalid meta scheduler method: {self.meta_lr_scheduler_method}"
                )

            if self.checkpoint is not None:
                meta_lr_scheduler.load_state_dict(
                    self.checkpoint["scheduler_state_dict"]
                )
        else:
            meta_lr_scheduler = None

        return meta_lr_scheduler

    ### --- Saving and checkpointing helpers --- ###

    def save_checkpoint(self, checkpoint_file: str) -> None:
        """
        Helper function for saving a checkpoint of the learner and optimizer state dicts.

        Args:
            * checkpoint_file (str): name of checkpoint file to save
        """
        checkpoint = {
            "learner_state_dict": self.learner.state_dict(),
            "optimizer_state_dict": self.meta_optimizer.state_dict(),
        }

        if self.meta_lr_scheduler is not None:
            checkpoint[
                "scheduler_state_dict"
            ] = self.meta_lr_scheduler.state_dict()

        torch.save(checkpoint, os.path.join(wandb.run.dir, checkpoint_file))
        wandb.save(checkpoint_file, policy="now")

    def _track_training_progress(self) -> None: 
        """ 
        Over the course of training, we want to track the progress of the model - we have no
        guarantee that the model will finish training in the time allotted by the scheduler. We 
        track the progress so far by keeping a runfile that contains the current task batch number
        along with the run id that is stored in wandb. This allows us to resume training from the
        correct task batch number.
        """

        if not os.path.exists("run_files"):
            os.mkdir("run_files")

        with open(f"run_files/{self.run_id}.runfile", "w+") as f:
            f.write(str(self.num_task_batches) + '\n')
            f.write(str(wandb.run.id))

        self.save_checkpoint("latest-checkpoint.pt")

    ### --- Meta training loop helper --- ###

    def meta_optimizer_step(self, grad_norm_constant: int) -> None:
        """
        Helper function for performing meta optimization step. Normalizes the gradients by the
        value of grad_norm_constant and performs a step on the meta optimizer. Optionally performs
        a step on the meta learning rate scheduler.

        Args:
            * grad_norm_constant (int): value to normalize gradients by; typically the number of
                tasks in the current task batch
        """

        for param_group in self.learner.outerloop_optimizer_param_groups():
            for param in param_group["params"]:
                param.grad /= grad_norm_constant

        self.meta_optimizer.step()

        if self.meta_lr_scheduler is not None:
            self.meta_lr_scheduler.step()

        self.meta_optimizer.zero_grad()

    def __call__(self) -> None:
        """
        Train or evaluate the self.base_model via the self.learner training procedure
        on data stored in self.meta_dataloader
        """

        self._log_parameters()

        ### --------- Inference Mode (no model training) ----------

        if self.mode == "inference":
            logger.info("### RUNNING PIPELINE IN INFERENCE MODE ###")

            if not hasattr(self, "evaluator"):
                logger.error(
                    "Need to specify an evaluator to run in inference-only mode"
                )
                return

            self.evaluator.run(self.learner)

            # evaluator logs out avg_eval_metric and avg_eval_finetune_steps both of which 
            # track num_task_batches as their step_metric, so we need to log the associated 
            # num_task_batches here
            wandb.log({
                "num_task_batches": self.num_task_batches,
            })

            logger.info("### PIPELINE FINISHED ###")
            return

        ### -------------------- Training Mode --------------------

        logger.info("### RUNNING PIPELINE IN TRAINING MODE ###")

        ### Setting up tracking variables and w&b metrics

        # counter tracks loss over an entire batch of tasks
        task_batch_loss = 0

        # metric for logging training data
        wandb.define_metric(
            "train_loss", step_metric="num_task_batches", summary="min"
        )

        if self.learner_method != "baseline":
            # any meta-learning approach will want to track the learned learning rates
            wandb.define_metric(
                "classifier_lr", step_metric="num_task_batches"
            )

            # for inner layers we need to track lr per layer
            for layer in self.learner.inner_layers_lr:
                wandb.define_metric(
                    f"inner_layer_{layer}_lr",
                    step_metric="num_task_batches",
                )

        if (
            wandb.config["PIPELINE"]["run_initial_eval"]
            and self.num_task_batches == 0
        ):
            # num_task_batches would only ever not be 0 if we're resuming training because of
            # previous timeout failure, in that case don't run initial eval
            if not hasattr(self, "evaluator"):
                logger.warning(
                    "Evaluation missing in config - skipping evaluator run"
                )
            else:
                self.evaluator.run(self.learner, num_task_batches=0)

        if self.num_task_batches == 0: 
            # logging out initial information before training starts 

            if self.learner_method != "baseline":
                wandb.log(
                    {"classifier_lr": self.learner.classifier_lr.item()}
                )

                for (
                    layer_num,
                    layer,
                ) in self.learner.inner_layers_lr.items():
                    wandb.log(
                        {f"inner_layer_{layer_num}_lr": layer.item()}
                    )

            wandb.log({
                "num_task_batches": self.num_task_batches,
            })


        # if we are resuming training, we need to set the task_sample_idx_shift_factor
        task_sample_idx_shift_factor = self.num_task_batches * self.num_tasks_per_iteration

        ### --- Meta training loop --- ###

        for _task_sample_idx, task_batch in enumerate(self.meta_dataloader):

            task_sample_idx = _task_sample_idx + task_sample_idx_shift_factor
            logger.debug(
                f"\t (Task Sample Idx {task_sample_idx}) Language: {task_batch[0]}"
            )

            ## Basic training with just a single GPU
            task_name, support_batch_list, query_batch = task_batch

            task_loss = self.learner.run_train_loop(
                support_batch_list, query_batch
            )
            task_loss = (
                task_loss / self.num_tasks_per_iteration
            )  # normalizing loss
            task_batch_loss += task_loss

            if (_task_sample_idx + 1) % self.num_tasks_per_iteration == 0:
                ##### NOTE: Just finished a batch of tasks -- taking a global (meta) update step

                self.num_task_batches += 1

                # single GPU: taking optimizer step
                self.meta_optimizer_step(
                    grad_norm_constant=self.num_tasks_per_iteration
                )

                ### Logging out training results
                logger.info(
                    f"No. batches of tasks processed: {self.num_task_batches}"
                )
                logger.info(f"\t(Meta) training loss: {task_batch_loss}")

                if self.learner_method != "baseline":
                    # wandb logging info for any meta-learner
                    wandb.log(
                        {"classifier_lr": self.learner.classifier_lr.item()}
                    )

                    for (
                        layer_num,
                        layer,
                    ) in self.learner.inner_layers_lr.items():
                        wandb.log(
                            {f"inner_layer_{layer_num}_lr": layer.item()}
                        )

                wandb.log(
                    {
                        "train_loss": task_batch_loss,
                        "num_task_batches": self.num_task_batches,
                    },
                )

                task_batch_loss = 0

                ### possibly run evaluation of the model
                if (
                    self.eval_every_n_iteration
                    and self.num_task_batches % self.eval_every_n_iteration
                    == 0
                ):
                    if not hasattr(self, "evaluator"):
                        logger.warning(
                            "Evaluation missing in config - skipping evaluator run"
                        )
                    else:
                        new_best = self.evaluator.run(
                            self.learner,
                            num_task_batches=self.num_task_batches,
                        )

                        if self.save_best_checkpoints:
                            if new_best:
                                self.save_checkpoint(
                                    f"checkpoint-{self.num_task_batches}.pt"
                                )

                if self.num_task_batches % self.max_task_batch_steps == 0:
                    # NOTE: stop training if we've done max_task_batch_steps global update steps
                    break

                self._track_training_progress()

        ### Model done training - final clean up before exiting

        logger.info("### PIPELINE FINISHED ###")
        self.save_checkpoint("final.pt")
