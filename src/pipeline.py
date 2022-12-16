__author__ = 'Richard Diehl Martinez'
""" Wrapper class for training and evaluating a model using a given meta learning technique """

import sys 
sys.path.insert(0, '../lib')

import typing
import torch
import logging 
import wandb
import time 
import os 

from transformers import AdamW, get_linear_schedule_with_warmup

from .models import XLMR
from .metalearners import MAML, BaselineLearner
from .evaluation import Evaluator
from .utils import device as DEFAULT_DEVICE, num_gpus
from .datasets import MetaDataset, MetaDataLoader

# used to get vocab size for model
from .datasets.train.metadataset import VOCAB_SIZE as XLMR_VOCAB_SIZE

# Importing type hints  
from configparser import ConfigParser
from typing import Union
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

logger = logging.getLogger(__name__)

class Pipeline(object):
    """
    Orchestrates model loading, training and evaluation using a specific type of (meta-)learner.
    """

    def __init__(self, config: ConfigParser, resume_num_task_batches: int = 0) -> None:
        """ Initialize base model and meta learning method based on a config 
        
        NOTE: The optional keyword argument (resume_num_task_batches) should never be manually set,
        rather it is passed in automatically by the program if it encounters a time expiration
        error and thus spawns a new job to continue running the program.

        Args: 
            * config (ConfigParser): config file containing all necessary information for
                loading in the base model and meta learning method
            * resume_num_task_batches (int): number of task batches to resume training from
        """

        # config params need to be accessed by several methods
        self.config = config

        self.mode = config.get("EXPERIMENT", "mode", fallback='train')
        if self.mode not in ['train', 'inference']:
            raise Exception(
                f"Invalid pipeline run mode: {self.mode} - must be 'train' or 'inference'"
            )
        
        # whether to log out information to w&b
        self.use_wandb = config.getboolean('EXPERIMENT', 'use_wandb', fallback=True)
        self.save_checkpoints = config.getboolean("EXPERIMENT", "save_checkpoints", fallback=True)

        # Setting device 
        self.base_device = config.get("PIPELINE", "device", fallback=DEFAULT_DEVICE)
        self.use_multiple_gpus = self.base_device == torch.device("cuda") and num_gpus > 1
        logger.info(f"Running pipeline on device: {self.base_device}")
        if self.base_device == torch.device("cuda"):
            logger.info(f"Number of GPUs available: {num_gpus}")

        # setting num_task_batches before learner, to inform learner if we are resuming training 
        # or starting fresh 
        self.num_task_batches = resume_num_task_batches if resume_num_task_batches else 0

        if self.use_wandb:
            # setting up metrics for logging to wandb
            # counter tracks number of batches of tasks seen by metalearner
            wandb.define_metric("num_task_batches")
        
        # Possibly loading in a checkpoint file
        self.checkpoint = self.load_checkpoint()
    
        if self.mode == 'train':
            # Setting up metadataset and meta dataloading (only for training)
            self.meta_dataset = MetaDataset(config)

            self.use_smlmt_labels = config.getboolean(
                "LANGUAGE_TASK",
                "use_smlmt_labels",
                fallback=False
            )

            self.meta_dataloader = MetaDataLoader(
                self.meta_dataset,
                use_smlmt_labels=self.use_smlmt_labels
            )

        # setting base model 
        self.base_model_name = config.get("BASE_MODEL", "name")
        self.base_model = self.load_model(self.base_model_name)

        # setting learner 
        self.learner_method = self.config.get("LEARNER", "method")
        self.learner = self.load_learner(self.learner_method)
        
        if self.mode == 'train':
            # setting meta optimization and training hyper-parameters
            self.num_tasks_per_iteration = self.config.getint(
                "PIPELINE",
                "num_tasks_per_iteration",
                fallback=1
            )
            self.eval_every_n_iteration = self.config.getint(
                "PIPELINE",
                "eval_every_n_iteration",
                fallback=0
            )
            self.max_task_batch_steps = self.config.getint(
                "PIPELINE",
                "max_task_batch_steps",
                fallback=1
            )

            # setting meta learning rate 
            self.meta_lr = config.getfloat("PIPELINE", "meta_lr", fallback=1e-3)

            # setting up the optimizer and learning rate scheduler for meta learning
            self.meta_optimizer = self.setup_meta_optimizer()
            
            self.meta_lr_scheduler_method = self.config.get(
                "PIPELINE",
                "meta_lr_scheduler_method",
                fallback=None
            )
            self.meta_lr_scheduler = self.setup_meta_lr_scheduler(self.meta_lr_scheduler_method)


        # setting evaluator 
        if 'EVALUATION' in config:
            self.evaluator = Evaluator(config, use_multiple_gpus=self.use_multiple_gpus)

    ### -- Initialization helper functions -- ###

    def _log_parameters(self): 
        """ 
        Helper functionality for logging out parameters and hyperparameters of the pipeline
        """ 
        logger.debug("*"*40)
        logger.debug("Pipeline Parameters")
        logger.debug(f"\t General parameters: ")
        logger.debug(f"\t\t * Use Wandb: {self.use_wandb}")
        logger.debug(f"\t\t * Save Checkpoints: {self.save_checkpoints}")
        logger.debug(f"\t\t * Use Multiple GPUs: {self.use_multiple_gpus}")
        logger.debug(f"\t\t * Base Device: {self.base_device}")

        if self.mode == "train":
            logger.debug(f"\t Training parameters: ")
            logger.debug(f"\t\t * Meta Learning Rate: {self.meta_lr}")
            logger.debug(f"\t\t * Meta Learning Rate Scheduler: {self.meta_lr_scheduler_method}")
            logger.debug(f"\t\t * Max Task Batch Steps: {self.max_task_batch_steps}")
            logger.debug(f"\t\t * Num Tasks Per Iteration: {self.num_tasks_per_iteration}")
            logger.debug(f"\t\t * Eval Every N Iterations: {self.eval_every_n_iteration}")

        logger.debug(f"\t Modeling parameters: ")
        logger.debug(f"\t\t * Learner Method: {self.learner_method}")
        logger.debug(f"\t\t * Base Model: {self.base_model_name}")
        logger.debug("*"*40)

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
            checkpoint_file = self.config.get("LEARNER", "checkpoint_file", fallback="")
            checkpoint_run = self.config.get("LEARNER", "checkpoint_run", fallback="")

        if checkpoint_file:
            if not self.use_wandb:
                logger.warning("Could not load in checkpoint file, use_wandb is set to False")
            else:
                logger.info(f"Loading in checkpoint file: {checkpoint_file}")
                wandb_checkpoint = wandb.restore(checkpoint_file, run_path=checkpoint_run)
                checkpoint = torch.load(wandb_checkpoint.name)
                os.rename(os.path.join(wandb.run.dir, checkpoint_file),
                          os.path.join(wandb.run.dir, "loaded_checkpoint.pt"))
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

        logger.info(f"Loading base model: {base_model_name}")
        model_kwargs = dict(self.config.items("BASE_MODEL"))

        if base_model_name == 'xlm_r':
            model_cls = XLMR
        else:
            logger.exception(f"Invalid base model type: {base_model_name}")
            raise Exception(f"Invalid base model type: {base_model_name}")

        model = model_cls.from_kwargs(**model_kwargs)

        return model

    def load_learner(self, learner_method: str) -> Union[MAML, BaselineLearner]:
        """ 
        Helper function for reading in (meta) learning procedure 
        
        Args:
            * learner_method (str): name of learner to load
            
        Returns: 
            * learner (either MAML or BaselineLearner): learner to be used for meta learning
        """

        logger.info(f"Using learner: {learner_method}")

        learner_kwargs = dict(self.config.items("LEARNER"))
        del learner_kwargs['method']

        if self.mode == "train":
            if self.use_smlmt_labels: 
                learner_kwargs['lm_head_n'] = self.config.getint("LANGUAGE_TASK", "n")
            else:
                # size of the tokenizer vocab; NOTE that we currently only support XLM-R
                learner_kwargs['lm_head_n'] = XLMR_VOCAB_SIZE

        if learner_method == 'maml':
            learner_cls = MAML
        elif learner_method == 'baseline': 
            learner_cls = BaselineLearner
        else:
            logger.exception(f"Invalid learner method: {learner_method}")
            raise Exception(f"Invalid learner method: {learner_method}")

        learner = learner_cls(
            base_model=self.base_model,
            base_device=self.base_device,
            seed=self.config.getint("EXPERIMENT", "seed"),
            **learner_kwargs
        )

        if self.checkpoint is not None:
            learner.load_state_dict(self.checkpoint['learner_state_dict'], strict=False)
        else:
            logger.info("No checkpoint used - learning from scratch")

        return learner


    def setup_meta_optimizer(self) -> Optimizer:
        """
        Helper function for setting up meta optimizer and optionally an associated learning 
        rate scheduler.

        Returns: 
            * meta_optimizer (Optimizer): meta optimizer to be used for meta learning
        """
        meta_optimizer = AdamW(self.learner.outerloop_optimizer_param_groups(), lr=self.meta_lr)
        meta_optimizer.zero_grad()

        if self.checkpoint is not None:
            meta_optimizer.load_state_dict(self.checkpoint['optimizer_state_dict'])
        return meta_optimizer


    def setup_meta_lr_scheduler(self, meta_lr_scheduler_method) -> Union[LambdaLR, None]: 
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
                    num_warmup_steps=0.1*self.max_task_batch_steps,
                    num_training_steps=self.max_task_batch_steps
                )
            else: 
                raise Exception(
                    f"Invalid meta scheduler method: {self.meta_lr_scheduler_method}"
                )

            if self.checkpoint is not None:
                meta_lr_scheduler.load_state_dict(self.checkpoint['scheduler_state_dict'])
        else: 
            meta_lr_scheduler = None

        return meta_lr_scheduler

    ### --- Saving and shutdown helpers --- ### 

    def save_checkpoint(self, checkpoint_file: str) -> None:
        """
        Helper function for saving a checkpoint of the learner and optimizer state dicts.

        Args:
            * checkpoint_file (str): name of checkpoint file to save
        """
        if not self.use_wandb:
            logger.warning("Cannot save model checkpoint - requires use_wandb to be True")
            return
        
        checkpoint = {
            'learner_state_dict': self.learner.state_dict(),
            'optimizer_state_dict': self.meta_optimizer.state_dict(),
        }

        if self.meta_lr_scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.meta_lr_scheduler.state_dict()
        
        torch.save(checkpoint, os.path.join(wandb.run.dir, checkpoint_file))
        wandb.save(checkpoint_file, policy="now")

    def shutdown_processes(self) -> None:
        """Helper function for shutting down any spawned processes """

        self.meta_dataset.shutdown()

    def timeout_handler(self, signum, frame) -> None:
        """
        Gracefully handles early termination signals. Catches termination signals sent from  
        slurm just before the program is about to terminate and saved out a model checkpoint, as
        well as shutting down any spawned workers.

        Args: 
            * signum (int): signal number
            * frame (frame): stack frame
        """

        logger.info("Timeout (SIGINT) termination signal received")
        logger.info("Attempting to save latest checkpoint of model")

        # Will try to save checkpoint to resume training from (even if self.checkpoints is False)
        self.save_checkpoint("latest-checkpoint.pt")

        # writing out the current task batch number to the run_file 
        if not os.path.exists('tmp'):
            os.mkdir('tmp')
        with open(f"tmp/{wandb.run.id}.runfile", "w+") as f:
            f.write(str(max(self.num_task_batches-1, 0)))

        self.shutdown_processes()

        # exit code 124 triggers re-run
        exit(124)


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
            for param in param_group['params']:
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
            logger.info("Running pipeline in inference-only mode")

            if not hasattr(self, "evaluator"):
                logger.error("Need to specify an evaluator to run in inference-only mode")
                return 
            
            self.evaluator.run(self.learner)
            logger.info("Finished running inference-only model")
            return 

        ### -------------------- Training Mode -------------------- 

        logger.info("Running pipeline in training mode")

        ### Setting up tracking variables and w&b metrics  

        # counter tracks loss over an entire batch of tasks  
        task_batch_loss = 0 

        # metric for logging training data
        if self.use_wandb:
            wandb.define_metric("train.loss", step_metric="num_task_batches", summary='min')

            if self.learner_method != "baseline":
                # any meta-learning approach will want to track the learned learning rates
                wandb.define_metric("classifier_lr", step_metric="num_task_batches")
                
                # for inner layers we need to track lr per layer
                num_layers = len(self.learner.inner_layers_lr)
                for layer_idx in range(num_layers):
                    wandb.define_metric(
                        f"inner_layer_{layer_idx}_lr",
                        step_metric="num_task_batches"
                    )

        if self.config.getboolean("PIPELINE", "run_initial_eval", fallback=True) and \
            self.num_task_batches == 0:
            # num_task_batches would only ever not be 0 if we're resuming training because of 
            # previous timeout failure, in that case don't run initial eval
            logger.info("Initial evaluation before model training")
            if not hasattr(self, "evaluator"):
                logger.warning("Evaluation missing in config - skipping evaluator run")
            else: 
                self.evaluator.run(self.learner, num_task_batches=0)

        ### Model training loop

        logger.info("Starting model training")
        for task_batch_idx, task_batch in enumerate(self.meta_dataloader):
            logger.debug(f"\t (Task idx {task_batch_idx}) Language: {task_batch[0]}")

            ## Basic training with just a single GPU 
            task_name, support_batch_list, query_batch = task_batch

            task_loss = self.learner.run_train_loop(support_batch_list, query_batch)            
            task_loss = task_loss/self.num_tasks_per_iteration # normalizing loss 
            task_batch_loss += task_loss
       
            if ((task_batch_idx + 1) % self.num_tasks_per_iteration == 0):        
                ##### NOTE: Just finished a batch of tasks -- taking a global (meta) update step

                self.num_task_batches += 1

                # single GPU: taking optimizer step
                self.meta_optimizer_step(grad_norm_constant=self.num_tasks_per_iteration)

                ### Logging out training results
                logger.info(f"No. batches of tasks processed: {self.num_task_batches}")
                logger.info(f"\t(Meta) training loss: {task_batch_loss}")
                if self.use_wandb:
                    
                    if self.learner_method != "baseline": 
                        # wandb logging info for any meta-learner
                        wandb.log(
                            {"classifier_lr": self.learner.classifier_lr.item()},
                            commit=False
                        )

                        for layer_num, layer in self.learner.inner_layers_lr.items():
                                wandb.log(
                                    {f"inner_layer_{layer_num}_lr": layer.item()}, 
                                    commit=False
                                )

                    wandb.log(
                        {
                            "train.loss": task_batch_loss,
                            "num_task_batches": self.num_task_batches
                        },
                    )

                task_batch_loss = 0 

                ### possibly run evaluation of the model
                if (self.eval_every_n_iteration 
                    and self.num_task_batches % self.eval_every_n_iteration == 0
                ):
                    if not hasattr(self, "evaluator"):
                        logger.warning("Evaluation missing in config - skipping evaluator run")
                    else: 
                        new_best = self.evaluator.run(
                            self.learner,
                            num_task_batches=self.num_task_batches
                        )

                        if self.save_checkpoints:
                            self.save_checkpoint('latest-checkpoint.pt')
                            if new_best:
                                self.save_checkpoint(f"checkpoint-{self.num_task_batches}.pt")

                if (self.num_task_batches % self.max_task_batch_steps == 0):
                    # NOTE: stop training if we've done max_task_batch_steps global update steps
                    break

        ### Model done training - final clean up before exiting 

        logger.info("Finished training model")
        if self.save_checkpoints:
            self.save_checkpoint("final.pt")
        
        self.shutdown_processes()

