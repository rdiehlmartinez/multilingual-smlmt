__author__ = 'Richard Diehl Martinez'
""" Wrapper class for training and evaluating a model using a given meta learning technique """

import typing
import torch
import logging 
import wandb
import time 
import os 
import subprocess

import torch.multiprocessing as mp

from .models import XLMR
from .metalearners import Platipus, MAML, BaselineLearner
from .evaluation import Evaluator
from .utils import device as DEFAULT_DEVICE, num_gpus
from .datasets import MetaDataset, MetaDataLoader

logger = logging.getLogger(__name__)

class Problyglot(object):
    """
    Orchestrates model loading, training and evaluation using a specific 
    type of (meta-)learner.
    """

    def __init__(self, config, resume_run_config_path=None, resume_run_id=None,
                 resume_num_task_batches=None) -> None:
        """ Initialize base model and meta learning method based on a config 
        
        NOTE: The optional keyword arguments (resume_run_config_path, resume_run_id, 
        resume_num_task_batches) should ever be manually set, rather they are passed in
        automatically by the program if it encounters a time expiration error and thus spawns a
        new job to continue running the program.
        """

        # keeps track of the config path in case the program exits and we need to start over
        self.resume_run_config_path = resume_run_config_path

        # config params need to be accessed by several methods
        self.config = config
        
        # whether to log out information to w&b
        self.use_wandb = config.getboolean('EXPERIMENT', 'use_wandb', fallback=True)

        # setting up meta dataset for training if provided in config
        if 'META_DATASET' in config:
            self.meta_dataset = MetaDataset(config)

            self.return_standard_labels = config.getboolean("META_DATASET",
                                                            "return_standard_labels",
                                                            fallback=False)
            self.meta_dataloader = MetaDataLoader(self.meta_dataset, return_standard_labels=\
                                                                     self.return_standard_labels)

        # Setting device 
        self.base_device = config.get("PROBLYGLOT", "device", fallback=DEFAULT_DEVICE)
        self.use_multiple_gpus = self.base_device == torch.device("cuda") and num_gpus > 1
        logger.info(f"Running problyglot on device: {self.base_device}")

        # setting base model 
        self.base_model_name = config.get("BASE_MODEL", "name")
        self.base_model = self.load_model(self.base_model_name)

        # setting learner 
        self.learner_method = self.config.get("LEARNER", "method")
        self.learner = self.load_learner(self.learner_method, resume_run_id)

        self.num_task_batches = resume_num_task_batches if resume_num_task_batches else 0

        if self.use_wandb:
            # setting up metrics for logging to wandb
            # counter tracks number of batches of tasks seen by metalearner
            wandb.define_metric("num_task_batches")
        
        # setting evaluator 
        if 'EVALUATION' in config:
            self.evaluator = Evaluator(config)


    def load_model(self, base_model_name):
        """
        Helper function for reading in base model, should be intialized with the 
        from_kwargs() class method 
        """

        logger.info(f"Loading base model: {base_model_name}")
        model_kwargs = dict(self.config.items("BASE_MODEL"))

        if base_model_name == 'xlm_r':
            model_cls = XLMR
        else:
            logger.exception(f"Invalid base model type: {base_model_name}")
            raise Exception(f"Invalid base model type: {base_model_name}")

        model = model_cls.from_kwargs(**model_kwargs)

        logger.debug("Base Model Architecture: ")
        logger.debug(model)

        return model

    def load_learner(self, learner_method, resume_run_id=None):
        """ Helper function for reading in (meta) learning procedure """

        logger.info(f"Using learner: {learner_method}")

        learner_kwargs = dict(self.config.items("LEARNER"))

        if hasattr(self, "return_standard_labels") and self.return_standard_labels: 
            # The final classification layer of the learner is over the entire vocab,
            # thus cannot infer the size of the classication layer from the LANGUAGE_TASK config
            assert("lm_head_n" in learner_kwargs),\
                "Must defined lm_head_n in LEARNER config (cannot be inferred)"
        else: 
            # NOTE: If not defined, size of lm head classification task is taken from LANGUAGE_TASK
            if "lm_head_n" not in learner_kwargs:
                logger.info("Attempting to infer lm_head_n from LANGUAGE_TASK config")
                learner_kwargs['lm_head_n'] = self.config.getint("LANGUAGE_TASK", "n")

        if learner_method == 'platipus':
            learner_cls = Platipus
        elif learner_method == 'maml':
            learner_cls = MAML
        elif learner_method == 'baseline': 
            learner_cls = BaselineLearner
        else:
            logger.exception(f"Invalid learner method: {learner_method}")
            raise Exception(f"Invalid learner method: {learner_method}")

        learner = learner_cls(self.base_model, base_device=self.base_device,
                                               seed=self.config.getint("EXPERIMENT", "seed"),
                                               **learner_kwargs)

        # NOTE: possibly load in learner checkpoint
        # if resume_run_id we start from the latest checkpoint instead of whatever checkpoint 
        # might have been specified in the config
        if resume_run_id is not None:
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
                learner.load_state_dict(checkpoint['learner_state_dict'], strict=False)
                learner.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                os.rename(os.path.join(wandb.run.dir, checkpoint_file),
                          os.path.join(wandb.run.dir, "loaded_checkpoint.pt"))
        else:
            logger.info("No checkpoint used - learning from scratch")

        return learner


    def shutdown_processes(self):
        """Helper function for shutting down any spawned processes """

        self.meta_dataset.shutdown()

        # Shut down workers if using multiple GPUs
        if hasattr(self, "gpu_workers") and self.use_multiple_gpus: 
            logger.info("Shutting down GPU workers used for model training")
            for p in self.gpu_workers:
                p.terminate()
                time.sleep(1)
                p.join()

    def timeout_handler(self, signum, frame):
        """
        Gracefully handles early termination signals. Catches termination signals sent from  
        slurm just before the program is about to terminate and saved out a model checkpoint, as
        well as shutting down any spawned workers.
        """

        logger.info("Timeout (SIGINT) termination signal received")
        logger.info("Attempting to save final checkpoint of model")

        if self.config.getboolean('PROBLYGLOT', 'save_latest_checkpoint', fallback=True):
            if not self.use_wandb:
                logger.error("Cannot save model checkpoint because use_wandb set to False")
            else:
                logger.info(f"Saving model checkpoint")
                checkpoint = {
                    'learner_state_dict': self.learner.state_dict(),
                    'optimizer_state_dict': self.learner.optimizer.state_dict(),
                }
                torch.save(checkpoint, os.path.join(wandb.run.dir, "latest-checkpoint.pt"))
                # forcing move to save out latest checkpoint before spawning new job
                wandb.save('latest-checkpoint.pt', policy="now")

                # spawning new job 
                subprocess.run(["sbatch", "run_model.wilkes3", self.resume_run_config_path,
                                "--resume_run_id", str(wandb.run.id), "--resume_num_task_batches",
                                str(max(self.num_task_batches-1, 0))], cwd='scripts')

        else:
            logger.error("Failed to save checkpoint - save_latest_checkpoint set to False")

        self.shutdown_processes()

        exit(1)

    def __call__(self) -> None: 
        """ 
        Train or evaluate the self.base_model via the self.learner training procedure 
        on data stored in self.meta_dataloader
        """

        ### --------- Evaluation Mode (will return early) ---------- 
        if not hasattr(self, "meta_dataset"):
            logger.info("Running problygot in evaluation mode")

            if not hasattr(self, "evaluator"):
                logger.warning("No evaluator specified - running problygot is a no-op")
                return 
            
            logger.info("Finished running evaluation model")
            self.evaluator.run(self.learner)
            return 

        ### --------- Training Mode ---------- 

        logger.info("Running problyglot in training mode")

        ### reading in training configs

        num_tasks_per_iteration = self.config.getint("PROBLYGLOT", "num_tasks_per_iteration",
                                                     fallback=1)
        eval_every_n_iteration = self.config.getint("PROBLYGLOT", "eval_every_n_iteration",
                                                    fallback=0)
        max_task_batch_steps = self.config.getint("PROBLYGLOT", "max_task_batch_steps",
                                                  fallback=1)

        ### If using n GPUs we launch n processes that run the run_inner_loop_mp function 

        # If using multiple GPUs
        if self.use_multiple_gpus:
            
            if num_tasks_per_iteration % num_gpus != 0:
                error_msg = "Num tasks per iteration has to be dividable by num_pus!"
                logger.exception(error_msg)
                raise Exception(error_msg)

            logger.info(f"Running data parallel training with {num_gpus} workers")
            spawn_context = mp.get_context('spawn')

            data_queue = spawn_context.Queue()
            loss_queue = spawn_context.Queue()

            step_optimizer = spawn_context.Event()

            self.gpu_workers = []
            for rank in range(num_gpus):
                p = spawn_context.Process(target=self.learner.run_inner_loop_mp,
                                          args=(rank, num_gpus, data_queue, loss_queue,
                                                step_optimizer, num_tasks_per_iteration  
                                               )
                                          )
                p.start()
                self.gpu_workers.append(p)


        ### Setting up tracking variables and w&b metrics  

        # counter tracks loss over an entire batch of tasks  
        task_batch_loss = 0 
        if self.learner_method == "platipus":
            # platipus tracks the ce and kl parts of the loss function
            task_batch_ce_loss = 0
            task_batch_kl_loss = 0

        # metric for logging training data
        if self.use_wandb:
            wandb.define_metric("train.loss", step_metric="num_task_batches", summary='min')

            if self.learner_method == "platipus":
                # platipus tracks additional information about the parts of the loss functions,
                # and the learning procedure of the hyper-parameters
                wandb.define_metric("train.loss_ce", step_metric="num_task_batches", summary='min')
                wandb.define_metric("train.loss_kl", step_metric="num_task_batches", summary='min')
                wandb.define_metric("gamma_p", step_metric="num_task_batches")
                wandb.define_metric("gamma_q", step_metric="num_task_batches")

            if self.learner_method != "baseline":
                # any meta-learning approach will want to track the learned learning rates
                wandb.define_metric("classifier_lr", step_metric="num_task_batches")
                wandb.define_metric("inner_lr", step_metric="num_task_batches")

        if self.config.getboolean("PROBLYGLOT", "run_initial_eval", fallback=True) and \
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
        for batch_idx, batch in enumerate(self.meta_dataloader):
            logger.debug(f"\t (Task idx {batch_idx}) Language: {batch[0]}")
            if self.use_multiple_gpus:
                ## Filling up data queue for workers to process
                data_queue.put([batch], False)
            else:
                ## Basic training with just a single GPU 
                task_name, support_batch, query_batch = batch

                if self.learner_method == "platipus":
                    task_loss, (ce_loss, kl_loss) = self.learner.run_inner_loop(support_batch,
                                                                                query_batch)
                else: 
                    task_loss = self.learner.run_inner_loop(support_batch, query_batch)
            
                task_loss = task_loss/num_tasks_per_iteration # normalizing loss 
                if self.learner_method == "platipus":
                    ce_loss = ce_loss/num_tasks_per_iteration
                    kl_loss = kl_loss/num_tasks_per_iteration
            
                task_loss.backward()

                task_batch_loss += task_loss.detach().item()
                if self.learner_method == "platipus":
                    task_batch_ce_loss += ce_loss.detach().item()
                    task_batch_kl_loss += kl_loss.detach().item()

            if ((batch_idx + 1) % num_tasks_per_iteration == 0):
                #### NOTE: Just finished a batch of tasks 

                if self.use_multiple_gpus: 
                    while True:
                        # Waiting for all processes to finish computing gradients
                        time.sleep(1)
                        if loss_queue.qsize() == num_tasks_per_iteration:
                            break

                    ## Multi GPU: gathering up all of the task losses
                    for _ in range(num_tasks_per_iteration):

                        loss = loss_queue.get()[0]

                        # loss should already be normalized
                        if self.learner_method == "platipus":
                            task_loss, ce_loss, kl_loss = loss
                            task_batch_ce_loss += ce_loss
                            task_batch_kl_loss += kl_loss
                        else:
                            task_loss = loss
                        
                        task_batch_loss += task_loss
                                                
                ##### NOTE: Taking a global (meta) update step
                self.num_task_batches += 1
                if self.use_multiple_gpus: 
                    # informing/waiting for workers to all take an optimizer step 
                    step_optimizer.set()
                    while step_optimizer.is_set():
                        time.sleep(1)
                else: 
                    self.learner.optimizer_step(set_zero_grad=True)

                ### Logging out training results
                logger.info(f"No. batches of tasks processed: {self.num_task_batches}")
                logger.info(f"\t(Meta) training loss: {task_batch_loss}")
                if self.use_wandb:
                    if self.learner_method == "platipus": 
                        # wandb logging additional info for platipus
                        wandb.log({"train.loss_ce": task_batch_ce_loss,
                                   "train.loss_kl": task_batch_kl_loss,
                                   "gamma_p": self.learner.gamma_p.item(),
                                   "gamma_q": self.learner.gamma_q.item()},
                                   commit=False
                                  )
                    
                    if self.learner_method != "baseline": 
                        # wandb logging info for any meta-learner
                        wandb.log({"classifier_lr": self.learner.classifier_lr.item(), 
                                   "inner_lr": self.learner.inner_lr.item()},
                                   commit=False
                                  )

                    wandb.log({"train.loss": task_batch_loss,
                               "num_task_batches": self.num_task_batches},
                             )

                task_batch_loss = 0 
                if self.learner_method == "platipus":
                    task_batch_ce_loss = 0
                    task_batch_kl_loss = 0

                ### possibly run evaluation of the model
                if (eval_every_n_iteration and self.num_task_batches % eval_every_n_iteration == 0):
                    if not hasattr(self, "evaluator"):
                        logger.warning("Evaluation missing in config - skipping evaluator run")
                    else: 
                        self.evaluator.run(self.learner, num_task_batches=self.num_task_batches)

                if (self.num_task_batches % max_task_batch_steps == 0):
                    # NOTE: stop training if we've done max_task_batch_steps global update steps
                    break

        ### Model done training - final clean up before exiting 

        logger.info("Finished training model")
        if self.config.getboolean('PROBLYGLOT', 'save_final_model', fallback=True):
            if not self.use_wandb:
                logger.error("Cannot save model checkpoint because use_wandb set to False")
            else:
                logger.info(f"Saving trained model")
                checkpoint = {
                    'learner_state_dict': self.learner.state_dict(),
                    'optimizer_state_dict': self.learner.optimizer.state_dict(),
                }
                torch.save(checkpoint, os.path.join(wandb.run.dir, f"final.pt"))
                # NOTE: checkpoint will be uploaded to wandb on exiting program
        
        self.shutdown_processes()

