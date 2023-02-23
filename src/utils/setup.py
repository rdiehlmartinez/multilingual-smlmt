__author__ = "Richard Diehl Martinez"
""" Utilities for setting up logging and reading configuration files"""

import logging
import multiprocessing as mp
import os
import random

import numpy as np
import torch

import wandb


def set_seed(seed: int) -> None:
    """Sets seed for reproducibility"""
    if seed < 0:
        logging.warning("Skipping seed setting for reproducibility")
        logging.warning(
            "If you would like to set a seed, set seed to a positive value in config"
        )
        return

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available() > 0:
        torch.cuda.manual_seed_all(seed)


def setup_logger(config_fp: str, run_id: int) -> None:
    """Set up logging functionality"""
    # Removing handlers that might be associated with environment; and logs
    # out to both stderr and a log file
    experiment_directory = os.path.dirname(
        os.path.join(os.getcwd(), config_fp)
    )

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    # storing the logs in the config file directory
    log_directory = os.path.join(experiment_directory, "logs")
    if not os.path.exists(log_directory):
        os.mkdir(log_directory)

    log_file_name = os.path.join(log_directory, f"experiment_{run_id}.log")
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%m/%d/%Y %I:%M:%S %p",
        level=logging.DEBUG,
        handlers=[logging.FileHandler(log_file_name), logging.StreamHandler()],
    )
    logging.info(f"Initializing experiment: {experiment_directory}")


def setup_wandb(config_fp: str, run_id: str, resume_training: bool) -> None:
    """
    Initialize experiment with weights & biases
    """
    wandb.init(
        project="Multilingual-SMLMT",
        entity="problyglot",
        config=config_fp,
        id=run_id,
        resume="must" if resume_training else None,
    )


def setup(
    config_fp: str,
    run_id: str,
    wandb_run_id: str,
    resume_num_task_batches: int,
    offline_mode: bool,
    sweep_agent: bool,
) -> None:
    """
    Reads in config file into wandb init, sets up logger and sets a seed to ensure reproducibility.

    NOTE: The arguments 'run_id' and 'resume_num_task_batches' should not be manually set,
    rather they are passed in automatically by the program if it encounters a time expiration
    error and thus spawns a new job to continue running the program.

    Args:
        * config_fp: path to config file (possibly NONE)
        * run_id: id of the run
        * wandb_run_id: id of the wandb run (possibly NONE)
        * resume_num_task_batches: number of task batches to resume training from
        * offline_mode: whether or not to run in offline mode
        * sweep_agent: whether or not the program is being run by a sweep agent
    """
    resume_training = resume_num_task_batches > 0

    # setting the start method to spawn to avoid issues with CUDA and WandB
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        if mp.get_start_method() != "spawn":
            raise Exception("Could not set start method to spawn")

    if offline_mode:
        os.environ["WANDB_MODE"] = "offline"

    setup_wandb(config_fp, wandb_run_id, resume_training)

    if sweep_agent:
        # If this is being run by a sweep agent, setuping wandb in previous call will
        # assign the wandb run_id
        wandb_run_id = wandb.run.id

    # NOTE: wandb possibly flattens the config file (e.g during sweeps)
    # To be safe, we need to unflatten it (i.e. back into a nested dictionary)
    # After unflatting, we ideally would remove the config items that are set by the sweep
    # agent, but wandb.config does not allow easily removal (i.e. del doesn't work)

    for config_key, config_val in wandb.config.items():
        if not isinstance(config_val, dict):
            primary_key, sub_key = config_key.split(".")
            wandb.config[primary_key][sub_key] = config_val

    # If config_fp is None we must be resuming a run. In this case, we can use the config
    # to determine the original config file path
    if config_fp is None:
        config_fp = wandb.config["EXPERIMENT"]["config_fp"]

    setup_logger(config_fp, run_id)

    # we are resuming training if resume_num_task_batches is greater than 0

    if resume_training:
        logging.info(
            f"Resuming run with run id: {run_id} - wandb run id: {wandb_run_id}"
        )
    else:
        logging.info(
            f"Initializing run with id: {run_id} - wandb run id: {wandb_run_id}"
        )

    seed = int(wandb.config["EXPERIMENT"]["seed"])
    # shifting over the random seed by resume_num_task_batches steps in order for the meta
    # dataset to not yield the same sentences as already seen by the model
    # also added benefit that if the same job is run again we are very likely (but not guaranteed)
    # to achieve the same result
    seed += resume_num_task_batches

    wandb.config["EXPERIMENT"]["seed"] = seed
    set_seed(seed)
