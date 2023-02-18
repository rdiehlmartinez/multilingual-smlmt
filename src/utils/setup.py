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
    resume_num_task_batches: int,
    offline_mode: bool,
) -> None:
    """
    Reads in config file into wandb init, sets up logger and sets a seed to ensure reproducibility.

    NOTE: The arguments 'run_id' and 'resume_num_task_batches' should not be manually set,
    rather they are passed in automatically by the program if it encounters a time expiration
    error and thus spawns a new job to continue running the program.

    Args:
        * config_fp: path to config file
        * run_id: id of the run
        * resume_num_task_batches: number of task batches to resume training from
        * offline_mode: whether or not to run in offline mode
    """
    resume_training = resume_num_task_batches > 0

    # setting the start method to spawn to avoid issues with CUDA and WandB
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        if mp.get_start_method() != "spawn":
            raise Exception("Could not set start method to spawn")

    setup_wandb(config_fp, run_id, resume_training)

    # NOTE: wandb flattens the config file during sweeps
    # we need to unflatten it (i.e. back into a nested dictionary)
    for config_key, config_val in wandb.config.items():
        if not isinstance(config_val, dict):
            primary_key, sub_key = config_key.split(".")
            wandb.config[primary_key][sub_key] = config_val

    # If config_fp is None we must be resuming a run. In this case, we will have already set the
    # config_file parameter.
    if config_fp is None:
        # NOTE: if the config_file has not been set in the config file, then something's gone wrong
        assert (
            "config_file" in wandb.config["EXPERIMENT"]
        ), "config_file not found in config"
        config_fp = wandb.config["EXPERIMENT"]["config_file"]
    else:
        wandb.config["EXPERIMENT"]["config_file"] = config_fp

    setup_logger(config_fp, run_id)

    # we are resuming training if resume_num_task_batches is greater than 0

    if resume_training:
        logging.info(f"Resuming run with id: {run_id}")
    else:
        logging.info(f"Initializing run with id: {run_id}")

    if offline_mode:
        logging.info("Running in offline mode")
        os.environ["WANDB_MODE"] = "offline"

    seed = int(wandb.config["EXPERIMENT"]["seed"])
    # shifting over the random seed by resume_num_task_batches steps in order for the meta
    # dataset to not yield the same sentences as already seen by the model
    # also added benefit that if the same job is run again we are very likely (but not guaranteed)
    # to achieve the same result
    seed += resume_num_task_batches

    wandb.config["EXPERIMENT"]["seed"] = seed
    set_seed(seed)
