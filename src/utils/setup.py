__author__ = 'Richard Diehl Martinez'
""" Utilities for setting up logging and reading configuration files"""

import os
import logging 
import torch 
import numpy as np
import random
import wandb
import json
import multiprocessing as mp

from configparser import ConfigParser

def set_seed(seed: int) -> None:
    ''' Sets seed for reproducibility '''
    if seed < 0: 
        logging.info("Skipping seed setting for reproducibility")
        logging.info("If you would like to set a seed, set seed to a positive value in config")
        return

    logging.info(f"Setting seed: {seed}")
        
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available() > 0:
        torch.cuda.manual_seed_all(seed)

def setup_config(config_file_path: str) -> ConfigParser:
    ''' Reads in a config file using ConfigParser '''
    config = ConfigParser()
    config.read(config_file_path)
    return config

def setup_logger(config_file_path: str) -> None:
    ''' Sets up logging functionality '''
    # Removing handlers that might be associated with environment; and logs
    # out to both stderr and a log file
    experiment_directory = os.path.dirname(os.path.join(os.getcwd(), config_file_path)) 

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    log_file_name = os.path.join(experiment_directory, "experiment.log")
    logging.basicConfig(
                        format='%(asctime)s [%(levelname)s] %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p',
                        level=logging.DEBUG,
                        handlers=[
                            logging.FileHandler(log_file_name),
                            logging.StreamHandler()
                        ]
    )
    logging.info(f"Initializing experiment: {experiment_directory}")

def setup_wandb(config: ConfigParser, run_id: str, resume_training: bool) -> None:
    """
    Sets up logging and model experimentation using weights & biases 
    """
    if config.getboolean('EXPERIMENT', 'use_wandb', fallback=True):
        dict_config = json.loads(json.dumps(config._sections))
        wandb.init(
            project='Multilingual-SMLMT',
            entity="problyglot",
            config=dict_config,
            id=run_id,
            resume="must" if resume_training else None,
        )

def setup(config_file_path: str, run_id: str, resume_num_task_batches: int) -> ConfigParser:
    """
    Reads in config file, sets up logger and sets a seed to ensure reproducibility.

    NOTE: The optional keyword arguments (run_id and resume_num_task_batches) should never
    be manually set, rather they are passed in automatically by the program if it encounters a 
    time expiration error and thus spawns a new job to continue running the program.
    """

    # setting the start method to spawn to avoid issues with CUDA and WandB
    mp.set_start_method("spawn")

    config = setup_config(config_file_path)

    # we are resuming training if resume_num_task_batches is greater than 0
    resume_training = resume_num_task_batches > 0 
    if resume_training:
       logging.info(f"Resuming run with id: {run_id}")
    else: 
        logging.info(f"Starting run with id: {run_id}")

    setup_logger(config_file_path)
    setup_wandb(config, run_id, resume_training)

    seed=config.getint("EXPERIMENT", "seed", fallback=-1)

    # shifting over the random seed by resume_num_task_batches steps in order for the meta
    # dataset to not yield the same sentences as already seen by the model 
    # also added benefit that if the same job is run again we are likely to achieve the same  
    # result
    seed += resume_num_task_batches
    
    config["EXPERIMENT"]["seed"] = str(seed)
    set_seed(seed)

    return config