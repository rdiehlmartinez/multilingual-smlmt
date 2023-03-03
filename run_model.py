__author__ = "Richard Diehl Martinez"
""" Entry point for launching the model training and evaluation pipeline """

import argparse
import os
import random
import signal

import wandb

from src.pipeline import Pipeline
from src.utils import setup

parser = argparse.ArgumentParser(
    description="Parses config files passed in via CLI"
)
parser.add_argument(
    "--path", metavar="path", type=str, help="path to the config yaml file"
)
parser.add_argument(
    "--run_id", type=str, help="Unique identifier for the run of the model"
)
parser.add_argument(
    "--offline_mode",
    action="store_true",
    help="Run the model without pushing to wandb",
)
parser.add_argument(
    "--sweep_agent",
    action="store_true",
    help="The model is being run by a sweep agent",
)
args, _ = parser.parse_known_args()

if __name__ == "__main__":

    # NOTE: Path can be None, but if it is, run_id must be specified (either as an argument passed
    # in or as an environment variable)
    # NOTE: The run_id must be associated with a run that has been stored in wandb, and which 
    # we can use to pull in the associated config file.
    if (
        args.path is None
        and args.run_id is None
        and "RUN_ID" not in os.environ
    ):
        raise ValueError("If path is not specified, run_id must be specified")

    # Setting up an id for the specific run
    if args.run_id is None and "RUN_ID" not in os.environ:
        # NOTE: This is a new run, so we assign an INTERNAL random run_id to the run
        # NOTE: This may or may not be the same as the run_id that wandb assigns to the run
        run_id = str(random.randint(1, 1e9))
    else:
        # run_id is either passed in as an argument or is an environment variable
        if args.run_id is not None:
            run_id = args.run_id
        else:
            run_id = os.environ["RUN_ID"]

    if args.sweep_agent: 
        # NOTE: Sweep agents assign their own wandb run id 
        wandb_run_id = None
    else: 
        wandb_run_id = run_id

    resume_training = False

    # Possibly reading in a runfile (only exists if we are resuming training)
    run_file_path = f"run_files/{run_id}.runfile"
    if os.path.exists(run_file_path):
        # we must be resuming a run - reading in relevant information
        with open(run_file_path, "r") as f:
            resume_num_task_batches = int(f.readline().strip())
            # the second line in the runfile is the wandb run id (might be different from 
            # the internal run id)
            wandb_run_id = f.readline().strip()
        resume_training = True
    else:
        resume_num_task_batches = 0

    # Setting up logging, config read in and seed setting
    setup(args.path, run_id, wandb_run_id, resume_num_task_batches, resume_training, args.offline_mode, args.sweep_agent)

    # Initializing the modeling pipeline with configuration and options
    pipeline = Pipeline(run_id, resume_num_task_batches)

    # setting up timeout handler - called if the program receives a SIGINT either from the user
    # or from SLURM if it is about to timeout
    signal.signal(signal.SIGINT, pipeline.timeout_handler)

    # launching training or eval script
    pipeline()

    if os.path.exists(run_file_path):
        os.remove(run_file_path)
