__author__ = "Richard Diehl Martinez"
""" Entry point for launching the model training and evaluation pipeline """

import argparse
import os
import random
import signal

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
args, _ = parser.parse_known_args()

if __name__ == "__main__":
    # Path can be None, but if it is, run_id must be specified and wandb.config must include
    # the path to the config file
    # NOTE: this happens when we are resuming training from a sweep job
    if (
        args.path is None
        and args.run_id is None
        and "RUN_ID" not in os.environ
    ):
        raise ValueError("If path is not specified, run_id must be specified")

    # Setting up an id for the specific run
    if args.run_id is None and "RUN_ID" not in os.environ:
        # NOTE: Calling on this module directly (i.e. not from SLURM)
        run_id = str(random.randint(1, 1e9))
    else:
        # NOTE: Either calling on this module from SLURM or resuming training from a failed job
        if args.run_id is not None:
            run_id = args.run_id
        else:
            # run_id can also be exported as an environment variable
            run_id = os.environ["RUN_ID"]

    # Possibly reading in a runfile (only exists if we are resuming training)
    run_file_path = f"tmp/{run_id}.runfile"
    if os.path.exists(run_file_path):
        # we must be resuming a run - reading in relevant information
        with open(run_file_path, "r") as f:
            resume_num_task_batches = int(f.readline())
    else:
        resume_num_task_batches = 0

    # Setting up logging, config read in and seed setting
    setup(args.path, run_id, resume_num_task_batches, args.offline_mode)

    # Initializing the modeling pipeline with configuration and options
    pipeline = Pipeline(resume_num_task_batches)

    # setting up timeout handler - called if the program receives a SIGINT either from the user
    # or from SLURM if it is about to timeout
    signal.signal(signal.SIGINT, pipeline.timeout_handler)

    # launching training or eval script
    pipeline()

    if os.path.exists(run_file_path):
        os.remove(run_file_path)
