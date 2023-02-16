import torch

import wandb

from .data import move_to_device
from .setup import set_seed, setup

# setting global device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_gpus = torch.cuda.device_count()
