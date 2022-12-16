import torch

from .data import move_to_device
from .setup import setup, set_seed

# global device (used unless explicitly overriden)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_gpus = torch.cuda.device_count()