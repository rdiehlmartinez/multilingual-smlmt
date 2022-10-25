__author__ = 'Richard Diehl Martinez'
""" General purpose dataloader for NLU datasets """

import torch
from torch.utils.data import DataLoader

from ..utils.data import base_collate_fn

class NLUCollator(object):
    def __call__(self, batch): 
        return base_collate_fn(batch, True)

class NLUDataLoader(DataLoader):
    """
    Minimal wrapper around DataLoader to override the default collate_fn to be 
    nlu_collate.
    """
    def __init__(self, dataset, **kwargs):
        nlu_collator = NLUCollator()
        super().__init__(dataset, collate_fn=nlu_collator, **kwargs)