__author__ = "Richard Diehl Martinez"
""" General purpose dataloader for NLU datasets """

import torch
from torch.utils.data import DataLoader

from ...utils.data import base_collate_fn

# import statements for type hints
from typing import Dict, Union, Tuple, List, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from .nludataset import NLUDataset


class NLUCollator(object):
    def __call__(
        self,
        batch: Union[List[Tuple[int, List[int]]], Dict[int, List[List[int]]]],
    ) -> Dict[str, torch.Tensor]:
        """
        Collate function for NLU datasets. Takes a batch of samples and collates them into a
        dictionary of tensors.

        Args:
            * batch (List[Tuple[int, List[int]]] or Dict[int, List[List[int]]]):
                * If List[Tuple[int, List[int]]]: List of tuples containing the following:
                    * label (int): Label of the sample
                    * input_ids (List[int]): List of input ids
                * If Dict[int, List[List[int]]]: Dictionary containing the following:
                    * label (int): Label of the sample
                    * list of input_ids (List[List[int]]): A list of input ids for each sample
        """
        return base_collate_fn(batch, use_smlmt_labels=False)


class NLUDataLoader(DataLoader):
    """
    Minimal wrapper around DataLoader to override the default collate_fn to be
    nlu_collate.
    """

    def __init__(self, dataset: "NLUDataset", **kwargs) -> None:
        nlu_collator = NLUCollator()
        super().__init__(dataset, collate_fn=nlu_collator, **kwargs)
