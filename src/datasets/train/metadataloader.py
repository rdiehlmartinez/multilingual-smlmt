__author__ = 'Richard Diehl Martinez '
""" Utilities for dataloading a MetaDataset """

import torch
from torch.utils.data import DataLoader

from .metadataset import MASK_TOKEN_ID
from ...utils.data import base_collate_fn

# typing imports
from typing import List, Tuple, Any, Dict
from .metadataset import MetaDataset

RawDataBatch = Dict[int, List[List[int]]]
ProcessedDataBatch = Dict[str, torch.Tensor]
SupportQueryBatch =  Tuple[List[RawDataBatch], RawDataBatch]

class MetaCollator(object):

    def __init__(self, return_standard_labels: bool) -> None:
        """ 
        Helper class to define a collate function. In order to supply additional arguments to 
        the function, we wrap the function in this class and pass in params via instance attributes.

        Args:
            * return_standard_labels (bool): Whether to collate the batch to use the token ids of 
                the masked tokens, or whether to transform the labels to be in range [0, n]. 
                Typically we don't want to return the standard labels - the main use case is to 
                prepare batches of data for models that don't use meta learning methods.
        """
        self.return_standard_labels = return_standard_labels

    def __call__(
        self,
        lm_task_samples: List[Tuple[str, SupportQueryBatch]]
    ) -> Tuple[str, List[ProcessedDataBatch], ProcessedDataBatch]:
        """ 
        Transform a batch of task data into input and label tensors that can be fed 
        into a model. 
        
        Args: 
            * lm_task_samples: Single-element list of a Tuple containing the following: 
                * task_name (str): Task name (e.g. the language) of the current batch of data 
                * (support_samples_list, query_samples), where this tuple contains the following: 
                    * support_samples_list [{token id: [K samples where token id occurs]}]: List of 
                        a mapping of N token ids to K samples per token id occurs
                    * query_samples {token id: [Q samples where token id occurs]}: Mapping of N 
                        tokenids to Q samples per token id occurs

        Returns: 
            * task_name (str): Task name (i.e. the language) of batch
            * support_batch_list: A list of dictionaries, each containing the following information
                for the support set:
                * input_ids (torch.tensor): Input tensors of shape (N*K, max_seq_len)
                * input_target_idx (torch.tensor): Tensor indicating for each sample at what index 
                    we apply the final classification layer 
                * label_ids (torch.tensor): Tensor of labels corresponding to masked out subword id
                * attention_mask (torch.tensor): Tensor indicating which tokens in input_ids
                    are not pad tokens
            * query_batch: Same as support_batch, but for the data of the query set 

        """
        
        # Unpack the task name and the support and query samples
        task_name, (support_samples_list, query_samples) = lm_task_samples[0]

        support_batch_list = [
            base_collate_fn(support_samples, self.return_standard_labels)
            for support_samples in support_samples_list
        ]    
        query_batch = base_collate_fn(query_samples, self.return_standard_labels)

        return (task_name, support_batch_list, query_batch)
    

class MetaDataLoader(DataLoader):
    """
    Stripped down basic dataloader meant to be used with MetaDataset,
    note that MetaDataset does most of the heavy-lifting with processing 
    the data. 
    
    Copied from: 
    https://github.com/tristandeleu/pytorch-meta/blob/master/torchmeta/utils/data/dataloader.py#L32
    """
    def __init__(
        self,
        dataset: MetaDataset,
        return_standard_labels: bool = False,
        batch_size: int = 1,
        shuffle: bool = False,
        sampler: Any = None,
        batch_sampler: Any = None,
        num_workers: int = 0,
        pin_memory: bool = False,
        drop_last: bool = False,
        timeout: int = 0,
        worker_init_fn: Any = None,
    ) -> None:
                
        """ Resetting basic defaults  """

        meta_collator = MetaCollator(return_standard_labels=return_standard_labels)

        super().__init__(
            dataset, 
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            batch_sampler=batch_sampler,
            num_workers=num_workers, 
            collate_fn=meta_collator,
            pin_memory=pin_memory,
            drop_last=drop_last,
            timeout=timeout,
            worker_init_fn=worker_init_fn
        )