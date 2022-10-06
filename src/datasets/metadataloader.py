__author__ = 'Richard Diehl Martinez '
""" Utilities for dataloading a MetaDataset """

import torch

from torch.utils.data import DataLoader
from .metadataset import MASK_TOKEN_ID

class MetaCollator(object):

    def __init__(self, return_standard_labels: bool):
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

    def __call__(self, batch):
        """ 
        Transform a batch of task data into input and label tensors that can be fed 
        into a model. 
        
        Args: 
            * batch: Tuple containing the following: 
                * task_name (str): Task name (e.g. the language) of the current batch of data 
                * (support_set_list, query_set), where this tuple contains the following: 
                    * support_set_list [{token id: [K samples where token id occurs]}]: List of 
                        a mapping of N token ids to K samples per token id occurs
                    * query_set {token id: [Q samples where token id occurs]}: Mapping of N token
                        ids to Q samples per token id occurs

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
        
        task_name, (support_samples, query_samples) = batch[0] # only 1-task per batch 

        def process_batch(batch_samples):
            """ 
            Helper function to process samples from either the support or query sets,
            and return a dictionary of input_ids, input_target_idx, label_ids and attention_mask
            (i.e. the expected data structure returned by meta_collate).
            """

            batch_inputs = []
            batch_input_target_idx = []
            batch_labels = []
            batch_max_seq_len = 0 

            for idx, (masked_tok_id, subword_samples) in enumerate(batch_samples.items()):
                # randomly assigns each subword_idx to a number in range(N) if not using 
                # return_standard_labels; otherwise use the actual token id that is masked
                # recall batch_samples is a dict({tok id: [samples]})
                for subword_sample in subword_samples:

                    if self.return_standard_labels:
                        batch_labels.append(masked_tok_id)
                    else:
                        batch_labels.append(idx) 
                        
                    if len(subword_sample) > batch_max_seq_len:
                        batch_max_seq_len = len(subword_sample)
                    batch_inputs.append(subword_sample)
                
            # NOTE: Padding token needs to be 1, in order to be consistent with HF tokenizer: 
            input_tensor = torch.ones((len(batch_inputs), batch_max_seq_len))

            for idx, sample in enumerate(batch_inputs): 
                input_tensor[idx, :len(sample)] = torch.tensor(sample)
                # NOTE: we are using CLS as the token we apply the classification layer over
                batch_input_target_idx.append(0)

            input_target_idx = torch.tensor(batch_input_target_idx)
            label_tensor = torch.tensor(batch_labels)
            attention_mask_tensor = (input_tensor != 1)

            processed_batch = {
                "input_ids": input_tensor.long(),
                "input_target_idx": input_target_idx.long(),
                "label_ids": label_tensor.long(),
                "attention_mask": attention_mask_tensor.int()
            }

            return processed_batch

        support_batch_list = [process_batch(support_sample) for support_sample in support_samples]    
        query_batch = process_batch(query_samples)

        return (task_name, support_batch_list, query_batch)
    

class MetaDataLoader(DataLoader):
    """
    Stripped down basic dataloader meant to be used with MetaDataset,
    note that MetaDataset does most of the heavy-lifting with processing 
    the data. 
    
    Copied from: 
    https://github.com/tristandeleu/pytorch-meta/blob/master/torchmeta/utils/data/dataloader.py#L32
    """
    def __init__(self, dataset, return_standard_labels=False, batch_size=1, shuffle=False, sampler=None,
                 batch_sampler=None, num_workers=0, pin_memory=False, drop_last=False, timeout=0,
                 worker_init_fn=None):
                
        """ Resetting basic defaults  """

        meta_collator = MetaCollator(return_standard_labels=return_standard_labels)

        super().__init__(dataset, batch_size=batch_size,
            shuffle=shuffle, sampler=sampler, batch_sampler=batch_sampler,
            num_workers=num_workers, collate_fn=meta_collator,
            pin_memory=pin_memory, drop_last=drop_last, timeout=timeout,
            worker_init_fn=worker_init_fn)