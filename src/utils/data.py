__author__ = 'Richard Diehl Martinez'
""" Utility functions related to data processing functionality """

import torch
import random
import itertools

from typing import Dict, Union, List, Tuple

def move_to_device(
    batch: Dict[str, torch.Tensor],
    device: torch.device
) -> Dict[str, torch.Tensor]:
    """ 
    Helper functionality for moving a batch of data that is structured as a dictionary (possibly
    nested) onto a certain device.
    
    Args: 
        * batch (dict): dictionary of data that needs to be moved to a device
        * device (torch.device): device to move the data to
    
    Returns: 
        * updated_batch (dict): dictionary of data that has been moved to a device
    """

    updated_batch = {}
    for key, val in batch.items():
        if isinstance(val, dict):
            if key not in updated_batch:
                updated_batch[key] = {}
            for sub_key, sub_val in val.items():
                if sub_val is not None:
                    updated_batch[key][sub_key] = sub_val.to(device)
        else:
            if val is not None:
                updated_batch[key] = val.to(device)
    return updated_batch

def base_collate_fn(
    batch: Union[List[Tuple[int, List[int]]], Dict[int, List[List[int]]]],
    use_smlmt_labels: bool
) -> Dict[str, torch.Tensor]:
    """
    Base functionality for collating a batch of samples. Is used to both generate batches of 
    language modeling task data, as well as NLU task data.

    Args: 
        * batch: Either a dictionary or a list of tuples that can be iterated over and 
            yields a tuple containing a label and samples corresponding to a label.
        * use_smlmt_labels (bool): Whether to transform the labels to be in range [0, n] or
                whether to collate the batch to use the token ids of the masked token. Only 
                relevant for meta-training; for NLU tasks this should always be True.

    Returns:
        * processed_batch (dict): Dictionary containing the following: 
            * input_ids (torch.tensor): Input tensors of shape (N*K, max_seq_len)
            * input_target_idx (torch.tensor): Tensor indicating for each sample at what index
                we apply the final classification layer
            * label_ids (torch.tensor): Tensor of labels corresponding to masked out subword id
            * attention_mask (torch.tensor): Tensor indicating which tokens in input_ids 
                are not pad tokens
    """

    if isinstance(batch, dict):
        batch = batch.items()

    batch_inputs = []
    batch_input_target_idx = []
    batch_labels = []
    batch_max_seq_len = 0 

    for idx, (label, samples) in enumerate(batch):
        """
        Each element of the data batch needs to have both a label as well as associated samples.
        NOTE: 
        During training and finetuning these samples are a list of K samples associated with a 
        given label. During evaluation samples is a single sample that we cast into a list to
        conform with the expected format. 
        """

        if not isinstance(samples[0], list): 
            # NOTE: samples is a single sample of data - need to wrap the sample into a list 
            # to align with the expected format 
            samples = [samples]

        for sample in samples:

            if use_smlmt_labels:
                batch_labels.append(idx)    
            else:
                batch_labels.append(label)
                
            if len(sample) > batch_max_seq_len:
                batch_max_seq_len = len(sample)

            batch_inputs.append(sample)
        
    # NOTE: Padding token needs to be 1, in order to be consistent with HF tokenizer
    input_tensor = torch.ones((len(batch_inputs), batch_max_seq_len))

    for idx, sample_inputs in enumerate(batch_inputs): 
        input_tensor[idx, :len(sample_inputs)] = torch.tensor(sample_inputs)

    # NOTE: the target idx is index over which we want to apply the classifier 
    # for both LM and NLU tasks we apply the classifier over the CLS token (index 0), but 
    # this can be readily changed 
    input_target_idx = torch.zeros((len(batch_inputs)))
    label_tensor = torch.tensor(batch_labels)
    attention_mask_tensor = (input_tensor != 1)

    processed_batch = {
        "input_ids": input_tensor.long(),
        "input_target_idx": input_target_idx.long(),
        "label_ids": label_tensor.long(),
        "attention_mask": attention_mask_tensor.int()
    }

    return processed_batch


class ShuffledIterableDataset(torch.utils.data.IterableDataset):
    """
    NOTE: Copied from https://discuss.pytorch.org/t/how-to-shuffle-an-iterable-dataset/64130/5

    This class is used to shuffle an IterableDataset. This is useful for training on datasets
    that are too large to fit into memory. The dataset is shuffled by sampling a buffer of
    size `buffer_size` from the dataset and then sampling from this buffer. This is repeated
    until the dataset is exhausted. Note that the dataset must have at least `buffer_size`
    elements for this to work.

    """

    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        buffer_size: int = 10000,
        hold_out_dev_batch: bool = True,
        batch_size: int = 256,
    ):
        super().__init__()
        self.dataset = dataset
        self.buffer_size = buffer_size
        self.hold_out_dev_batch = hold_out_dev_batch
        self.batch_size = batch_size

        assert(self.buffer_size > self.batch_size), "Buffer size must be larger than batch size"

        if self.hold_out_dev_batch: 
            self.hold_out_idxs, self.dev_batch = self._generate_dev_batch()
        else:
            self.dev_batch = None

    def _generate_dev_batch(self) -> Tuple[List[int], Dict[str, torch.Tensor]]:
        """
        Generates a batch of data that is used for validation during training. The dev batch is 
        generated by random picking self.batch_size number of indices from the first buffer of 
        data that is generated and using these indices to sample the hold out dev batch data. We 
        then keep track of these indices so that during training these are skipped over when 
        generating the training batches.

        Returns: 
            * Tuple containing the following:
                * hold_out_idx (Set): List of indices that are used to sample the dev batch
                * dev_batch (Dict): Batch of dev data that is used for validation during 
                    finetuning; stored in the form of a dictionaru containing the following:
                    * input_ids (torch.tensor): Input tensors of shape (N*K, max_seq_len)
                    * input_target_idx (torch.tensor): Tensor indicating for each sample at what
                        index we apply the final classification layer
                    * label_ids (torch.tensor): Label for the NLU task
                    * attention_mask (torch.tensor): Tensor indicating which tokens in input_ids
                        are not pad tokens                 
        """

        dataset_iter = iter(self.dataset)
        
        hold_out_idxs = set(random.sample(range(self.buffer_size), self.batch_size))

        dev_batch = []

        for idx in range(self.buffer_size):
            
            curr_sample = next(dataset_iter)

            if idx in hold_out_idxs:
                dev_batch.append(curr_sample)

        dev_batch = base_collate_fn(dev_batch, use_smlmt_labels=False)

        return (hold_out_idxs, dev_batch)


    def __iter__(self):
        """
        This function is called when the dataset is iterated over. It returns an iterator
        over the shuffled dataset.
        """
        shufbuf = []

        dataset_iter = iter(self.dataset)
        
        for idx in range(self.buffer_size):

            curr_sample = next(dataset_iter)

            if self.hold_out_dev_batch and idx in self.hold_out_idxs:
                # NOTE: we hold out a single batch for dev set, so we skip over this batch
                continue 
            
            shufbuf.append(curr_sample)
        
        if self.hold_out_dev_batch:
            # NOTE: we hold out a single batch for dev set, so we need to add a different 
            # batch to the buffer
            for _ in range(self.batch_size):
                shufbuf.append(next(dataset_iter))

        try:
            while True:
                try:
                    item = next(dataset_iter)
                    evict_idx = random.randint(0, self.buffer_size - 1)
                    yield shufbuf[evict_idx]
                    shufbuf[evict_idx] = item
                except StopIteration:
                    break

            while len(shufbuf) > 0:
                yield shufbuf.pop()
        except GeneratorExit:
            pass