__author__ = "Richard Diehl Martinez"
""" Utility functions related to data processing functionality """

import torch
import random
import itertools

from typing import Dict, Union, List, Tuple


def move_to_device(
    batch: Union[Dict[str, torch.Tensor], torch.Tensor], device: torch.device
) -> Dict[str, torch.Tensor]:
    """
    Helper functionality for moving a batch of data that is structured as a dictionary (possibly
    nested) or as just a Tensor onto a certain device.

    Args:
        * batch (dict): dictionary of data that needs to be moved to a device
        * device (torch.device): device to move the data to

    Returns:
        * updated_batch (dict): dictionary of data that has been moved to a device
    """

    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    elif not isinstance(batch, dict):
        raise ValueError("batch must be either a dictionary or a tensor")

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
    use_smlmt_labels: bool,
) -> Dict[str, torch.Tensor]:
    """
    Base functionality for collating a batch of samples. Is used to generate batches of
    language modeling task data.
    Args:
        * batch: Either a dictionary or a list of tuples that can be iterated over and
            yields a tuple containing a label and samples corresponding to a label.
        * use_smlmt_labels (bool): Whether to transform the labels to be in range [0, n] or
                whether to collate the batch to use the token ids of the masked token. Only
                relevant for meta-training.

    Returns:
        * processed_batch (dict): Dictionary containing the following:
            * input_ids (torch.tensor): Input tensors of shape (N*K, max_seq_len)
            * label_ids (torch.tensor): Tensor of labels corresponding to masked out subword id
            * attention_mask (torch.tensor): Tensor indicating which tokens in input_ids
                are not pad tokens
    """

    if isinstance(batch, dict):
        batch = batch.items()

    batch_inputs = []
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
        input_tensor[idx, : len(sample_inputs)] = torch.tensor(sample_inputs)

    label_tensor = torch.tensor(batch_labels)
    attention_mask_tensor = input_tensor != 1

    processed_batch = {
        "input_ids": input_tensor.long(),
        "label_ids": label_tensor.long(),
        "attention_mask": attention_mask_tensor.int(),
    }

    return processed_batch
