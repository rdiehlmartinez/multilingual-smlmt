__author__ = 'Richard Diehl Martinez'
""" Utility functions related to data processing functionality """

import torch

def move_to_device(batch, device):
    """ 
    Helper functinoality for moving a batch of data that is structured as a dictionary (possibly
    nested) onto a certain device.
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

def base_collate_fn(batch, return_standard_labels): 
    """
    Base functionality for collating a batch of samples. Is used to both generate batches of 
    language modeling task data, as well as NLU task data.
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

            if return_standard_labels:
                batch_labels.append(label)
            else:
                batch_labels.append(idx) 
                
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