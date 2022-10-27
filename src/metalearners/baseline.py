__author__ = 'Richard Diehl Martinez'
""" Implements a standard fully-supervised learning process (i.e. a baseline)"""

import copy
import itertools
import time
import logging

from multiprocessing.queues import Empty as EmptyQueue

import torch
import torch.distributed as dist

from .base import BaseLearner
from ..taskheads import TaskHead
from ..utils import move_to_device

# typing imports

from ..models import BaseModel

logger = logging.getLogger(__name__)

# TODO: Add implementation of BaselineLearner
raise NotImplementedError("BaselineLearner not implemented yet")

class BaselineLearner(BaseLearner):

    def __init__(
        self,
        base_model: BaseModel,
        optimizer_type: str = 'adam',
        lr: int = 1e-2,
        **kwargs
    ) -> None: 
        """
        BaselineLearner implements a fully-supervised learning process to train a given base_model 
        (serves as a baseline). 

        Args: 
            * base_model (implementation of BaseModel): The model to be trained
            * optimizer_type (str): The type of optimizer (e.g. 'adam')
            * lr (int): Learning rate of the optimizer
        """
        
        super().__init__(base_model, *args, **kwargs)

        # setting up optimizer
        base_params = [p for p in self.base_model.parameters() if p.requires_grad]
        all_params = itertools.chain(base_params, 
                                     self.retained_lm_head.values() if self.retain_lm_head else [])
        if optimizer_type == 'adam': 
            self.optimizer = torch.optim.Adam(params=all_params, lr=float(lr))
        else: 
            logger.exception(f"Invalid optimizer type: {optimizer_type}")
            raise Exception(f"Invalid optimizer type: {optimizer_type}")

    ###### Model training methods ######

    def run_inner_loop(self, support_batch, query_batch=None, device=None, *args, **kwargs): 
        """ 
        Run an inner loop optimization step. Usually this is in the context of meta-learning, but
        in the case of a baseline model an inner_loop simply amounts to running a forward pass
        through the model and returning the corresponding loss.

        Args:
            * support_batch: a dictionary containing the following information for the support set
                * input_ids (torch.tensor): Input tensors
                * input_target_idx (torch.tensor): Tensor indicating for each sample at what index
                    we apply the final classification layer 
                * label_ids (torch.tensor): Tensor of labels corresponding to masked out subword id
                * attention_mask (torch.tensor): Tensor indicating which tokens in input_ids are
                    not pad tokens
            * query_batch [optional]: same as support_batch, but for the data of the query set.
                This argument os optional in the case of the baseline model. If it is provided, we 
                simply concatenate this data and the support batch data together.
            * device: Optional string to specify a device to override base_device

        Returns: 
            * loss (torch.tensor): a tensor containing the loss that results from the inner loop 
        """
        if device is None:
            device = self.base_device

        self.base_model.train()

        # Moving data to appropriate device
        support_batch = move_to_device(support_batch, device)
        if query_batch is not None:
            query_batch = move_to_device(query_batch, device)

        # combine support and query batch together 
        if query_batch: 
            input_batch = {}

            for key in support_batch.keys():
                support_batch_tensor = support_batch[key]
                query_batch_tensor = query_batch[key]

                if key == "input_ids" or key == "attention_mask":
                    # For input_ids and attention_mask we need to make sure that 
                    # the first dimension (sequence length dim) is the same 
                    # so we pad the shorter dim  
                    max_seq_len_support = support_batch_tensor.size(1)
                    max_seq_len_query = query_batch_tensor.size(1)

                    if max_seq_len_support != max_seq_len_query: 
                        tensor_dim_diff = abs(max_seq_len_support - max_seq_len_query)

                        if max_seq_len_support > max_seq_len_query: 
                            # expansion tensor batch size must match query batch size 
                            batch_size = query_batch_tensor.size(0)
                        else: 
                            # expansion tensor batch size must match support batch size 
                            batch_size = support_batch_tensor.size(0)

                        expansion_tensor_dims = (batch_size, tensor_dim_diff)
                        # NOTE: if expanding input_ids we use 1 to indicate pad;
                        #       or 0 to indicate pad if using attention_mask
                        if key == "input_ids": 
                            expansion_tensor = torch.ones(expansion_tensor_dims,
                                                          device=device)
                        else: 
                            expansion_tensor = torch.zeros(expansion_tensor_dims,
                                                           device=device)

                        if max_seq_len_support > max_seq_len_query: 
                            # expanding query 
                            query_batch_tensor = torch.cat((query_batch_tensor, expansion_tensor),
                                                          dim=1).long()
                        else: 
                            # expanding support 
                            support_batch_tensor = torch.cat((support_batch_tensor,
                                                              expansion_tensor),
                                                              dim=1).long()
                    
                input_batch[key] = torch.cat((support_batch_tensor, query_batch_tensor), dim=0)
        
        else: 
            input_batch = support_batch

        if self.retain_lm_head:
            lm_head = self.retained_lm_head
        else:
            init_kwargs = self.get_task_init_kwargs(self.lm_head_init_method, self.lm_head_n,
                                                    data_batch=input_batch, device=device)
            lm_head = TaskHead.initialize_task_head(task_type='classification',
                                                    method=self.lm_head_init_method,
                                                    init_kwargs=init_kwargs)

        outputs = self.base_model(input_ids=input_batch['input_ids'],
                                  attention_mask=input_batch['attention_mask'])

        _, loss = self._compute_task_loss(outputs, input_batch, lm_head, 
                                          task_type='classification')

        return loss

    ###### Model evaluation methods ######

    def run_finetuning(self, task_type, finetune_dataloader, n_labels, task_head_init_method,
                       max_finetuning_batch_steps=-1, **kwargs):
        """
        Finetunes the model on the data of finetune_dataloader. Creates a copy of the model and 
        continues to finetune the copy on a given NLU task (task_type) with the corresponding data
        stored in finetune_dataloader.

        Args: 
            * task_type (str): Type of task (e.g. 'classification')
            * finetune_dataloader (torch.data.Dataloader): The dataset for finetuning the model on
                is passed in as a dataloader (i.e. NLUDataloader)
            * n_labels (int): The number of labels in the given finetuning task
            * task_head_init_method (str): Method for initializing task head 
            * max_finetuning_batch_steps (int): Optional maximum number of batch steps to take 
                for model finetuning 

        Returns:
            * inference_params dict containing: 
                * finetuned_model ([torch.nn.Module]): Finetuned model
                * task_head_weights (dict): weights of classifier layer
        """ 
        self.base_model.train()

        ### Initializing the task head used for the downstream NLU task
        task_init_data_batch = move_to_device(next(iter(finetune_dataloader)), self.base_device)
        init_kwargs = self.get_task_init_kwargs(task_head_init_method, n_labels,
                                                data_batch=task_init_data_batch)
        task_head_weights = TaskHead.initialize_task_head(task_type=task_type,
                                                          method=task_head_init_method,
                                                          init_kwargs=init_kwargs)

        finetuned_model = copy.deepcopy(self.base_model)

        finetuned_task_head_weights = {}
        for k, p in task_head_weights.items():
            detached_p = p.detach()
            detached_p.requires_grad = True
            finetuned_task_head_weights[k] = detached_p

        finetuned_model_params = [p for p in finetuned_model.parameters() if p.requires_grad]
        finetune_params = itertools.chain(finetuned_model_params,
                                          finetuned_task_head_weights.values())
        finetune_optimizer = torch.optim.Adam(params=finetune_params)

        for batch_idx, data_batch in enumerate(finetune_dataloader):
            data_batch = move_to_device(data_batch, self.base_device)
            finetune_optimizer.zero_grad()

            # run SGD on the finetuned theta parameters
            outputs = finetuned_model(input_ids=data_batch['input_ids'],
                                      attention_mask=data_batch['attention_mask'],)

            _, loss = self._compute_task_loss(outputs, data_batch, finetuned_task_head_weights,
                                              task_type=task_type)

            loss.backward()
            finetune_optimizer.step()

            if max_finetuning_batch_steps > 0 and (batch_idx + 1) >= max_finetuning_batch_steps:
                break

        inference_params = {
            "finetuned_model": finetuned_model, 
            "task_head_weights": finetuned_task_head_weights
        }

        return inference_params

    def run_inference(self, task_type, inference_dataloader, finetuned_model, task_head_weights,
                      **kwargs):
        """
        This method is to be called after the run_finetuning. Runs inference on the data stored
        in inference_dataloader, using the finetuned_model.

        Args: 
            * task_type (str): Type of task (e.g. 'classification')
            * inference_dataloader (torch.data.Dataloader): The dataset for inference is passed
                in as a dataloader (in most cases this will be an NLUDataloader)
            * finetuned_model ([torch.nn.Module]): Finetuned model
            * task_head_weights (dict): weights of task head (classifier layer)

        Returns: 
            * predictions ([int]): a dictionary storing the model's predictions for each 
                datapoint passed in from the inference_dataloader as an int. 
            * loss (int): the value of the classification loss on the inference dataset.
        """

        predictions = []
        total_loss = 0.0
        total_samples = 0

        # Running final inference script over the evaluation data
        with torch.no_grad():

            finetuned_model.eval()

            for data_batch in inference_dataloader: 
                data_batch = move_to_device(data_batch, self.base_device)

                outputs = finetuned_model(input_ids=data_batch['input_ids'],
                                          attention_mask=data_batch['attention_mask'],)

                logits, loss = self._compute_task_loss(outputs, data_batch, task_head_weights,
                                                       task_type=task_type)
            
                predictions.extend(torch.argmax(logits, dim=-1).tolist())

                batch_size = logits.size(0)
                total_loss += loss.item() * batch_size # loss is averaged across batch
                total_samples += batch_size 

            total_loss /= total_samples

        return (predictions, total_loss)