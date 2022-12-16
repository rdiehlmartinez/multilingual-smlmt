__author__ = 'Richard Diehl Martinez'
""" Implements a standard fully-supervised learning process (i.e. a baseline)"""

import copy
import itertools
import time
import logging

import torch

from .base import BaseLearner
from ..taskheads import TaskHead
from ..utils import move_to_device

# typing imports
from typing import Dict, List

logger = logging.getLogger(__name__)

class BaselineLearner(BaseLearner):

    def __init__(self, **kwargs) -> None: 
        """
        BaselineLearner implements a fully-supervised learning process to train a given base_model 
        (serves as a baseline). The main idea is to use what would be the support and 
        query sets to train the meta-model and instead simply use this data to train the model 
        in a supervised fashion. NOTE importantly that the task head now classifies over the 
        entire vocabulary (i.e. not just the vocabulary of the task).
        """
        
        super().__init__(**kwargs)
        assert(self.retain_lm_head is True), "BaselineLearner requires retain_lm_head to be True"
        

    ###### Model training methods ######

    def run_train_loop(
        self,
        support_batch_list: List[Dict[str, torch.Tensor]],
        query_batch: Dict[str, torch.Tensor],
        device: torch.device = None, 
    ) -> torch.Tensor:
        """ 
        Trains a model to perform a language modeling task by giving it access to the data in the 
        support_batch_list and query_batch. In the context of the baseline, this simply amounts to
        training the model in a supervised fashion.

        Args:
            * support_batch_list: A list of task batches, each batch of task data is represented
                as a dictionary containing the following information:
                * input_ids (torch.tensor): Input tensors of shape (N*K, max_seq_len)
                * input_target_idx (torch.tensor): Tensor indicating for each sample at what index
                    we apply the final classification layer 
                * label_ids (torch.tensor): Tensor of labels corresponding to masked out subword id
                * attention_mask (torch.tensor): Tensor indicating which tokens in input_ids are
                    not pad tokens
            * query_batch: Same as the dictionary structure of the support_batch, but for the data
                of the query set 
            * device: Optional string to specify a device to override base_device

        Returns: 
            * loss (torch.tensor): A tensor containing the loss that results from the inner loop 
        """

        if device is None:
            device = self.base_device

        # retain LM head is required for baseline

        self.train()

        num_total_samples = len(support_batch_list) + 1 # +1 for query batch

        average_loss = 0.0
        
        for data_batch in support_batch_list + [query_batch]:
            data_batch = move_to_device(data_batch, device)

            outputs = self.base_model(
                input_ids=data_batch['input_ids'],
                attention_mask=data_batch['attention_mask'],
            )

            _, loss = self._compute_task_loss(
                outputs, 
                data_batch,
                self.retained_lm_head_weights,
                'classification'
            )

            average_loss += loss.detach() / num_total_samples

             # --- 1) UPDATING THE BASE MODEL PARAMETERS
             
            grads = torch.autograd.grad(
                outputs=loss,
                inputs=[p for p in self.base_model.parameters() if p.requires_grad],
                retain_graph=True
            )


            for param, grad in zip(
                [p for p in self.base_model.parameters() if p.requires_grad],
                grads
            ):
                if param.grad is not None:
                    param.grad += grad.detach() / num_total_samples
                else:
                    param.grad = grad.detach() / num_total_samples

            
            # --- [ OPTIONAL 2)] UPDATING THE RETAINED TASK HEAD PARAMETERS
            
            lm_head_grads = torch.autograd.grad(
                outputs=loss,
                inputs=self.retained_lm_head_weights.values(),
                retain_graph=True
            )
            for param, lm_head_grad in zip(
                self.retained_lm_head_weights.values(),
                lm_head_grads
            ):
                if param.grad is not None:
                    param.grad += lm_head_grad.detach() / num_total_samples
                else:
                    param.grad = lm_head_grad.detach() / num_total_samples

                
        return average_loss