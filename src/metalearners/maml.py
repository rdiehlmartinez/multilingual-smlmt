__author__ = 'Richard Diehl Martinez'
"""
Implements Model Agnostic Meta Learning: https://arxiv.org/abs/1703.03400
"""

import time
import itertools
import logging
import copy

# custom higher version
import lib.higher.higher as higher

import torch
import torch.distributed as dist
from transformers import AdamW

from .base import BaseMetaLearner
from ..taskheads import TaskHead
from ..utils import move_to_device

# typing imports 
from typing import Dict, Tuple, List, Dict, Iterator, Any, Union

logger = logging.getLogger(__name__)

class MAML(BaseMetaLearner):
    def __init__(self, **kwargs) -> None:
        """
        MAML implements a basic type of BaseMetaLearner.
        The core idea is to train a model using a two-loop approach:
            * the inner loop is used to adapt the model to a given task
            * the outer loop is used to update the model parameters based on the loss of the 
                adapted model on the query set of the task.
        """
        super().__init__(**kwargs)

    ###### Model training methods #####
 
    ### Helper function for adapting the functionalized parameters based on some data_batch
    def run_adaptation(
        self,
        support_batch_list: List[Dict[str, torch.Tensor]],
        query_batch: Dict[str, torch.Tensor],
        lm_head_weights: Dict[str, torch.Tensor],
        num_adaptation_steps: int,
    ) -> float:
        """ 
        Helper function for adapting the weights of the base_model on a given task using the 
        training data in support_batch_list. 

        Args:
            * support_batch_list: A list of task batches, each batch of task data is represented
                as a dictionary containing the following information:
                * input_ids (torch.tensor): Input tensors of shape (N*K, max_seq_len)
                * input_target_idx (torch.tensor): Tensor indicating for each sample at what index
                    we apply the final classification layer
                * label_ids (torch.tensor): Tensor of labels corresponding to masked out subword id
                * attention_mask (torch.tensor): Tensor indicating which tokens in input_ids are
                    not pad tokens
            * query_batch: A dictionary containing the query data. It should be of the same
                structure as the support_batch_list.
            * lm_head_weights: A dictionary containing the weights of the lm head to be used
                for the adaptation.
            * num_adaptation_steps: The number of adaptation steps to perform.        
        Returns:
            * loss (float): The loss value of the adaptation process.

        """              

        # Getting lrs for each parameter group
        inner_loop_optimizer_lrs = {'lr': []}
        for param_group in self.innerloop_optimizer_param_groups(): 
            inner_loop_optimizer_lrs['lr'].append(param_group['lr'])

        inner_loop_optimizer_params = [
            {'params': pg['params']} for pg in self.innerloop_optimizer_param_groups()
        ]

        # Setting up the inner loop optimizer; note that we need pass in the learning rates 
        # for each parameter group as an override argument to the higher diffopt optimizer
        inner_loop_optimizer = torch.optim.SGD(inner_loop_optimizer_params, lr=0.0)

        with torch.backends.cudnn.flags(enabled=True), higher.innerloop_ctx(
            self.base_model,
            inner_loop_optimizer,
            copy_initial_weights=False,
            track_higher_grads=False,
            override=inner_loop_optimizer_lrs,
        ) as (flearner, diffopt):
            
            # Running the inner loop adaptation to the support set 
            self.train()
            flearner.train()
            flearner.zero_grad()

            # --- RUNNING ADAPTATION
            for num_step in range(num_adaptation_steps):

                if self.use_multiple_samples: 
                    support_batch = support_batch_list[num_step]
                else:
                    support_batch = support_batch_list[0]

                # Forward pass
                outputs = flearner(
                    input_ids=support_batch['input_ids'],
                    attention_mask=support_batch['attention_mask']
                )

                _, loss = self._compute_task_loss(
                    outputs,
                    support_batch,
                    lm_head_weights, 
                    task_type='classification'
                )

                # updating the task head
                classifier_grads = torch.autograd.grad(
                    outputs=loss,
                    inputs=lm_head_weights.values(),
                    retain_graph=True
                )
                for idx, weight_name in enumerate(lm_head_weights.keys()):
                    lm_head_weights[weight_name] = lm_head_weights[weight_name] - \
                                                    self.classifier_lr * classifier_grads[idx]

                # Inner loop backward pass
                diffopt.step(loss)

            # --- FINISHED ADAPTATION        

            # Disable dropout
            for module in flearner.modules():
                if isinstance(module, torch.nn.Dropout):
                    module.eval()

            query_outputs = flearner(
                    input_ids=query_batch['input_ids'],
                    attention_mask=query_batch['attention_mask'],
            )
            _, query_loss = self._compute_task_loss(
                query_outputs,
                query_batch,
                lm_head_weights,
                task_type='classification'
            )

            """ 
            NOTE: It is up to us to ensure that the gradients with respect to the learnable 
            meta parameters are computed and stored in the grad attribute of the meta parameters
            so that the meta optimizer can in the outerloop update the meta parameters.
            """

            # --- 1) UPDATING THE BASE MODEL PARAMETERS
            
            # gradient of query_loss with respect to the adapted params --> First Order Approximation
            meta_grads = torch.autograd.grad(
                outputs=query_loss, 
                inputs=[p for p in flearner.parameters() if p.requires_grad],
                retain_graph=True
            )

            if "protomaml" in self.lm_head_init_method:                
                # if using protomaml, we need to account for the gradient that results from 
                # the gradient of the query loss with respect to initialization of the task head
                proto_grads = torch.autograd.grad(
                    outputs=query_loss,
                    inputs=[p for p in self.base_model.parameters() if p.requires_grad],
                    retain_graph=True
                )
                meta_grads = [mg + pg for (mg, pg) in zip(meta_grads, proto_grads)]

            for param, meta_grad in zip(
                [p for p in self.base_model.parameters() if p.requires_grad],
                meta_grads
            ):
                if param.grad is not None:
                    param.grad += meta_grad.detach()
                else:
                    param.grad = meta_grad.detach()

            # --- [ OPTIONAL 2)] UPDATING THE RETAINED TASK HEAD PARAMETERS
            if self.retain_lm_head:
                # gradient of query_loss with respect to the task head weights
                lm_head_grads = torch.autograd.grad(
                    outputs=query_loss,
                    inputs=self.retained_lm_head_weights.values(),
                    retain_graph=True
                )
                for param, lm_head_grad in zip(
                    self.retained_lm_head_weights.values(),
                    lm_head_grads
                ):
                    if param.grad is not None:
                        param.grad += lm_head_grad.detach()
                    else:
                        param.grad = lm_head_grad.detach()


            # --- 3) UPDATING THE CLASSIFIER LEARNING RATE 

            if self.classifier_lr.grad is not None: 
                self.classifier_lr.grad += torch.autograd.grad(
                    outputs=query_loss,
                    inputs=self.classifier_lr,
                    retain_graph=True
                )[0].detach()
            else:
                self.classifier_lr.grad = torch.autograd.grad(
                    outputs=query_loss,
                    inputs=self.classifier_lr,
                    retain_graph=True
                )[0].detach()


            # --- 4) UPDATING THE INNER LOOP LEARNING RATES

            for layer_num, inner_layer_lr in self.inner_layers_lr.items():
                
                if inner_layer_lr.grad is not None:
                    inner_layer_lr.grad += torch.autograd.grad(
                        outputs=query_loss,
                        inputs=inner_layer_lr,
                        retain_graph=True
                    )[0].detach()
                else:
                    inner_layer_lr.grad = torch.autograd.grad(
                        outputs=query_loss,
                        inputs=inner_layer_lr,
                        retain_graph=True
                    )[0].detach()

        return query_loss.detach().item()


    ### Main Inner Training Loop 
    def run_inner_loop(
        self,
        support_batch_list: List[Dict[str, torch.Tensor]],
        query_batch: Dict[str, torch.Tensor],
        device: torch.device = None, 
    ) -> torch.Tensor: 
        """
        Implements the inner loop of the MAML process.
        
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
            * loss (torch.Tensor): Loss value of the inner loop calculations
        """
        if device is None:
            device = self.base_device

        # Moving data to appropriate device
        support_batch_list = [
            move_to_device(support_batch, device) 
            for support_batch in support_batch_list
        ]
        query_batch = move_to_device(query_batch, device)
       
        num_adaptation_steps = self.num_inner_loop_steps

        # Setting up LM head for task training (either using existing one or setting up new one)
        if self.retain_lm_head:
            # we need to clone the parameters of the LM head so that we can temporarily 
            # update them during the inner loop
            lm_head_weights = {
                key: torch.clone(val) for key, val in self.retained_lm_head_weights.items()
            }

        else:
            init_kwargs = self.get_task_init_kwargs(
                'classification',
                self.lm_head_init_method,
                self.lm_head_n,
                data_batch=support_batch_list[0] if 'protomaml' in self.lm_head_init_method else None, 
                device=device
            )

            lm_head_weights = TaskHead.initialize_task_head(**init_kwargs)
            
            if 'protomaml' in self.lm_head_init_method and self.use_multiple_samples:
                # If we're using protomaml, the first batch is used for sampling the task head 
                support_batch_list = support_batch_list[1:]
                num_adaptation_steps = self.num_inner_loop_steps - 1 

        loss = self.run_adaptation(
            support_batch_list,
            query_batch,
            lm_head_weights,
            num_adaptation_steps
        )

        return loss


    def run_evaluation(
        self, 
        finetune_dataloader: torch.utils.data.DataLoader,
        eval_dataloader: torch.utils.data.DataLoader,
        task_type: str, 
        task_head_init_method: str,
        num_classes: int,
    ) -> Tuple[List[int], float]:
        """
        Runs finetuning of the model on the support set data stored in the support_dataloader,
        and then evaluates the model on the eval_dataloader.

        Args: 
            * finetune_dataloader: A torch.utils.data.DataLoader object containing the data for 
                finetuning the pre-trained model for a given NLU task
            * eval_dataloader: A torch.utils.data.DataLoader object containing the evaluation
                set data for a given NLU task
            * task_type: A string indicating the type of task (e.g. 'classification', 'qa', etc.)
            * task_head_init_method: A string indicating the method used to initialize the task
                head weights
            * num_classes: An integer indicating the number of classes in the task

        Returns:
            * eval_predictions: A list of integers indicating the predicted class for each sample
                in the evaluation set
            * eval_loss: A float indicating the loss of the model on the evaluation set 
        """

        if 'protomaml' in task_head_init_method:
            task_init_batch = move_to_device(next(iter(finetune_dataloader)), self.base_device)

        # Setting up the task head for the task
        with torch.no_grad(): 
            init_kwargs = self.get_task_init_kwargs(
                task_type,
                task_head_init_method,
                num_classes,
                data_batch=task_init_batch if 'protomaml' in task_head_init_method else None,
            )
            task_head_weights = TaskHead.initialize_task_head(**init_kwargs)

        finetune_model = copy.deepcopy(self.base_model)
        # NOTE: We set up an optimizer to train the finetuned_model on the finetune data 
        # (i.e. the support set data) -- we do not use the optimizer that is used to train the
        # base_model on the meta-training data.
        finetune_model_params_groups = itertools.chain(
            # We both optimize the finetune model, along with the task head weights 
            self.innerloop_optimizer_param_groups(
                base_model_override=finetune_model,
                cast_lr_to_float=True
            ),
            [{
                'params': task_head_weights.values(),
                'lr': self.classifier_lr,
            }]
        )

        finetune_optimizer = AdamW(finetune_model_params_groups)

        patience = 5
        min_train_loss = None

        # Finetuning the model on the data in the finetune dataloader 
        finetune_model.train()
        for finetune_batch in itertools.cycle(finetune_dataloader):
            finetune_optimizer.zero_grad()

            finetune_batch = move_to_device(finetune_batch, self.base_device)

            outputs = finetune_model(
                input_ids=finetune_batch['input_ids'],
                attention_mask=finetune_batch['attention_mask']
            )

            _, loss = self._compute_task_loss(
                outputs,
                finetune_batch,
                task_head_weights, 
                task_type=task_type
            )

            loss.backward()
            finetune_optimizer.step()

            # TODO: BETTER EARLY STOPPING

            # We need to train the model to convergence on the finetune data, so we don't
            # break out of the loop until we've gone through all the batches in the dataloader
            # (i.e. we don't use the early stopping mechanism that is used in the inner loop)
            
            if min_train_loss is None or loss < min_train_loss:
                min_train_loss = loss
                patience = 3

            if loss > min_train_loss: 
                patience -= 1
                if patience == 0:
                    break
        
        # Running full evaluation
        finetune_model.eval()

        eval_predictions = []
        total_eval_loss = 0.0
        total_eval_samples = 0

        for eval_batch in eval_dataloader: 
            eval_batch = move_to_device(eval_batch, self.base_device)

            eval_outputs = finetune_model(
                input_ids=eval_batch['input_ids'],
                attention_mask=eval_batch['attention_mask'],
            )

            eval_logits, eval_loss = self._compute_task_loss(
                eval_outputs, 
                eval_batch,
                task_head_weights,
                task_type=task_type
            )

            eval_predictions.extend(torch.argmax(eval_logits, dim=-1).tolist())

            batch_size = eval_logits.size(0)
            total_eval_loss += eval_loss.item() * batch_size # loss is averaged across batch
            total_eval_samples += batch_size 

        total_eval_loss /= total_eval_samples

        return (eval_predictions, total_eval_loss)