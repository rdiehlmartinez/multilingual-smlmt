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

# from transformers import AdamW
from torch.optim import AdamW
from collections import defaultdict

from .base import BaseMetaLearner
from ..taskheads import TaskHead
from ..utils import move_to_device
from ..datasets import NLUDataLoader

# typing imports 
from typing import Dict, Tuple, List, Callable, Union
from ..datasets import NLUDataset
from ..utils.data import ShuffledIterableDataset
from ..evaluation.evaluator import Metric


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
    def _run_innerloop(
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

        innerloop_optimizer_param_groups = self.innerloop_optimizer_param_groups()

        # Getting lrs for each parameter group
        innerloop_optimizer_lrs = {'lr': []}
        for param_group in innerloop_optimizer_param_groups: 
            innerloop_optimizer_lrs['lr'].append(param_group['lr'])

        innerloop_optimizer_params = [
            {'params': pg['params']} for pg in innerloop_optimizer_param_groups
        ]

        # Setting up the inner loop optimizer; NOTE we need to pass in the learning rates 
        # for each parameter group as an override argument to the higher diffopt optimizer - 
        # this is an unfortunate hack 
        innerloop_optimizer = torch.optim.SGD(innerloop_optimizer_params, lr=0.0)

        with torch.backends.cudnn.flags(enabled=True), higher.innerloop_ctx(
            self.base_model,
            innerloop_optimizer,
            copy_initial_weights=False,
            track_higher_grads=False,
            override=innerloop_optimizer_lrs,
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
    # NOTE: this method might be moved to the base class
    def run_train_loop(
        self,
        support_batch_list: List[Dict[str, torch.Tensor]],
        query_batch: Dict[str, torch.Tensor],
        device: torch.device = None, 
    ) -> torch.Tensor: 
        """
        Implements the task training (inner loop) of the MAML process.
        The main logic lives in the private `_run_innerloop` method() that is specific to MAML.
        
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
       
        num_adaptation_steps = self.num_innerloop_steps

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
                num_adaptation_steps = self.num_innerloop_steps - 1 

        loss = self._run_innerloop(
            support_batch_list,
            query_batch,
            lm_head_weights,
            num_adaptation_steps
        )

        return loss


    def run_evaluation(
        self, 
        finetune_dataset: ShuffledIterableDataset,
        eval_dataset: NLUDataset,
        metric: Metric,
        task_type: str, 
        task_head_init_method: str,
        num_classes: int,
        max_epochs: int = 10,
        batch_size: int = 256,
        device: torch.device = None,
        return_finetune_info: bool = True,
    ) -> Union[Tuple[float, float], Tuple[float, float, Dict[str, List[float]]]]:
        """
        Runs finetuning of the model on the support set data stored in the support_dataloader,
        and then evaluates the model on the eval_dataloader.

        Args: 
            * finetune_dataset: An NLU Dataset containing the data for finetuning the
                pre-trained model on a given NLU task
            * eval_dataset: An NLU Dataset containing the evaluation set data for the given NLU 
                task
            * metric: A callable metric instance that uses the finetuned model to compute the
                target evaluation metric on the eval_dataloader
            * task_type: A string indicating the type of task (e.g. 'classification', 'qa', etc.)
            * task_head_init_method: A string indicating the method used to initialize the task
                head weights
            * num_classes: An integer indicating the number of classes in the task
            * max_epochs: An integer indicating the max number of epochs to run the finetuning;
                we break out of the loop if the validation loss stops improving
            * batch_size: An integer indicating the batch size to use for the finetuning
            * device: Optional string to specify a device to override base_device
            * return_finetune_info: A boolean indicating whether to return the training losses
                and accuracies for the finetuning process

        Returns (if return_finetune_info is False):
            * eval_metric: The evaluation metric for the given task
            * total_eval_loss: A float indicating the loss of the model on the evaluation set 
        Returns (if return_finetune_info is True):
            * eval_metric: The evaluation metric for the given 
            * total_eval_loss: A float indicating the loss of the model on the evaluation set
            * finetuning_info: A dictionary containing the following information:
                * train_losses: A list of training losses for every evaluation step
                * train_accuracies: A list of training accuracies for every evaluation step
                * val_losses: A list of validation losses for every evaluation step
                * val_accuracies: A list of validation accuracies for every evaluation step
        """

        assert 'protomaml' not in task_head_init_method,\
            "Protomaml task head initialization is not supported for evaluation"
        
        if device is None:
            device = self.base_device

        self.base_model.to(device)

        # Setting up the task head for the task
        with torch.no_grad(): 
            init_kwargs = self.get_task_init_kwargs(
                task_type,
                task_head_init_method,
                num_classes,
                device=device
            )
            task_head_weights = TaskHead.initialize_task_head(**init_kwargs)

        finetune_model = copy.deepcopy(self.base_model)

        # Training parameters - should not be touched by the user
        MAX_PATIENCE = 5
        EVAL_EVERY_N_STEPS = 100
        INITIAL_LR = 1e-5

        patience = MAX_PATIENCE
        best_dev_metric = None

        dev_batch = move_to_device(finetune_dataset.dev_batch, device)

        # Setting up the optimizer
        finetune_optimizer_param_groups = self.finetune_optimizer_param_groups(
            finetune_model,
            task_head_weights,
            add_decay_information=True,
            weight_decay_val=0.0
        )

        finetune_optimizer = AdamW(finetune_optimizer_param_groups, lr=INITIAL_LR)

        finetune_model.train()

        total_step_num = 0

        early_exit_training = False

        if return_finetune_info:
            # Setting up the training info dictionary
            finetune_info = [] 

        for epoch in range(max_epochs):

            if early_exit_training:
                break
            
            finetune_dataloader = NLUDataLoader(
                finetune_dataset,
                batch_size=batch_size
            )

            # Finetune the model on the data in the finetune dataloader 
            for finetune_batch in finetune_dataloader:

                if early_exit_training:
                    break

                finetune_optimizer.zero_grad()

                finetune_batch = move_to_device(finetune_batch, device)

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

                total_step_num += 1

                if total_step_num % EVAL_EVERY_N_STEPS == 0:
                    # Evaluating the model on the dev set to possbily break out early
                    with torch.no_grad():
                        finetune_model.eval()

                        outputs = finetune_model(
                            input_ids=dev_batch['input_ids'],
                            attention_mask=dev_batch['attention_mask']
                        )

                        dev_logits, dev_loss = self._compute_task_loss(
                            outputs,
                            dev_batch,
                            task_head_weights,
                            task_type=task_type
                        )
                        
                        dev_predictions = torch.argmax(dev_logits, dim=-1).tolist()
                        dev_labels = dev_batch['label_ids'].tolist()

                        dev_metric = metric(dev_predictions, dev_labels)

                        if best_dev_metric is None or \
                            metric.summary(dev_metric, best_dev_metric) == dev_metric:

                            best_dev_metric = dev_metric
                            patience = MAX_PATIENCE
                        else:
                            patience -= 1
                            if patience == 0:
                                early_exit_training = True 

                        finetune_model.train()

                        if return_finetune_info:
                            finetune_info.append({
                                'train_loss': loss.item(),
                                'dev_loss': dev_loss.item(),
                                'dev_metric': dev_metric,
                                'step_num': total_step_num
                            })
        
        # Running full evaluation
        finetune_model.eval()

        eval_labels = []
        eval_predictions = []
    
        total_eval_loss = 0.0
        total_eval_samples = 0

        eval_dataloader = NLUDataLoader(
            eval_dataset,
            batch_size=batch_size
        )

        with torch.no_grad():

            for eval_batch in eval_dataloader: 
                eval_batch = move_to_device(eval_batch, device)

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
                eval_labels.extend(eval_batch['label_ids'].tolist())

                batch_size = eval_logits.size(0)
                total_eval_loss += eval_loss.detach().item() * batch_size # loss avg across batch
                total_eval_samples += batch_size 

            total_eval_loss /= total_eval_samples

            eval_metric = metric(eval_predictions, eval_labels)

        if return_finetune_info:
            return (eval_metric, total_eval_loss, finetune_info)
        else:
            return (eval_metric, total_eval_loss)