__author__ = 'Richard Diehl Martinez'
""" Interface class for (meta) learners """

import abc 
# custom higher
import lib.higher.higher as higher
import logging
import math
import os
import re 
import time

from multiprocessing.queues import Empty as EmptyQueue
from collections import OrderedDict

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from ..taskheads import TaskHead, ClassificationHead
from ..utils import set_seed

# imports for typing 
from ..models import BaseModel
from typing import Tuple, List, Dict, Union, Any, Iterator
from multiprocessing import Queue, Event

logger = logging.getLogger(__name__)

class BaseLearner(torch.nn.Module, metaclass=abc.ABCMeta):

    def __init__(
        self,
        base_model: BaseModel,
        base_device: torch.device,
        seed: int,
        lm_head_init_method: str = 'protomaml',
        lm_head_n: Union[int, str] = 100,
        retain_lm_head: Union[bool, str] = False,
        use_multiple_samples: Union[bool, str] = True,
        **kwargs
    ) -> None:
        """
        BaseLearner establishes the inferface for the learner class.
        
        Args: 
            * base_model (implements a BaseModel)
            * base_device: What device to use for training (either 'cpu' or 'gpu) 
            * seed: Seed for reproducibility 
            * language_head_init_method: How to initialize the language head classifier layer
            * lm_head_n: Size of n-way classification used for generating the language 
                modeling tasks used for training.
            * retain_lm_head: Indicate whether we should maintain a single task head 
                that is learned over the course of meta training, or whether for each task we 
                should initialize a new task head.
            * use_multiple_samples: We can specify whether each learning step that the BaseLearner
                takes relies on only a single sample of an N-way K-shot task, or whether the 
                learner can have access to multiple sample of an N-way K-shot task. 
        """
        super().__init__()
        self.seed = seed
        self.base_device = base_device

        # hidden dimensions of the outputs of the base_model
        self.base_model = base_model
        self.base_model.to(self.base_device)
        self.base_model_hidden_dim = base_model.hidden_dim 

        self.lm_head_init_method = lm_head_init_method
        self.lm_head_n = int(lm_head_n)

        if isinstance(retain_lm_head, str):
            retain_lm_head = eval(retain_lm_head)
        self.retain_lm_head = retain_lm_head

        if self.retain_lm_head: 
            # If we only keep a single task head, then there is no obvious way how to initialize 
            # the task head with protomaml 
            assert("protomaml" not in self.lm_head_init_method),\
                "retain_task_head cannot be used with protomaml lm head initialization"
            init_kwargs = self.get_task_init_kwargs(
                'classification',
                self.lm_head_init_method,
                self.lm_head_n,
            )
            self.retained_lm_head_weights = TaskHead.initialize_task_head(**init_kwargs)

        else: 
            # If we are re-initializing the LM head for each training task, then we should use 
            # protomaml
            if "protomaml" not in self.lm_head_init_method:
                logger.warning("LM head will be reinitialized without protomaml (NOT RECOMMENDED)")

        logger.info(f"LM head retaining set to: {self.retain_lm_head}")
            
        # set flag to indicate whether we want to use the same or different N-way K-shot tasks 
        # during each training loop
        if isinstance(use_multiple_samples, str):
            self.use_multiple_samples = eval(use_multiple_samples)
        else: 
            self.use_multiple_samples = use_multiple_samples

    ###### Task head initialization methods ######

    def get_task_init_kwargs(
        self,
        task_type: str,
        task_init_method: str,
        n_labels: int,
        data_batch: Dict[str, torch.Tensor] = None,
        device: torch.device = None,
    ) -> Dict[str, Any]:
        """ 
        Helper method for generating keyword arguments that can be passed into a task head 
        initialization method
        
        Args: 
            * task_type: Type of task head to initialize (e.g. 'classification')
            * task_init_method: Method for initializing the task head
            * n_labels: Number of labels defined by the task (i.e. classes)
            * data_batch: Batch of data used to initialize the task head if using 
                the protomaml task_init_method
            * device: Device type used to initialize the task head with, if not 
                specified defaults to self.base_device

        Returns:
            * init_kwargs (dict): Keyword arguments used by the task head initialization function 
        """

        init_kwargs = {}

        init_kwargs['task_type'] = task_type  
        init_kwargs['task_init_method'] = task_init_method
        init_kwargs['n_labels'] = n_labels

        init_kwargs['base_model_hidden_dim'] = self.base_model_hidden_dim
        init_kwargs['device'] = device if device is not None else self.base_device 

        if 'protomaml' in task_init_method:
            assert(data_batch is not None),\
                "Use of protomaml as a classification head initializer requires a data_batch"
            init_kwargs['model'] = self.base_model
            init_kwargs['data_batch'] = data_batch

        return init_kwargs

    ###### Model training and evaluation helper methods ######

    @staticmethod
    def _compute_task_loss(
        model_outputs: torch.Tensor,
        data_batch: Dict[str, torch.Tensor],
        task_head_weights: torch.nn.ParameterDict,
        task_type: str,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Helper function for computing the task loss on a given batch of data. We assume that the 
        data has already been passed through the base_model - the result of which is model_outputs
        (i.e. the final layer's hidden states). 

        Args: 
            * model_outputs: Result of passing data_batch through the base_model. 
                Should have shape: (batch_size, sequence_length, hidden_size)
            * data_batch: Batch of data for a forward pass through the model 
                (see run_inner_loop for information on the data structure)
            * task_head_weights: Weights used by the task head (in this the classifier head)
            * task_type (str): Type of task (e.g. 'classification')
        Returns: 
            * logits: logits for the classification task
            * loss: loss for the classification task
        """

        #indexing into sequence layer of model_outputs -> (batch_size, hidden_size) 
        batch_size = model_outputs.size(0)
        last_hidden_state = model_outputs[torch.arange(batch_size),
                                          data_batch['input_target_idx']]

        if task_type == 'classification':
            head = ClassificationHead()
        else: 
            logger.exception(f"Invalid task type: {task_type}")
            raise Exception(f"Invalid task type: {task_type}")

        logits, loss = head(
            model_output=last_hidden_state,
            labels=data_batch['label_ids'],
            weights=task_head_weights
        )

        return (logits, loss)

    @property
    @abc.abstractmethod
    def outerloop_optimizer_param_groups(self) -> Iterator[Dict[str, torch.nn.Parameter]]:
        """
        Returns the parameter groups that are learnable during the outer loop. Note that 
        we might not have any inner loop parameters that need to be optimized (baseline model),
        thus only this method needs to be implemented.
        """
        raise NotImplementedError

    ###### Model training methods ######

    # def optimizer_step(self, set_zero_grad: bool = False) -> None:
    #     """ 
    #     Take a global update step of the meta learner params; optionally set the gradients of the 
    #     meta learner gradient tape back to zero.

    #     Args:
    #         * set_zero_grad (bool): Whether to set the gradients of the meta learner gradient tape
    #             back to zero after the optimizer step. Defaults to False.

    #     """
    #     assert(hasattr(self, 'optimizer')),\
    #         "Learner cannot take optimizer step - needs to define an optimizer attribute"

    #     self.optimizer.step()
    #     if set_zero_grad:
    #         self.optimizer.zero_grad()
    
    # def forward(self, learner, support_batch, query_batch, device):
    #     """ 
    #     NOTE: Only the DistributedDataParallel version of this model should indirectly call this.
    #           Used as a wrapper to run_inner_loop. 
    #           Unless you know what you're doing, don't call this method.
    #     """
    #     return learner.run_inner_loop(support_batch, query_batch, device)

    # def setup_DDP(self, rank: int, world_size: int) -> Tuple[torch.device, DDP]:
    #     """ 
    #     Helper method for setting up distributed data parallel process group and returning 
    #     a wrapper DDP instance of the learner
        
    #     Args: 
    #         * rank (int): Rank of current GPU 
    #         * world_size (int): Number of GPUs should be the same as utils.num_gpus

    #     Returns:
    #         * device (int): Device to run model on
    #         * ddp (torch.nn.parallel.DistributedDataParallel): Wrapped DDP learner
    #     """
    #     device = torch.device(f"cuda:{rank}")
    #     os.environ['MASTER_ADDR'] = 'localhost'
    #     os.environ['MASTER_PORT'] = '32432'
    #     dist.init_process_group("nccl", rank=rank, world_size=world_size)
    #     set_seed(self.seed)

    #     self.to(device)

    #     ddp = DDP(self, device_ids=[rank], find_unused_parameters=True)
    #     return (device, ddp)

    # def run_inner_loop_mp(
    #     self,
    #     rank: int,
    #     world_size: int,
    #     data_queue: Queue,
    #     loss_queue: Queue,
    #     step_optimizer: Event, 
    #     num_tasks_per_iteration: int,
    # ) -> None:
    #     """
    #     Entry point for running inner loop using multiple processes. Sets up DDP init process
    #     group, wraps learner in DDP and calls forward/backward on the DDP-wrapped model.

    #     Args: 
    #         * rank (int): Rank of current GPU 
    #         * world_size (int): Number of GPUs should be the same as utils.num_gpus
    #         * data_queue (multiprocessing.Queue): Queue from which we read passed in data
    #         * loss_queue (multiprocessing.Queue): Queue to which we write loss values
    #         * step_optimizer (multiprocessing.Event): Event to signal workers to take an optimizer
    #             step
    #         * num_tasks_per_iteration (int): Number of tasks per iteration that the user specifies
    #             in the experiment config file
    #     """

    #     device, ddp = self.setup_DDP(rank, world_size)

    #     while True: 
    #         # The main process sends signal to update optimizers

    #         while True: 
    #             # Waiting for the next batch of data 
    #             # NOTE: If there is no data either 1) the dataloading pipeline is taking a while 
    #             # or 2) the main process is waiting for all the workers to finish 
    #             try:
    #                 batch = data_queue.get(block=False)[0]
    #                 break
    #             except EmptyQueue: 
    #                 pass

    #             if step_optimizer.is_set():
    #                 # make sure all workers have taken an optimizer step
    #                 self.optimizer_step(set_zero_grad=True)
    #                 dist.barrier()

    #                 # once all workers have update params clear the flag to continue training
    #                 step_optimizer.clear()

    #             time.sleep(1) 

    #         task_name, support_batch, query_batch = batch

    #         task_loss = ddp(self, support_batch, query_batch, device)
    #         task_loss = task_loss/num_tasks_per_iteration
    #         task_loss.backward()

    #         loss_queue.put([task_loss.detach().item()])

    @abc.abstractmethod
    def run_inner_loop(
        self,
        support_batch_list: List[Dict[str, torch.Tensor]],
        query_batch: Dict[str, torch.Tensor],
        device: torch.device = None, 
    ) -> torch.Tensor:
        """ 
        Run an inner loop optimization step (in the context of meta learning); assumes 
        that the class contains the model that is to-be meta-learned.

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
        raise NotImplementedError()

    # ###### Model evaluation methods ######


class BaseMetaLearner(BaseLearner):

    def __init__(
        self,
        initial_inner_lr: Union[float, str] = 1e-2,
        initial_classifier_lr: Union[float, str] = 1e-1,
        num_inner_loop_steps: Union[int, str] = 7,
        use_first_order: Union[bool, str] = False,
        **kwargs
    ) -> None:
        """ 
        Inherents from BaseLearner and establishes the base class for all meta-learners 
        (e.g. maml). Provides useful functionality for meta-learners which rely on the torch
        higher library to functionalize the model 

        NOTE! The base_model we are meta learning needs to contain parameter names 
        that specify what layer the parameter is in (e.g. 'attention.layer.1'). This is because 
        we store per-layer parameter weights.

        Args:
            * initial_inner_lr: Initial inner-loop learning rate of the base_model - this value is 
                learned over the course of meta-learning 
            * initial_classifier_lr: Initial inner-loop learning rate of the classifier head - this value
                is learned over the course of meta-learning 
            * num_inner_loop_steps: Number of gradients steps in the inner loop used to learn the
                meta-learning task
            * use_first_order: Whether a first order approximation of higher-order gradients
                should be used                
        """
        super().__init__(**kwargs)

        self.inner_layers_lr = {
            layer: torch.nn.Parameter(torch.tensor(float(initial_inner_lr)).to(self.base_device)) 
                for layer in self.base_model.trainable_layers
        }   
        
        # NOTE: The final classification layer also gets its own learning rate 
        self.classifier_lr = torch.nn.Parameter(
            torch.tensor(float(initial_classifier_lr)).to(self.base_device)
        )
        
        if len(self.inner_layers_lr) == 0:
            # NOTE: we are not storing any learning rates for any of the layers.
            logger.error(
            """
            Could not specify per-layer learning rates. Ensure that the model parameters that are 
            in the same layer contain the string 'layer.[number]' as part of their name.
            """
            )

        # number of steps to perform in the inner loop
        self.num_inner_loop_steps = int(num_inner_loop_steps)
        
        # set flag to indicate if first-order approximation should be used (à la Reptile)
        if isinstance(use_first_order, str):
            self.use_first_order = eval(use_first_order)
        else: 
            self.use_first_order = use_first_order

    ### Base setup functionality for meta learning models

    def innerloop_optimizer_param_groups(
        self,
        base_model_override: torch.nn.Module = None, 
        cast_lr_to_float: bool = False
    ) -> Iterator[Dict[str, torch.nn.Parameter]]: 
        """
        Returns the parameter groups that are passed to the innerloop (diferentiable) optimizer.

        Args: 
            * base_model_override: Optional torch.nn.Module to override the base_model stored in 
                the class. This is useful if we want to use a different model for the inner loop
                than the one we are meta-learning; for instance if we want to copy the base model 
                and train it on a specific downstream NLU task. 
        """

        if base_model_override is None:
            base_model = self.base_model
        else:
            base_model = base_model_override

        def extract_layer_num(layer_name): 
            layer = re.findall(r"layer.\d*", layer_name)

            if len(layer) == 1:
                # layer will return a list of strings of the form: ["layer.[number]"]
                try: 
                    return int(layer[0][6:])
                except:
                    return None
            else: 
                return None

        param_groups = []

        for idx, (name, param) in enumerate(base_model.named_parameters()): 
            if not param.requires_grad:
                continue

            layer_num = extract_layer_num(name)
            if layer_num is None: 
                raise Exception(
                    "Could not find an innerloop learning rate for param: {}".format(name)
                )

            layer_lr = self.inner_layers_lr[layer_num]

            param_groups.append({
                'params': param,
                'lr': layer_lr.item() if cast_lr_to_float else layer_lr,
            })

        return param_groups
            
    def outerloop_optimizer_param_groups(self) -> Iterator[Dict[str, torch.nn.Parameter]]:
        """
        Returns the parameter groups that are passed to the outerloop optimizer.
        """
        
        param_groups = [
            {
                'params': [p for p in self.base_model.parameters() if p.requires_grad],
            }, 
            {
                'params': self.inner_layers_lr.values(),
            },
            {
                'params': [self.classifier_lr],
            }
        ]

        if self.retain_lm_head:
            params_groups.append(
                {
                    'params': self.retained_lm_head_weights.values()
                }
            )

        return param_groups
