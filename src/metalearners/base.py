__author__ = 'Richard Diehl Martinez'
""" Interface class for (meta) learners """

import abc 
import higher
import logging
import math
import os
import re 

from collections import OrderedDict
from typing import Union

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from ..taskheads import TaskHead, ClassificationHead

from ..utils import set_seed

# imports for typing 
from ..models import BaseModel
from typing import Tuple, List, Dict, Union
from collections.abc import Iterator
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

        # hidden dimensions of the outputs of the base_model
        self.base_model = base_model
        self.base_model.to(base_device)
        self.base_model_hidden_dim = base_model.hidden_dim 

        self.base_device = base_device

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
            self.retained_lm_head = TaskHead.initialize_task_head(**init_kwargs)

        else: 
            # If we are re-initializing the LM head for each training task, then we should use 
            # protomaml (but it is still possible to use a random initialization)
            if lm_head_init_method != "protomaml": 
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

        init_kwargs['base_model_hidden_dim'] = self.base_model_hidden_dim
        init_kwargs['n_labels'] = n_labels
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

    ###### Model training methods ######

    def optimizer_step(self, set_zero_grad: bool = False) -> None:
        """ 
        Take a global update step of the meta learner params; optionally set the gradients of the 
        meta learner gradient tape back to zero.

        Args:
            * set_zero_grad (bool): Whether to set the gradients of the meta learner gradient tape
                back to zero after the optimizer step. Defaults to False.

        """
        assert(hasattr(self, 'optimizer')),\
            "Learner cannot take optimizer step - needs to define an optimizer attribute"

        self.optimizer.step()
        if set_zero_grad:
            self.optimizer.zero_grad()
    
    def forward(self, learner, support_batch, query_batch, device):
        """ 
        NOTE: Only the DistributedDataParallel version of this model should indirectly call this.
              Used as a wrapper to run_inner_loop. 
              Unless you know what you're doing, don't call this method.
        """
        return learner.run_inner_loop(support_batch, query_batch, device)

    def setup_DDP(
        self,
        rank: int,
        world_size: int
    ) -> Tuple[torch.device, torch.nn.parallel.DistributedDataParallel]:
        """ 
        Helper method for setting up distributed data parallel process group and returning 
        a wrapper DDP instance of the learner
        
        Args: 
            * rank (int): Rank of current GPU 
            * world_size (int): Number of GPUs should be the same as utils.num_gpus

        Returns:
            * device (int): Device to run model on
            * ddp (torch.DistributedDataParallel): Wrapped DDP learner
        """
        device = torch.device(f"cuda:{rank}")
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '32432'
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        set_seed(self.seed)

        self.to(device)

        ddp = DDP(self, device_ids=[rank], find_unused_parameters=True)
        return (device, ddp)

   def run_inner_loop_mp(
        self,
        rank: int,
        world_size: int,
        data_queue: Queue,
        loss_queue: Queue,
        step_optimizer: Event, 
        num_tasks_per_iteration: int,
    ) -> None:
        """
        Entry point for running inner loop using multiple processes. Sets up DDP init process
        group, wraps learner in DDP and calls forward/backward on the DDP-wrapped model.

        Args: 
            * rank (int): Rank of current GPU 
            * world_size (int): Number of GPUs should be the same as utils.num_gpus
            * data_queue (multiprocessing.Queue): Queue from which we read passed in data
            * loss_queue (multiprocessing.Queue): Queue to which we write loss values
            * step_optimizer (multiprocessing.Event): Event to signal workers to take an optimizer
                step
            * num_tasks_per_iteration (int): Number of tasks per iteration that the user specifies
                in the experiment config file
        """

        device, ddp = self.setup_DDP(rank, world_size)

        while True: 
            # The main process sends signal to update optimizers

            while True: 
                # Waiting for the next batch of data 
                # NOTE: If there is no data either 1) the dataloading pipeline is taking a while 
                # or 2) the main process is waiting for all the workers to finish 
                try:
                    batch = data_queue.get(block=False)[0]
                    break
                except EmptyQueue: 
                    pass

                if step_optimizer.is_set():
                    # make sure all workers have taken an optimizer step
                    self.optimizer_step(set_zero_grad=True)
                    dist.barrier()

                    # once all workers have update params clear the flag to continue training
                    step_optimizer.clear()

                time.sleep(1) 

            task_name, support_batch, query_batch = batch

            task_loss = ddp(self, support_batch, query_batch, device)
            task_loss = task_loss/num_tasks_per_iteration
            task_loss.backward()

            loss_queue.put([task_loss.detach().item()])

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

    ###### Model evaluation methods ######

    @abc.abstractmethod
    def run_finetuning(
        self,
        support_batch: Dict[str, torch.Tensor],
        task_type: str,
        n_labels: int,
    ) -> Dict[str, Any]:
        """
        Finetunes the model on a given support_batch for a model of type task_type with n_labels.

        Args:
            * support_batch: A batch of task data represented as a dictionary containing the
                data for the support set of a task to finetune on.
            * task_type: The type of task to finetune on.
            * n_labels: The number of labels for the task to finetune on.

        Returns:
            * inference_params: Returns any sort of params that need to be used for the 
                run_inference_classification method 
        """ 
        raise NotImplementedError()

    @abc.abstractmethod
    def run_inference(
        self,
        inference_dataloader: torch.utils.data.DataLoader,
        task_type: str,
        **inference_params: Dict[str, Any],
    ) -> Tuple[List[int], int]:
        """
        Evaluates the model on the data of inference_dataloader which corresponds to a task of type 
        task_type. This should only be called once run_finetuning has been run.

        Args: 
            * inference_dataloader: The dataset on which to perform model inference on.
            * task_type: Type of task to evaluate on; should be the same as the task type 
                passed into run_finetuning
            * inference_params: Any params that need to be used for the inference task

        Returns: 
            * predictions ([int]): A dictionary storing the model's predictions for each 
                datapoint passed in from the inference_dataloader as an int. 
            * loss (int): The value of the classification loss on the inference dataset.
        """
        raise NotImplementedError()


class MetaBaseLearner(BaseLearner):

    def __init__(
        self,
        base_model: BaseModel,
        optimizer_type: str = 'adam',
        meta_lr: Union[float, str] = 1e-2,
        inner_lr: Union[float, str] = 1e-2,
        classifier_lr: Union[float, str] = 1e-1,
        num_learning_steps: Union[int, int] = 5,
        use_first_order: Union[bool, str] = False,
        **kwargs
    ) -> None:
        """ 
        Inherents from BaseLearner and establishes the base class for all meta-learners 
        (e.g. maml and platipus). Provides useful functionality for meta-learners which
        rely on the torch higher library to maintain a functionalized version of the model 
        with corresponding parameters that are clones and fed into the functionalized model.

        NOTE! The base_model we are meta learning needs to contain parameter names 
        that specify what layer the parameter is in (e.g. 'attention.layer.1'). This is because 
        we store per-layer parameter weights.

        Args:
            * base_model: The model to be meta-learned (implementation of BaseModel)
            * optimizer_type: The type of optimizer (e.g. 'adam') 
            * meta_lr: Learning rate for the outer loop meta learning step
            * inner_lr: Initial inner-loop learning rate of the base_model - this value is 
                learned over the course of meta-learning 
            * classifier_lr: Initial inner-loop learning rate of the classifier head - this value
                is learned over the course of meta-learning 
            * num_learning_steps: Number of gradients steps in the inner loop used to learn the
                meta-learning task
            * use_first_order: Whether a first order approximation of higher-order gradients
                should be used                
        """
        super().__init__(base_model, **kwargs)

        # Setting up meta optimization parameters
        self.optimizer_type = optimizer_type
        self.meta_lr = float(meta_lr)

        # Deducing the num of layers and mapping each parameter to the layer that parameter is in
        self.param_idx_to_layer = {}
        for idx, (name, _) in enumerate(base_model.named_parameters()): 
            layer = re.findall(r"layer.\d*", name)

            if len(layer) > 0:
                # layer will return a list of strings of the form: ["layer.[number]"]
                self.param_idx_to_layer[idx] = int(layer[0][6:])
            else: 
                self.param_idx_to_layer[idx] = None
        
        layers = set(self.param_idx_to_layer.values())       

        # NOTE: Each layer gets its own learning rate 
        # We store a list of learning rates for each layer of the model that are meta-learned 
        # and we initialize each of these to inner_lr
        self.inner_layers_lr = torch.nn.ParameterList([
                                        torch.nn.Parameter(
                                            torch.tensor(float(inner_lr)).to(self.base_device)
                                        )
                                        for layer in layers if layer is not None
                                    ])

        # NOTE: The final classification layer also gets its own learning rate 
        self.classifier_lr = torch.nn.Parameter(torch.tensor(float(classifier_lr)).\
                                                    to(self.base_device))
        
        if len(self.inner_layers_lr) == 0:
            # NOTE: we are not storing any learning rates for any of the layers.
            logger.error(
            """
            Could not specify per-layer learning rates. Ensure that the model parameters that are 
            in the same layer contain the string 'layer.[number]' as part of their name.
            """
            )

        # number of steps to perform in the inner loop
        self.num_learning_steps = int(num_learning_steps)
        
        # set flag to indicate if first-order approximation should be used (à la Reptile)
        if isinstance(use_first_order, str):
            self.use_first_order = eval(use_first_order)
        else: 
            self.use_first_order = use_first_order

    ### Base setup functionality for meta learning models

    def setup_optimizer(self) -> None:
        """ Helper function for setting up the (meta) optimizer """
        if self.optimizer_type == 'adam': 
            self.optimizer = torch.optim.Adam(params=self.meta_params_iter(), lr=self.meta_lr)
        else:
            logger.exception(f"Invalid optimizer type: {optimizer_type}")
            raise Exception(f"Invalid optimizer type: {optimizer_type}")

    def functionalize_model(self) -> None:
        """
        Helper function for converting base_model into a functionalized form
        """
        self.functional_model = higher.patch.make_functional(module=self.base_model)

        # NOTE: these two lines are SUPER important (otherwise computation graph explodes)w
        self.functional_model.track_higher_grads = False
        self.functional_model._fast_params = [[]]

    def parameters(self) -> List[torch.Tensor]:
        """ 
        Overriding parent behavior to only return meta parameters
        """
        return self.meta_params_iter()

    @abc.abstractmethod
    def meta_params_iter(self) -> Iterator[torch.Tensor]:
        """ 
        Returns an iterator over the meta parameters of the model. 
        """
        raise NotImplementedError()

    # Overriding nn.Module functionality 
    def state_dict(self) -> Dict[str, torch.Tensor]:
        """
        Overriding method to remove placeholder parameters defined by functional model. Called 
        implicitly when saving and loading checkpoints.

        Returns: 
            * updated_state_dict (Dict[str, torch.Tensor]): Dictionary of model parameters
        """
        original_state_dict = super().state_dict()
        updated_state_dict = OrderedDict()

        for key, val in original_state_dict.items():
            if "functional" in key:
                continue
            
            updated_state_dict[key] = val
        
        return updated_state_dict

    def setup_DDP( self, **kwargs) -> Tuple[torch.device, torch.DistributedDataParallel]:
        """ 
        Overriding parent behavior to ensure that the functionalized model is also wrapped in
        DDP.

        Returns:
            * device (int): Device to run model on
            * ddp (torch.DistributedDataParallel): Wrapped DDP learner
        """
        device, ddp = super().setup_DDP(**kwargs)
        self.functionalize_model()
        return (device, ddp)

    ### Helper function for adapting the functionalized parameters based on some data_batch
    def _adapt_params(
        self,
        data_batch_list: List[Dict[str, torch.Tensor]],
        params: List[torch.Tensor],
        task_head_weights: Dict[str, torch.Tensor],
        learning_rate: Union[torch.nn.Parameter, torch.nn.ParameterList],
        num_inner_steps: int,
        optimize_classifier: bool = False,
        clone_params: bool = True,
        evaluation_mode: bool = False,
    ) -> List[torch.Tensor]:
        """ 
        Adapted from: 
        https://github.com/cnguyen10/few_shot_meta_learning

        For a given batch of inputs and labels, we compute what the loss of the functional model 
        would be if the weights of the model are set to params.  
        
        Note that this is the main reason that we have to convert the base model to a functional
        version of the model that can take in its parameters as an argument
        (as opposed to these parameters being stored in the model's state). 

        The parameters are then updated using SGD with a given learning rate, and returned. 

        Params:
            * data_batch_list (dict): Batches of data that are used for training the model in the  
                inner loop (see run_inner_loop for information on the data structure)
            * params (iterable): Iterable of torch.nn.Parameter()s
                (the params that we want to evaluate our model at)
            * task_head_weights: Weights of the model head
            * learning_rate (torch.nn.Parameter or torch.nn.ParameterList): The learning rate 
                used to update the model parameters. Will be a ParameterList if we are storing
                per-layer specific learning rates. 
            * num_inner_steps: Number of inner steps to use for the adaptation process
            * optimize_classifier: Whether to train the final classification layer as 
                part of the adaptation process 
            * clone_params: Whether to clone the params passed in (defaults to True)
            * evaluation_mode: Whether running this method during evaluation
                (either in finetuning or inference) (defaults to False)

        Returns: 
            * adapted_params: List of torch.nn.Parameter()s that represent the updated 
                parameters after running SGD 
        """                                        

        if clone_params:
            # copy the parameters but allow gradients to propagate back to original params
            adapted_params = [torch.clone(p) for p in params]
        else:
            adapted_params = params

        for num_step in range(num_inner_steps):
            
            if self.use_multiple_samples and not evaluation_mode: 
                data_batch = data_batch_list[num_step]
            else:
                data_batch = data_batch_list[0]
            
            # Running forward pass through functional model and computing classificaiton loss
            outputs = self.functional_model.forward(
                input_ids=data_batch['input_ids'],
                attention_mask=data_batch['attention_mask'],
                params=adapted_params
            )
            _, loss = self._compute_task_loss(
                outputs,
                data_batch,
                task_head_weights,
                task_type='classification'
            )

            # Computing resulting gradients of the inner loop
            grad_params = [p for p in adapted_params if p.requires_grad]

            if self.use_first_order or evaluation_mode:
                grads = torch.autograd.grad(
                    outputs=loss,
                    inputs=grad_params,
                    retain_graph=True,
                )
            else:
                grads = torch.autograd.grad(
                    outputs=loss,
                    inputs=grad_params,
                    create_graph=True,
                )

            # updating params
            grad_counter = 0
            for idx, p in enumerate(adapted_params):
                
                if isinstance(learning_rate, torch.nn.ParameterList):
                    # Extracting the layer of the parameter
                    p_layer = self.param_idx_to_layer[idx]
                    if p_layer is None: 
                        # Param is not in a layer we train (e.g. a pooling or embedding layer)
                        # so the param should not have requires_grad set 
                        assert(not p.requires_grad),\
                            f"Model param at idx: {idx} requires grad but has no learning rate"
                        continue
                    param_lr = learning_rate[p_layer]
                else: 
                    param_lr = learning_rate

                if p.requires_grad:
                    if param_lr is None: 
                        logger.exception(f"Inner learning rate for param at idx: {idx} is None")
                        raise Exception(f"Inner learning rate for param at idx: {idx} is None")

                    adapted_params[idx] = adapted_params[idx] - param_lr * grads[grad_counter]
                    grad_counter += 1

            # optionally use SGD to update the weights of the final linear classification layer
            # should only be done once we've already sampled phi weights
            if optimize_classifier:
                classifier_grads = torch.autograd.grad(
                    outputs=loss,
                    inputs=task_head_weights.values(),
                )
                for idx, weight_name in enumerate(task_head_weights.keys()):
                    task_head_weights[weight_name] = task_head_weights[weight_name] - \
                                                    self.classifier_lr * classifier_grads[idx]

        return adapted_params
