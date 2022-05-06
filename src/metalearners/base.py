__author__ = 'Richard Diehl Martinez'
""" Interface class for (meta) learners """

import abc 
import higher 
import math
import os
import logging

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from ..taskheads import TaskHead, ClassificationHead

logger = logging.getLogger(__name__)

class BaseLearner(torch.nn.Module, metaclass=abc.ABCMeta):

    def __init__(self, base_model, base_device, lm_head_init_method='protomaml', lm_head_n=100,
                 retain_lm_head=False, **kwargs):
        """
        BaseLearner establishes the inferface for the learner class and 
        inherents from torch.nn.Module. 
        
        Args: 
            * base_model (implements a BaseModel)
            * base_device (str): what device to use for training (either 'cpu' or 'gpu) 
            * language_head_init_method (str): How to initialize the language head classifier layer
            * lm_head_n (int): Size of n-way classification used for generating the language 
                modeling tasks used for training.
            * retain_lm_head (bool): Indicate whether we should maintain a single task head 
                that is learned over the course of meta training, or whether for each task we 
                should initialie a new task head.

        """
        super().__init__()

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
            init_kwargs = self.get_task_init_kwargs(lm_head_init_method, self.lm_head_n)
            self.retained_lm_head = TaskHead.initialize_task_head(task_type='classification',
                                                                  method=lm_head_init_method,
                                                                  init_kwargs=init_kwargs)

        else: 
            # If we are re-initializing the LM head for each training task, then we should use 
            # protomaml (but it is still possible to use a random initialization)
            if lm_head_init_method != "protomaml": 
                logger.warning("LM head will be reinitialized without protomaml (NOT RECOMMENDED)")

        logger.info(f"LM head retaining set to: {self.retain_lm_head}")

    ###### Task head initialization methods ######

    def get_task_init_kwargs(self, task_init_method, n_labels, data_batch=None, device=None,
                             **kwargs):
        """ 
        Helper method for generating keyword arguments that can be passed into a task head 
        initialization method
        
        Args: 
            * task_init_method (str): Method for initializing the task head
            * n_labels (int): Number of labels defined by the task (i.e. classes)
            * data_batch (dict): Batch of data used to initialize the task head if using 
                the protomaml task_init_method
            * device (str): Device type used to initialize the task head with, if not 
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
    def _compute_task_loss(model_outputs, data_batch, task_head_weights, task_type):
        """
        Helper function for computing the task loss on a given batch of data. We assume that the 
        data has already been passed through the base_model - the result of which is model_outputs
        (i.e. the final layer's hidden states). 

        Args: 
            * model_outputs (torch.Tensor): Result of passing data_batch through the 
                base_model. Should have shape: (batch_size, sequence_length, hidden_size)
            * data_batch (dict): Batch of data for a forward pass through the model 
                (see run_inner_loop for information on the data structure)
            * task_head_weights (dict): weights used by the task head (in this the classifier head)
            * task_type (str): Type of task (e.g. 'classification')
        Returns: 
            * logits ([torch.Tensor]): Logits resulting from forward pass 
            * loss (int): Loss of data 
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

        logits, loss = head(model_output=last_hidden_state, labels=data_batch['label_ids'],
                            weights=task_head_weights)

        return (logits, loss)

    ###### Model training methods ######

    def optimizer_step(self, set_zero_grad=False):
        """ 
        Take a global update step of the meta learner params; optionally set the gradients of the 
        meta learner gradient tape back to zero.
        """
        assert(hasattr(self, 'optimizer')),\
            "Learner cannot take optimizer step - needs to define an optimizer attribute"

        self.optimizer.step()
        if set_zero_grad:
            self.optimizer.zero_grad()
    
    def forward(self, learner, support_batch, query_batch, device):
        """ 
        NOTE: Only torch.DistributedDataParallel should indirectly call this - used as a wrapper to 
              run_inner_loop. Unless you know what you're doing, don't call this method.
        """
        return learner.run_inner_loop(support_batch, query_batch, device)

    def setup_DDP(self, rank, world_size):
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
        device = f"cuda:{rank}"
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '32432'
        dist.init_process_group("nccl", rank=rank, world_size=world_size)

        self.to(device)

        ddp = DDP(self, device_ids=[rank], find_unused_parameters=True)
        return (device, ddp)

    @abc.abstractmethod
    def run_inner_loop_mp(self, rank, world_size, data_queue, loss_queue, step_optimizer,
                          num_tasks_per_iteration):
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
        raise NotImplementedError()

    @abc.abstractmethod
    def run_inner_loop(self, support_batch, query_batch=None, device=None, **kwargs): 
        """ 
        Run an inner loop optimization step (in the context of meta learning); assumes 
        that the class contains the model that is to-be meta-learned.

        Args:
            * support_batch: A dictionary containing the following information for the support set
                * input_ids (torch.tensor): Input tensors of shape (N*K, max_seq_len)
                * input_target_idx (torch.tensor): Tensor indicating for each sample at what index
                    we apply the final classification layer 
                * label_ids (torch.tensor): Tensor of labels corresponding to masked out subword id
                * attention_mask (torch.tensor): Tensor indicating which tokens in input_ids are
                    not pad tokens
            * query_batch [optional]: Same as support_batch, but for the data of the query set.
                This argument might be optional depending on how the learner is implemented.
            * device: Optional string to specify a device to override base_device

        Returns: 
            * loss (torch.tensor): A tensor containing the loss that results from the inner loop 
        """
        raise NotImplementedError()

    ###### Model evaluation methods ######

    @abc.abstractmethod
    def run_finetuning(self, task_type, finetune_dataloader, *args, **kwargs):
        """
        Finetunes the model on the data of finetune_dataloader.

        Args:
            * task_type (str): Type of task to finetune on (e.g. classification)
            * finetune_dataloader (torch.data.Dataloader): The dataset for finetuning the model is
                passed in as a dataloader (in most cases this will be an NLUDataloader)

        Returns:
            * inference_params ({}): Returns any sort of params that need to be used for the 
                run_inference_classification method 
        """ 
        raise NotImplementedError()

    @abc.abstractmethod
    def run_inference(self, task_type, inference_dataloader, *args, **kwargs):
        """
        Evaluates the model on the data of inference_dataloader. This should only be called once 
        run_finetuning has been run.

        Args: 
            * task_type (str): Type of task to evaluate on; should be the same as the task type 
                passed into run_finetuning
            * inference_dataloader (torch.data.Dataloader): The dataset for inference is passed
                in as a dataloader (in most cases this will be an NLUDataloader)

        Returns: 
            * predictions ([int]): A dictionary storing the model's predictions for each 
                datapoint passed in from the inference_dataloader as an int. 
            * loss (int): The value of the classification loss on the inference dataset.
        """
        raise NotImplementedError()


class MetaBaseLearner(BaseLearner):

    def __init__(self, *args, **kwargs):
        """ 
        Inherents from BaseLearner and establishes the base class for all meta-learners 
        (e.g. maml and platipus). Provides useful functionality for meta-learners which
        rely on the torch higher library to maintain a functionalized version of the model 
        with corresponding parameters that are clones and fed into the functionalized model.
        """
        super().__init__(*args, **kwargs)

    ### Base setup functionality for meta learning models

    def functionalize_model(self):
        """ Helper function for converting base_model into a functionalized form"""
        self.functional_model = higher.patch.make_functional(module=self.base_model)

        # NOTE: these two lines are SUPER important (otherwise computation graph explodes)
        self.functional_model.track_higher_grads = False
        self.functional_model._fast_params = [[]]

    def parameters(self):
        """ Overriding parent behavior to only return meta parameters """
        return self.meta_params_iter()

    @abc.abstractmethod
    def meta_params_iter(self):
        """ Returns an iterator over all of the meta parameters"""
        raise NotImplementedError()

    # Overriding nn.Module functionality 
    def state_dict(self):
        """
        Overriding method to remove placeholder parameters defined by functional model. Called 
        implicitly when saving and loading checkpoints.
        """
        original_state_dict = super().state_dict()
        updated_state_dict = OrderedDict()

        for key, val in original_state_dict.items():
            if "functional" in key:
                continue
            
            updated_state_dict[key] = val
        
        return updated_state_dict

    ### Helper function for adapting the functionalized parameters based on some data_batch
    def _adapt_params(self, data_batch, params, lm_head_weights, learning_rate, num_inner_steps,
                            optimize_classifier=False, clone_params=True, evaluation_mode=False,
                     ):
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
            * data_batch (dict): Batch of data for a forward pass through the model 
                (see run_inner_loop for information on the data structure)
            * params (iterable): Iterable of torch.nn.Paramater()s
                (the params that we want to evaluate our model at)
            * lm_head_weights (dict): Weights of the lm classification head
            * learning_rate (float): The internal learning rate used to update params
            * num_inner_steps (int): Number of inner steps to use for the adaptation process
            * optimize_classifier (bool): Whether to train the final classification layer as 
                part of the adaptation process 
            * clone_params (bool): Whether to clone the params passed in (defaults to True)
            * evaluation_mode (bool): Whether running this method during evaluation
                (either in finetuning or inference) (defaults to False)

        Returns: 
            * adapted_params (iterable): Iterable of torch.nn.Paramater()s that represent the
                updated the parameters after running SGD 
        """                                        

        if clone_params:
            # copy the parameters but allow gradients to propagate back to original params
            adapted_params = [torch.clone(p) for p in params]
        else:
            adapted_params = params

        for _ in range(num_inner_steps):
            
            # Running forward pass through functioanl model and computing classificaiton loss
            outputs = self.functional_model.forward(input_ids=data_batch['input_ids'],
                                                    attention_mask=data_batch['attention_mask'],
                                                    params=adapted_params)

            _, loss = self._compute_task_loss(outputs, data_batch, lm_head_weights,
                                              task_type='classification')
                                                        
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
            for i, p in enumerate(adapted_params):
                if p.requires_grad:
                    adapted_params[i] = adapted_params[i] - learning_rate * grads[grad_counter]
                    grad_counter += 1

            # optionally use SGD to update the weights of the final linear classification layer
            # should only be done once we've already sampled phi weights
            if optimize_classifier:
                classifier_grads = torch.autograd.grad(
                    outputs=loss,
                    inputs=lm_head_weights.values(),
                )
                for idx, weight_name in enumerate(lm_head_weights.keys()):
                    lm_head_weights[weight_name] = lm_head_weights[weight_name] - \
                                                    self.classifier_lr * classifier_grads[idx]

        return adapted_params
