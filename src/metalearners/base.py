__author__ = "Richard Diehl Martinez"
""" Interface class for (meta) learners """

import abc
import logging
import os
import re
import torch

from ..taskheads import TaskHead, ClassificationHead, QAHead

# imports for typing
from ..models import BaseModel
from typing import Tuple, List, Dict, Union, Any, Iterator

logger = logging.getLogger(__name__)


class BaseLearner(torch.nn.Module, metaclass=abc.ABCMeta):
    def __init__(
        self,
        base_model: BaseModel,
        base_device: torch.device,
        seed: int,
        lm_head_init_method: str = "protomaml",
        lm_head_n: Union[int, str] = 10,
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
            # the task head with protomaml (should just be random initialization)
            assert (
                "random" in self.lm_head_init_method
            ), "retain_task_head can only be set to True if lm_head_init_method is 'random'"
            init_kwargs = self.get_lm_head_init_kwargs()
            self.retained_lm_head_weights = TaskHead.initialize_task_head(**init_kwargs)
        else:
            # If we are re-initializing the LM head for each training task, then we should use
            # protomaml
            if "protomaml" not in self.lm_head_init_method:
                logger.warning(
                    "LM head will be reinitialized without protomaml (NOT RECOMMENDED)"
                )

        # set flag to indicate whether we want to use the same or different N-way K-shot tasks
        # during each training loop
        if isinstance(use_multiple_samples, str):
            self.use_multiple_samples = eval(use_multiple_samples)
        else:
            self.use_multiple_samples = use_multiple_samples

    ###### LM Head initialization method ######

    def get_lm_head_init_kwargs(
        self,
        lm_head_init_method: str = None,
        data_batch: Dict[str, torch.Tensor] = None,
        device: torch.device = None,
    ) -> Dict[str, Any]:
        """
        Helper method for generating keyword arguments that can be passed into a task head
        initialization method to generate a language modeling task head.
        NOTE: Should really only be called during training

        Args:
            * lm_head_init_method: Method for initializing the lm head head
            * data_batch: Batch of data used to initialize the lm head head if using
                the protomaml task_init_method
            * device: Device type used to initialize the task head with, if not
                specified defaults to self.base_device

        Returns:
            * init_kwargs (dict): Keyword arguments used by the task head initialization function
        """

        init_kwargs = {}

        lm_head_init_method = (
            lm_head_init_method
            if lm_head_init_method is not None
            else self.lm_head_init_method
        )

        init_kwargs["task_type"] = "classification"
        init_kwargs["task_init_method"] = lm_head_init_method
        init_kwargs["n_labels"] = self.lm_head_n

        init_kwargs["base_model_hidden_dim"] = self.base_model_hidden_dim
        init_kwargs["device"] = device if device is not None else self.base_device

        if "protomaml" in lm_head_init_method:
            assert (
                data_batch is not None
            ), "Use of protomaml as a classification head initializer requires a data_batch"
            init_kwargs["model"] = self.base_model
            init_kwargs["data_batch"] = data_batch

        return init_kwargs

    ###### Model training and evaluation helper methods ######

    @staticmethod
    def _compute_task_loss(
        model_outputs: torch.Tensor,
        data_batch: Dict[str, torch.Tensor],
        task_head_weights: torch.nn.ParameterDict,
        task_type: str,
    ) -> Tuple[Union[torch.Tensor, List[torch.Tensor]], torch.Tensor]:
        """
        Helper function for computing the task loss on a given batch of data. We assume that the
        data has already been passed through the base_model - the result of which is model_outputs
        (i.e. the final layer's hidden states).

        Args:
            * model_outputs: Result of passing data_batch through the base_model.
                Should have shape: (batch_size, sequence_length, hidden_size)
            * data_batch: Batch of data for a forward pass through the model
                (see run_innerloop for information on the data structure)
            * task_head_weights: Weights used by the task head (in this the classifier head)
            * task_type (str): Type of task (e.g. 'classification')
        Returns:
            * logits: logits for the classification task - can either be a list of tensors
                or a single tensor depending on the task type and the corresponding task head
            * loss: loss for the classification task
        """

        if task_type == "classification":
            head = ClassificationHead()
        elif task_type == "qa":
            head = QAHead()
        else:
            logger.exception(f"Invalid task type: {task_type}")
            raise Exception(f"Invalid task type: {task_type}")

        logits, loss = head(
            model_outputs=model_outputs,
            data_batch=data_batch,
            weights=task_head_weights
        )

        return (logits, loss)

    def outerloop_optimizer_param_groups(
        self,
    ) -> Iterator[Dict[str, torch.nn.Parameter]]:
        """
        Returns the parameter groups that are learnable during the outer loop - i.e. the global
        parameters of the model (this is in contrast to the inner loop parameteres).
        Note, however, that we might not have any inner loop parameters that need to be optimized
        (such as in the case of the baseline model), thus only this method needs to be implemented.

        Returns:
            * param_groups (list): List of parameter groups that are learnable during the
                outer loop
        """
        param_groups = [
            {
                "params": [p for p in self.base_model.parameters() if p.requires_grad],
            },
        ]

        if self.retain_lm_head:
            param_groups.append({"params": self.retained_lm_head_weights.values()})

        return param_groups

    @staticmethod
    def finetune_optimizer_param_groups(
        base_model: torch.nn.Module,
        task_head_weights: Dict[str, torch.nn.Parameter],
        add_decay_information: bool = True,
        weight_decay_val: float = 0.0,
    ) -> Iterator[Dict[str, torch.nn.Parameter]]:
        """
        Iterator that returns the parameters of the passed in base_model and the task head weights

        Args:
            * base_model: The base model that we are extracting parameters from
            * task_head_weights: The weights of the task head that we are extracting parameters
                from
            * add_decay_information: Whether to add weight decay information to the parameters
            * weight_decay_val: The weight decay value to use for the parameters
        Returns:
            * param_group: A list of dictionaries containing the parameters of the base_model and
                the task head weights
        """

        no_decay = ["bias", "LayerNorm.weight"]
        param_groups = []

        for param_name, param in base_model.named_parameters():
            if not param.requires_grad:
                continue

            param_group = {
                "params": param,
            }

            if add_decay_information:
                if any(nd in param_name for nd in no_decay):
                    param_group["weight_decay"] = 0.0
                else:
                    param_group["weight_decay"] = weight_decay_val

            param_groups.append(param_group)

        param_groups.append({"params": task_head_weights.values(), "weight_decay": 0.0})

        return param_groups

    @abc.abstractmethod
    def run_train_loop(
        self,
        support_batch_list: List[Dict[str, torch.Tensor]],
        query_batch: Dict[str, torch.Tensor],
        device: torch.device = None,
    ) -> torch.Tensor:
        """
        Trains a model to perform a language modeling task by giving it access to the data in the
        support_batch_list and query_batch. In the context of meta learning, this training loop
        is the inner loop optimization step.

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

class BaseMetaLearner(BaseLearner):
    def __init__(
        self,
        initial_base_model_lr: Union[float, str] = 1e-2,
        initial_classifier_lr: Union[float, str] = 1e-1,
        num_innerloop_steps: Union[int, str] = 7,
        use_first_order: Union[bool, str] = True,
        **kwargs,
    ) -> None:
        """
        Inherents from BaseLearner and establishes the base class for all meta-learners
        (e.g. maml). Provides useful functionality for meta-learners which rely on the torch
        higher library to functionalize the model

        NOTE! The base_model we are meta learning needs to contain parameter names
        that specify what layer the parameter is in (e.g. 'attention.layer.1'). This is because
        we store per-layer parameter weights.

        Args:
            * initial_base_model_lr: Initial inner-loop learning rate of the base_model
                - this value is learned over the course of meta-learning
            * initial_classifier_lr: Initial inner-loop learning rate of the classifier head
                - this value is learned over the course of meta-learning
            * num_innerloop_steps: Number of gradients steps in the inner loop used to learn the
                meta-learning task
            * use_first_order: Whether a first order approximation of higher-order gradients
                should be used (NOTE: THIS IS DEPRECATED - @rdiehlmartinez should be removed)
        """
        super().__init__(**kwargs)

        self.inner_layers_lr = {
            layer: torch.nn.Parameter(
                torch.tensor(float(initial_base_model_lr)).to(self.base_device)
            )
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
        self.num_innerloop_steps = int(num_innerloop_steps)

        # set flag to indicate if first-order approximation should be used (à la Reptile)
        if isinstance(use_first_order, str):
            self.use_first_order = eval(use_first_order)
        else:
            self.use_first_order = use_first_order

    ### Base setup functionality for meta learning models

    def innerloop_optimizer_param_groups(
        self,
    ) -> Iterator[Dict[str, torch.nn.Parameter]]:
        """
        Returns the parameter groups that are passed to the innerloop (diferentiable) optimizer.

        Yields:
            * param_groups: A list of dictionaries containing the parameters and learning rates
                for each parameter group
        """

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

        for name, param in self.base_model.named_parameters():
            if not param.requires_grad:
                continue

            param_group = {
                "params": param,
            }

            layer_num = extract_layer_num(name)
            if layer_num is None:
                raise Exception(
                    "Could not find an innerloop learning rate for param: {}".format(
                        name
                    )
                )

            param_group["lr"] = self.inner_layers_lr[layer_num]

            param_groups.append(param_group)

        return param_groups

    def outerloop_optimizer_param_groups(
        self,
    ) -> Iterator[Dict[str, torch.nn.Parameter]]:
        """
        Returns the parameter groups that are passed to the outerloop optimizer, extends the
        base behavior by adding in the learnable inner-loop learning rates.
        """

        param_groups = super().outerloop_optimizer_param_groups()

        param_groups.extend(
            [
                {
                    "params": self.inner_layers_lr.values(),
                },
                {
                    "params": [self.classifier_lr],
                },
            ]
        )

        return param_groups
