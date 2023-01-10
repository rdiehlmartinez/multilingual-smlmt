__author__ = "Richard Diehl Martinez"
""" Interface class for (meta) learners """

import abc
import copy
import logging
import math
import os
import re
import time

import torch
from torch.optim import AdamW

from ..utils import move_to_device
from ..datasets import NLUDataLoader
from ..taskheads import TaskHead, ClassificationHead

# imports for typing
from ..datasets import NLUDataset
from ..models import BaseModel
from typing import Tuple, List, Dict, Union, Any, Iterator
from ..utils.data import ShuffledIterableDataset
from ..utils.evaluation import Metric

logger = logging.getLogger(__name__)


class BaseLearner(torch.nn.Module, metaclass=abc.ABCMeta):
    def __init__(
        self,
        base_model: BaseModel,
        base_device: torch.device,
        seed: int,
        lm_head_init_method: str = "protomaml",
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
            init_kwargs = self.get_task_init_kwargs(
                "classification",
                self.lm_head_init_method,
                self.lm_head_n,
            )
            self.retained_lm_head_weights = TaskHead.initialize_task_head(**init_kwargs)

        else:
            # If we are re-initializing the LM head for each training task, then we should use
            # protomaml
            if "protomaml" not in self.lm_head_init_method:
                logger.warning(
                    "LM head will be reinitialized without protomaml (NOT RECOMMENDED)"
                )

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

        init_kwargs["task_type"] = task_type
        init_kwargs["task_init_method"] = task_init_method
        init_kwargs["n_labels"] = n_labels

        init_kwargs["base_model_hidden_dim"] = self.base_model_hidden_dim
        init_kwargs["device"] = device if device is not None else self.base_device

        if "protomaml" in task_init_method:
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
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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
            * logits: logits for the classification task
            * loss: loss for the classification task
        """

        # indexing into sequence layer of model_outputs -> (batch_size, hidden_size)
        batch_size = model_outputs.size(0)

        last_hidden_state = model_outputs[
            torch.arange(batch_size), data_batch["input_target_idx"]
        ]

        if task_type == "classification":
            head = ClassificationHead()
        else:
            logger.exception(f"Invalid task type: {task_type}")
            raise Exception(f"Invalid task type: {task_type}")

        logits, loss = head(
            model_output=last_hidden_state,
            labels=data_batch["label_ids"],
            weights=task_head_weights,
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

    def run_evaluation(
        self,
        finetune_dataset: Union[ShuffledIterableDataset, NLUDataset],
        eval_dataset: NLUDataset,
        eval_type: str,
        metric: Metric,
        task_type: str,
        task_head_init_method: str,
        num_classes: int,
        batch_size: int,
        max_epochs: int,
        lr: float,
        device: torch.device = None,
        return_finetune_info: bool = True,
    ) -> Union[Tuple[float, float], Tuple[float, float, Dict[str, List[float]]]]:
        """
        Runs finetuning of the model on the support set data stored in the finetune_dataset and
        evaluates the model on the evaluation set data stored in the eval_dataset.

        Args:
            * finetune_dataset: An NLU Dataset containing the data for finetuning the
                pre-trained model on a given NLU task
            * eval_dataset: An NLU Dataset containing the evaluation set data for the given NLU
                task
            * eval_type: A string indicating the type of evaluation (standard, few_shot or
                cross_lingual)
            * metric: A callable metric instance that uses the finetuned model to compute the
                target evaluation metric on the eval_dataloader
            * task_type: A string indicating the type of task (e.g. 'classification', 'qa', etc.)
            * task_head_init_method: A string indicating the method used to initialize the task
                head weights
            * num_classes: An integer indicating the number of classes in the task
            * batch_size: An integer indicating the batch size to use for the finetuning
            * max_epochs: An integer indicating the max number of epochs to run the finetuning;
                we break out of the loop if the validation loss stops improving
            * lr: A float indicating the initial learning rate to use for finetuning
            * device: Optional string to specify a device to override base_device
            * return_finetune_info: A boolean indicating whether to return the losses
                and accuracies for the finetuning process

        Returns:
            * eval_metric: A float indicating the evaluation metric on the eval_dataloader
            * eval_loss: A float indicating the loss on the eval_dataloader
            * finetune_info: A dictionary containing the losses and accuracies for the finetuning
                process

        """

        if device is None:
            device = self.base_device

        self.base_model.to(device)

        # Setting up the task head for the task
        with torch.no_grad():
            if task_head_init_method == "protomaml":
                # If we are using protomaml, we use the first batch of the finetune dataset
                # to initialize the task head
                finetune_dataloader = NLUDataLoader(
                    finetune_dataset, batch_size=batch_size
                )
                init_data_batch = move_to_device(
                    next(iter(finetune_dataloader)), device
                )
            else:
                init_data_batch = None

            init_kwargs = self.get_task_init_kwargs(
                task_type,
                task_head_init_method,
                num_classes,
                data_batch=init_data_batch,
                device=device,
            )

            task_head_weights = TaskHead.initialize_task_head(**init_kwargs)

        finetune_model = copy.deepcopy(self.base_model)

        # NOTE: we only train to convergence when we are training on the full dataset;
        # otherwise, we train for a fixed number of batches (i.e. epochs)
        train_to_convergence = eval_type != "few_shot"

        if train_to_convergence:
            # only applicable for full model tuning
            MAX_PATIENCE = 3  # NOTE: un-tuned hyperparameter
            EVAL_EVERY_N_STEPS = 1  # NOTE: un-tuned hyperparameter

            patience = MAX_PATIENCE
            best_dev_metric = None

        if train_to_convergence:
            dev_batch = move_to_device(finetune_dataset.dev_batch, device)

        # Setting up the optimizer
        finetune_optimizer_param_groups = self.finetune_optimizer_param_groups(
            finetune_model,
            task_head_weights,
            add_decay_information=True,
            weight_decay_val=0.0,  # NOTE: un-tuned hyperparameter
        )
        finetune_optimizer = AdamW(finetune_optimizer_param_groups, lr=lr)
        finetune_model.train()

        total_step_num = 0
        early_exit_training = False

        if return_finetune_info:
            # Setting up the training info dictionary
            finetune_info = []

        for epoch in range(max_epochs):

            if early_exit_training:
                break

            finetune_dataloader = NLUDataLoader(finetune_dataset, batch_size=batch_size)

            # Finetune the model on the data in the finetune dataloader
            for finetune_batch in finetune_dataloader:

                finetune_optimizer.zero_grad()

                finetune_batch = move_to_device(finetune_batch, device)

                outputs = finetune_model(
                    input_ids=finetune_batch["input_ids"],
                    attention_mask=finetune_batch["attention_mask"],
                )

                _, loss = self._compute_task_loss(
                    outputs, finetune_batch, task_head_weights, task_type=task_type
                )

                if train_to_convergence and total_step_num % EVAL_EVERY_N_STEPS == 0:
                    # Evaluating the model on the dev set to possbily break out early
                    with torch.no_grad():
                        finetune_model.eval()

                        outputs = finetune_model(
                            input_ids=dev_batch["input_ids"],
                            attention_mask=dev_batch["attention_mask"],
                        )

                        dev_logits, dev_loss = self._compute_task_loss(
                            outputs, dev_batch, task_head_weights, task_type=task_type
                        )

                        dev_predictions = torch.argmax(dev_logits, dim=-1).tolist()
                        dev_labels = dev_batch["label_ids"].tolist()

                        dev_metric = metric(dev_predictions, dev_labels)

                        if (
                            best_dev_metric is None
                            or metric.summary(dev_metric, best_dev_metric) == dev_metric
                        ):

                            best_dev_metric = dev_metric
                            patience = MAX_PATIENCE
                        else:
                            patience -= 1
                            if patience == 0:
                                early_exit_training = True

                        finetune_model.train()

                        if return_finetune_info:
                            finetune_info.append(
                                {
                                    "finetune_loss": loss.item(),
                                    "dev_loss": dev_loss.item(),
                                    "dev_metric": dev_metric,
                                    "step_num": total_step_num,
                                }
                            )

                if early_exit_training:
                    break

                loss.backward()
                finetune_optimizer.step()

                total_step_num += 1

        # Running full evaluation
        finetune_model.eval()

        eval_labels = []
        eval_predictions = []

        total_eval_loss = 0.0
        total_eval_samples = 0

        eval_dataloader = NLUDataLoader(eval_dataset, batch_size=batch_size)

        with torch.no_grad():

            for eval_batch in eval_dataloader:
                eval_batch = move_to_device(eval_batch, device)

                eval_outputs = finetune_model(
                    input_ids=eval_batch["input_ids"],
                    attention_mask=eval_batch["attention_mask"],
                )

                eval_logits, eval_loss = self._compute_task_loss(
                    eval_outputs, eval_batch, task_head_weights, task_type=task_type
                )

                eval_predictions.extend(torch.argmax(eval_logits, dim=-1).tolist())
                eval_labels.extend(eval_batch["label_ids"].tolist())

                batch_size = eval_logits.size(0)
                total_eval_loss += (
                    eval_loss.detach().item() * batch_size
                )  # loss avg across batch
                total_eval_samples += batch_size

            total_eval_loss /= total_eval_samples

            eval_metric = metric(eval_predictions, eval_labels)

        eval_results = {
            "eval_language": eval_dataset.language,
            "eval_loss": total_eval_loss,
            "eval_metric": eval_metric,
            "num_finetune_steps": total_step_num,
        }

        if return_finetune_info:
            eval_results["finetune_info"] = finetune_info

        if eval_type == "cross_lingual":
            eval_results["finetune_language"] = finetune_dataset.language
        elif eval_type == "few_shot":
            eval_results["k"] = finetune_dataset.K
            eval_results["n"] = num_classes

        return eval_results


class BaseMetaLearner(BaseLearner):
    def __init__(
        self,
        initial_base_model_lr: Union[float, str] = 1e-2,
        initial_classifier_lr: Union[float, str] = 1e-1,
        num_innerloop_steps: Union[int, str] = 7,
        use_first_order: Union[bool, str] = False,
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
                should be used
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
