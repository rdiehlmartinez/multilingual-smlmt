__author__ = "Richard Diehl Martinez"
""" Base interface class for NLU tasks (i.e. MLQA, XNLI, etc.)"""

import abc
import logging

# import statements for type hints
from configparser import ConfigParser
from typing import List, Dict
from torch import Tensor

logger = logging.getLogger(__name__)


class NLUTaskGenerator(metaclass=abc.ABCMeta):
    def __init__(self, config: ConfigParser, eval_type: str) -> None:
        """
        Base class for a NLU task (i.e. MLQA, XNLI, etc.). Each task generator stores parameters
        associated with the given task and provides an iterator that yields dicts of datasets
        for finetuning and evaluating the pre-trained model on the given task.
        """

        assert eval_type in [
            "standard",
            "few_shot",
            "cross_lingual",
        ], "eval_type must be one of standard, few-shot, cross-lingual"

        task_name = self.__class__.__name__.split("Generator")[0]
        self.config_name = task_name + "_" + eval_type.upper()

        self._batch_size = config.getint(self.config_name, "batch_size", fallback=128)
        self._max_epochs = config.getint(self.config_name, "max_epochs", fallback=5)
        self._lr = config.getfloat(self.config_name, "lr", fallback=1e-5)
        self._task_head_init_method = config.get(
            self.config_name, "task_head_init_method"
        )

        self._eval_type = eval_type  # few-shot, standard, cross-lingual

        # specify the languages to evaluate on
        self.eval_languages = config.get(self.config_name, "eval_languages").split(",")

    ### General properties of tasks ###

    @property
    def eval_type(self) -> str:
        """The type of evaluation that is being done (e.g. standard, few-shot, cross-lingual)"""
        return self._eval_type

    @property
    @abc.abstractmethod
    def task_type(self) -> str:
        """The type of NLU task (e.g. classifiation, qa)"""
        raise NotImplementedError

    ### Tasks define methods for computing a metric ###

    @property
    @abc.abstractmethod
    def metric_name(self) -> str:
        """The name of the metric used to evaluate the task"""
        raise NotImplementedError

    @abc.abstractmethod
    def compute_metric(self, logits: Tensor, labels: Tensor) -> Dict:
        """Compute the metric for the task given the predictions and labels"""
        raise NotImplementedError

    @abc.abstractmethod
    def metric_is_better(self, curr_metric: float, best_metric: float) -> bool:
        """Returns True if curr_metric is better than best_metric"""
        raise NotImplementedError

    ### Helper functions for processing data to pass into model ###
    @abc.abstractmethod
    def process_batch(self, batch: List[Tensor]) -> Dict[str, Tensor]:
        """
        Given a batch of data, transform it into a dictionary of tensors that can be passed
        into the model.
        """
        raise NotImplementedError

    ### A method for iterating over the task that yields dicts of datasets for each language ###

    @abc.abstractmethod
    def __iter__(self):
        """
        Should yield a dictionary structure storing the finetune, dev and eval datasets for the
        task for each language.

            ex. for cross-lingual evaluation english -> french
            [
                {
                    "finetune": {
                        "language": "en",
                        "dataset": torch.TensorDataset(...)
                    }
                    "dev": {
                        "language": "fr",
                        "dataset": torch.TensorDataset(...)
                    }
                    "eval": {
                        "language": "fr",
                        "dataset": torch.TensorDataset(...)
                    }
                },
                (... for other languages similar to above)
            ]
        """
        raise NotImplementedError()

    ### Properties for finetuning the model on this task ###

    @property
    def batch_size(self) -> int:
        """Batch size for finetuning the model on this task"""
        return self._batch_size

    @property
    def max_epochs(self) -> int:
        """Maximum number of epochs to finetune the model on this task"""
        return self._max_epochs

    @property
    def lr(self) -> float:
        """Learning rate for finetuning the model on this task"""
        return self._lr

    @property
    def task_head_init_method(self) -> str:
        """The method for initializing the task-specific head of the model"""
        return self._task_head_init_method

    ### Finetuning related properties ###
    @property
    def max_patience(self) -> int:
        return 3

    @property
    def eval_every_n_steps(self) -> int:
        return 30
