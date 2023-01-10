__author__ = "Richard Diehl Martinez"
"""
Helper classes for the evaluator; define interface for different types of evaluation metrics i.e. 
accuracy, F1, etc.
"""

import abc
import numpy as np

from typing import Callable, List


class Metric(object, metaclass=abc.ABCMeta):
    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Name of the metric"""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def summary(self) -> Callable:
        """Summary function to use for the metric"""
        raise NotImplementedError

    @staticmethod
    @abc.abstractmethod
    def __call__(self, predictions: List[int], labels: List[int]) -> float:
        """
        Computes the metric given the predictions and labels

        Args:
            predictions: List of predictions
            labels: List of labels

        Returns:
            metric: The metric value

        """
        raise NotImplementedError


class AccuracyMetric(Metric):
    @property
    def name(self):
        return "accuracy"

    @property
    def summary(self):
        return max

    @staticmethod
    def __call__(predictions: List[int], labels: List[int]) -> float:
        """
        Computes accuracy of predictions for the data of the eval_dataloader
        """
        accuracy = (np.array(predictions) == np.array(labels)).sum() / len(labels)
        return accuracy
