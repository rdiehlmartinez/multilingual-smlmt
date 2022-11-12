__author__ = 'Richard Diehl Martinez' 
""" Base interface class for NLUDatasets and NLUDatasetGenerators"""

import abc
import logging

from collections import defaultdict
from torch.utils.data import IterableDataset

from .nludataloader import NLUCollator

# import statements for type hints 
import torch
from configparser import ConfigParser
from typing import Tuple, List, Dict

logger = logging.getLogger(__name__)

nlu_collate = NLUCollator()

class NLUTaskDataGenerator(metaclass=abc.ABCMeta):

    def __init__(self, config: ConfigParser) -> None:
        """
        Base class for generators that yield NLUDataset classes. Requires children to be iterators.
        """
        pass

    @property
    @abc.abstractmethod
    def num_classes(self) -> int:
        """ The number of classes in the current NLU task"""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def task_type(self) -> str:
        """ The type of NLU task (e.g. classifiation)"""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def task_head_init_method(self) -> str:
        """ The method for initializing the task-specific head of the model """
        raise NotImplementedError

    @abc.abstractmethod
    def __iter__(self):
        raise NotImplementedError()

class NLUDataset(IterableDataset, metaclass=abc.ABCMeta):

    """
    Base dataset for processing data for a specific language of an NLU task. Should be 
    used in conjugtion with the NLUDataLoader.
    """

    def __init__(self, lng: str, file_path: str) -> None: 
        """
        For a given language string and data filepath, establishes an IterableDataset that 
        iterates over and processes the NLU data for that language. The language_task_kwargs
        keyword arg should only be set if we are using the platipus learner method.
        If so, we can use this class to generate a batch of data that is used by platipus to adapt
        and sample the weights of the model

        Args: 
            * lng (str): language string
            * file_path (str): path to the data file
        """

        self._lng = lng
        self.file_path = file_path

    @property
    def language(self) -> str:
        return self._lng

    @abc.abstractmethod
    def preprocess_line(self, line: str) -> Tuple[int, List[int]]:
        """
        For a given text input line, splits, tokenizes and otherwise preprocesses the line.

        Args: 
            * line (str): Line of text 
            
        Returns: 
            * label_id (int): Label for the current sample
            * input_ids (list): List of input tokens
        """
        raise NotImplementedError()

    def __iter__(self): 
        """ Reads over file and preprocesses each of the lines """
        with open(self.file_path, 'r') as f:
            for line in f:
                # tokenize line - recall this method needs to implemented by a subclass
                yield self.preprocess_line(line)
