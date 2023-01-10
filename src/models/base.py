__author__ = "Richard Diehl Martinez"
""" Base ABC class interface for base models """

import abc
import torch


class BaseModel(metaclass=abc.ABCMeta):

    """
    Base interface for model classes. We primarily define these so that if in the future we do not
    want to use a torch.nn.Module we can continue to ensure models share a similar interface.

    One important NOTE - the base_model should contain parameters that are specified names
    according to what layer the parameter is in (e.g. 'attention.layer.1'). This is because
    we store per-layer parameter weights that are used for the maml and platipus meta-learning
    methods.
    """

    @classmethod
    @abc.abstractmethod
    def from_kwargs(**kwargs):
        """
        Base models should be initialized via this class method that reads in
        keywords from an experiment config
        """
        raise NotImplementedError()

    @property
    @abc.abstractmethod
    def hidden_dim(self):
        """Return the hidden dimension size of the model"""
        raise NotImplementedError()

    @property
    @abc.abstractmethod
    def trainable_layers(self):
        """Return the layers that are trainable (identified by layer number)"""
        raise NotImplementedError()

    @abc.abstractmethod
    def to(self, *args, **kwargs):
        """Move model onto a given device"""

    @abc.abstractmethod
    def forward(self, *args, **kwargs):
        """
        Must implement a forward pass through the model. Should return the output of the
        final layer (i.e. the final hidden states).
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def __str__(self):
        """For debugging purposes printing model should return something useful"""
        raise NotImplementedError()

    @abc.abstractmethod
    def named_parameters(self):
        """Returns a list of tuple of (name, parameter) (SEE NOTE above re naming convention)"""
        raise NotImplementedError()

    @abc.abstractmethod
    def parameters(self):
        """Returns a list of parameters in the model"""
        raise NotImplementedError()
