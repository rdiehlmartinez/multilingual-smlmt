__author__ = "Richard Diehl Martinez"
""" XLM-r model implementation """

import typing
import torch
import json
import logging

from transformers import XLMRobertaModel
from .base import BaseModel

# typing imports
from typing import List, Union

logger = logging.getLogger(__name__)


@BaseModel.register
class XLMR(XLMRobertaModel):
    """Implementation of XLM-r model (Conneau et al. 2019) https://arxiv.org/abs/1911.02116"""

    @classmethod
    def from_kwargs(
        cls,
        pretrained_model_name: str = "xlm-roberta-base",
        trainable_layers: Union[List[int], str] = [],
        **kwargs,
    ) -> None:
        """Loading in huggingface XLM-R model for masked LM"""

        if pretrained_model_name:
            model = cls.from_pretrained(pretrained_model_name)

            if "base" in pretrained_model_name:
                model._hidden_dim = 768
            elif "large" in pretrained_model_name:
                model._hidden_dim = 1024
            else:
                logger.exception(
                    f"Cannot infer hidden dim for model: {pretrained_model_name}"
                )
                raise Exception(
                    f"Cannot infer hidden dim for model: {pretrained_model_name}"
                )

        else:
            logger.exception(f"{cls} can only be initialized from a pretrained model")
            raise NotImplementedError(
                f"{cls} can only be initialized from a pretrained model"
            )

        # update model to require gradients only for trainable layers
        if len(trainable_layers) == 0:
            logger.warning("No layers specified to be meta learned")

        if isinstance(trainable_layers, str):
            trainable_layers = json.loads(trainable_layers)

        model._trainable_layers = trainable_layers

        for name, param in model.named_parameters():
            if any(f"layer.{layer_num}" in name for layer_num in trainable_layers):
                param.requires_grad = True
            else:
                param.requires_grad = False

        return model

    @property
    def trainable_layers(self) -> List[int]:
        """Returns a list of the trainable layers (identified by their layer number)"""
        return self._trainable_layers

    def forward(self, *args, **kwargs) -> torch.Tensor:
        """
        Overriding default forward method to only return the output tensor of hidden states
        from the final layer - has shape: (batch_size, sequence_length, hidden_size)
        """
        outputs = super().forward(*args, **kwargs)
        return outputs.last_hidden_state

    @property
    def hidden_dim(self) -> int:
        return self._hidden_dim
