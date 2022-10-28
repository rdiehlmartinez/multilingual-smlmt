__author__ = 'Richard Diehl Martinez'
""" Defines a task head for classification tasks """

import math 
import logging 

import torch
import torch.nn as nn
import torch.nn.functional as F 

from .base import TaskHead

# typing imports 
from typing import Tuple, Union, List, Dict, Any

logger = logging.getLogger(__name__)

class ClassificationHead(TaskHead):
    """ Task head for classification tasks"""

    loss_function = torch.nn.CrossEntropyLoss()

    def __call__(
        self,
        model_output: torch.Tensor,
        labels: torch.Tensor,
        weights: nn.ParameterDict
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """ 
        Runs a forward pass of the classification head. Architecture inspired by the huggingface
        implementation of RobertaLMHead 

        Args:
            * model_output: output of the model
            * labels: labels for the classification task
            * weights: weights for the classification task
        Returns:
            * logits: logits for the classification task
            * loss: loss for the classification task
        """

        if "fc_weight" in weights:
            fc_weights = {"weight": weights["fc_weight"], "bias": weights["fc_bias"]}

            model_output = F.linear(model_output, **fc_weights)
            model_output = F.gelu(model_output)
            model_output = F.layer_norm(model_output, (model_output.size(-1),))

        classifier_weights = {"weight": weights["classifier_weight"],
                              "bias": weights["classifier_bias"]}

        logits = F.linear(model_output, **classifier_weights)
        loss = self.loss_function(input=logits, target=labels)
        
        return (logits, loss)


@TaskHead.register_initialization_method
def classification_random(
    base_model_hidden_dim: int,
    n_labels: int,
    device: str,
) -> nn.ParameterDict:
    """
    Initializes classification task head using a random Xavier-He initialization method.

    Args: 
        * base_model_hidden_dim: The hidden dimensions of the outputs of the base_model 
        * n_labels: Number of labels (i.e. classes) to classify over 
        * device: Device type ('cuda' or 'cpu')
    Returns: 
        * task_head_weights (nn.ParameterDict): {
            * classifier_weight -> (nn.Parameter): classification weight matrix
            * classifier_bias -> (nn.Parameter): classification bias vector
            }
    """
    # Xavier normal weight implementation
    std_weight = math.sqrt(2.0 / float(base_model_hidden_dim + n_labels))
    std_bias = math.sqrt(2.0 / float(n_labels))

    # weights need to be shape (out_features, in_features) to be compatible with linear layer
    classifier_weight = torch.randn((n_labels, base_model_hidden_dim), device=device) \
                            * std_weight
    classifier_bias = torch.randn((n_labels), device=device) * std_bias

    classifier_weight.requires_grad = True
    classifier_bias.requires_grad = True

    task_head_weights = nn.ParameterDict({
        "classifier_weight": nn.Parameter(classifier_weight),
        "classifier_bias": nn.Parameter(classifier_bias)
    })

    return task_head_weights

@TaskHead.register_initialization_method
def classification_random_fc(
    base_model_hidden_dim: int,
    n_labels: int,
    **kwargs: Dict[str, Any]
) -> nn.ParameterDict:
    """
    Initializes classification task head using a random Xavier-He initialization method. 
    Unlike the classification_random initialization method, this method also includes a 
    fully connected layer that is inserted before the final classification output layer. 

    Args: 
        * base_model_hidden_dim: The hidden dimensions of the outputs of the base_model 
        * n_labels: Number of labels (classes) to classify over 
    Returns: 
        * task_head_weights (nn.ParameterDict): {
            * fc_weight -> (nn.Parameter): weight matrix of fully connected layer
            * fc_bias -> (nn.Parameter): bias vector of fully connected layer 
            * classifier_weight -> (nn.Parameter): classification weight matrix
            * classifier_bias -> (nn.Parameter): classification bias vector
            }
    """
    # Little bit of a hack - can initialize weights of FC layer by
    # repurposing classification_random
    fc_head_weights = classification_random(base_model_hidden_dim, base_model_hidden_dim, **kwargs)
    classifier_weights = classification_random(base_model_hidden_dim, n_labels, **kwargs)

    task_head_weights = nn.ParameterDict({
        "fc_weight": fc_head_weights["classifier_weight"],
        "fc_bias": fc_head_weights["classifier_bias"],
        "classifier_weight": classifier_weights["classifier_weight"],
        "classifier_bias": classifier_weights["classifier_bias"]
    })

    return task_head_weights


@TaskHead.register_initialization_method
def classification_protomaml(
    base_model_hidden_dim: int,
    n_labels: int,
    model: nn.Module,
    data_batch: Dict[str, torch.Tensor],
    device: torch.device,
    params: List[torch.Tensor] = None
) -> nn.ParameterDict:
    """
    Initializes task head using the protomaml (prototypical network + MAML) method. 

    Args: 
        * base_model_hidden_dim: The hidden dimensions of the outputs of the base_model 
        * n_labels: Number of labels (classes) to classify over 
        * model (nn.Module): Either the model or the 'functionalized' version of the base model
        * data_batch: Batch of data for a forward pass through the model 
            (see run_inner_loop for information on the data structure).
        * device: Device type ('cuda' or 'cpu')
        * params: Only needs to be passed in if the model is a functional model;
    Returns: 
        * task_head_weights (nn.ParameterDict): {
            * classifier_weight -> (nn.Parameter): classification weight matrix
            * classifier_bias -> (nn.Parameter): classification bias vector
            }
    """

    if params is not None: 
        outputs = model.forward(input_ids=data_batch['input_ids'],
                                attention_mask=data_batch['attention_mask'],
                                params=[p for p in params])
    else:
        outputs = model(input_ids=data_batch['input_ids'], 
                        attention_mask=data_batch['attention_mask'])

    # outputs has form (batch_size, sequence_length, hidden_size);
    # NOTE: the CLS token at idx position 0 is used as the input representation
    batch_size = outputs.size(0)
    last_hidden_state = outputs[torch.arange(batch_size), 0]

    prototypes = torch.zeros((n_labels, base_model_hidden_dim), device=device)

    for c in range(n_labels):
        idx = torch.nonzero(data_batch['label_ids'] == c).squeeze()
        if idx.nelement() != 0:
            prototypes[c] = torch.mean(last_hidden_state[idx], dim=0)
        else:
            logger.warning("ProtoMaml weight initialization missing at least one class")

    classifier_weight = 2 * prototypes
    classifier_bias = -torch.norm(prototypes, dim=1)**2

    task_head_weights = nn.ParameterDict({
        "classifier_weight": nn.Parameter(classifier_weight),
        "classifier_bias": nn.Parameter(classifier_bias)
    })

    return task_head_weights
