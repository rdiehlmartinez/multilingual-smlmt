import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import TaskHead

# typing imports
from typing import Tuple, Dict


class QAHead(TaskHead):
    """Task head for question answering tasks"""

    def __call__(
        self,
        model_outputs: torch.Tensor,
        data_batch: Dict[str, torch.Tensor],
        weights: nn.ParameterDict,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Runs a forward pass of the question answering (i.e. span prediction head).
        Architecture inspired by the huggingface implementation of RobertaForQuestionAnswering head.

        Args:
            * model_outputs: output of the base model
            * labels: a dictionary containing the start and end labels for the question answering
                task, must contain keys "start_positions" and "end_positions"
            * weights: weights for qa projection, must contain keys "weight" and
                "bias"
        Returns:
            * logits: logits for the qa task (encapsulates both the start and end logits)
            * loss: loss for the qa task
        """

        logits = F.linear(model_outputs, **weights)

        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits, end_logits = start_logits.squeeze(-1), end_logits.squeeze(-1)

        # NOTE: as the ignore index we pass in the max size of
        loss_function = nn.CrossEntropyLoss(ignore_index=start_logits.size(1))

        start_positions = data_batch["start_positions"]
        end_positions = data_batch["end_positions"]

        start_loss = loss_function(start_logits, start_positions)
        end_loss = loss_function(end_logits, end_positions)

        total_loss = (start_loss + end_loss) / 2

        return (logits, total_loss)


@TaskHead.register_initialization_method
def qa_random(
    base_model_hidden_dim: int,
    device: str,
) -> nn.ParameterDict:
    """
    Initializes a random QA head.

    Args:
        * base_model_hidden_dim: hidden dimension of the base model
        * device: device to initialize the head on
    Returns:
        * task_head_weight (nn.ParameterDict): {
            * weight -> (nn.Parameter): weight of the projection matrix for the qa task
            * bias -> (nn.Parameter): bias of the projection for the qa task
            }
    """

    weight = nn.Parameter(torch.randn(2, base_model_hidden_dim), requires_grad=True).to(
        device
    )
    bias = nn.Parameter(torch.randn(2), requires_grad=True).to(device)

    task_head_weights = {
        "weight": weight,
        "bias": bias,
    }

    return task_head_weights
