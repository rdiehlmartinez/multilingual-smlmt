__author__ = "Richard Diehl Martinez"
""" Utils for modeling and training"""

import torch

# typing imports
from typing import List


def kl_divergence_gaussians(
    p: List[torch.Tensor], q: List[torch.Tensor]
) -> torch.Tensor:
    """Calculate KL divergence between 2 diagonal Gaussian

    Copied verbatim from:
    https://github.com/cnguyen10/few_shot_meta_learning

    Args:
        * p (List[torch.Tensor]): List of tensors containing the mean and log variance of the
            first Gaussian distribution
        * q (List[torch.Tensor]): List of tensors containing the mean and log variance of the
            second Gaussian distribution
    Returns:
        * kl_div (torch.Tensor): Tensor containing the KL divergence
    """
    assert len(p) == len(q)

    n = len(p) // 2

    kl_div = 0
    for i in range(n):
        p_mean = p[i]
        if not p_mean.requires_grad:
            # Only makes sense to compute kl divergence for parameters we actually learn
            continue

        p_log_std = p[n + i]

        q_mean = q[i]
        q_log_std = q[n + i]

        s1_vec = torch.exp(input=2 * q_log_std)
        mahalanobis = torch.sum(input=torch.square(input=p_mean - q_mean) / s1_vec)

        tr_s1inv_s0 = torch.sum(input=torch.exp(input=2 * (p_log_std - q_log_std)))

        log_det = 2 * torch.sum(input=q_log_std - p_log_std)

        kl_div_temp = mahalanobis + tr_s1inv_s0 + log_det - torch.numel(p_mean)
        kl_div_temp = kl_div_temp / 2

        kl_div = kl_div + kl_div_temp

    return kl_div
