from __future__ import annotations

import torch


def average_treatment_effect(mu0: torch.Tensor, mu1: torch.Tensor) -> float:
    """Return the Average Treatment Effect (ATE).

    Args:
        mu0: Potential outcomes under control.
        mu1: Potential outcomes under treatment.

    Returns:
        Mean effect ``E[mu1 - mu0]`` as ``float``.
    """
    return torch.mean(mu1 - mu0).item()


def gaussian_log_likelihood(
    y: torch.Tensor, mean: torch.Tensor, var: torch.Tensor
) -> float:
    """Return the average Gaussian log-likelihood.

    ``var`` must contain strictly positive values.

    Args:
        y: Observed outcomes.
        mean: Predicted mean of the Gaussian distribution.
        var: Predicted variance.

    Returns:
        Mean log probability of ``y`` under the Gaussian ``N(mean, var)``.
    """
    log_prob = -0.5 * (torch.log(2 * torch.pi * var) + (y - mean) ** 2 / var)
    return torch.mean(log_prob).item()
