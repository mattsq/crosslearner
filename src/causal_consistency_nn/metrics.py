"""Utility metrics for causal inference models."""

from __future__ import annotations

import math
import torch
import torch.nn.functional as F


def ate(mu0: torch.Tensor, mu1: torch.Tensor) -> float:
    """Return the Average Treatment Effect from potential outcomes."""
    return torch.mean(mu1 - mu0).item()


def att(mu0: torch.Tensor, mu1: torch.Tensor, t: torch.Tensor) -> float:
    """Return the Average Treatment Effect on the Treated."""
    mask = t.view(-1).bool()
    return torch.mean((mu1 - mu0).view(-1)[mask]).item()


def gaussian_log_likelihood(
    y: torch.Tensor, mean: torch.Tensor, sigma: float | torch.Tensor
) -> float:
    """Return the average Gaussian log-likelihood."""
    if isinstance(sigma, (float, int)):
        var = float(sigma) ** 2
    else:
        var = sigma**2
    ll = -(
        0.5 * torch.log(torch.tensor(2 * math.pi * var)) + (y - mean) ** 2 / (2 * var)
    )
    return ll.mean().item()


def bernoulli_log_likelihood(target: torch.Tensor, logits: torch.Tensor) -> float:
    """Return the average Bernoulli log-likelihood for ``logits``."""
    ll = -F.binary_cross_entropy_with_logits(logits, target, reduction="mean")
    return ll.item()
