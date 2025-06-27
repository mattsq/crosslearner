"""Uncertainty estimation utilities."""

from __future__ import annotations

from typing import Tuple

import torch

from crosslearner.models.acx import ACX
from crosslearner.utils import model_device


@torch.no_grad()
def predict_tau_mc_dropout(
    model: ACX, X: torch.Tensor, *, passes: int = 100
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return mean and standard deviation of MC dropout predictions.

    Args:
        model: Trained ``ACX`` model with dropout layers.
        X: Input covariates ``(n, p)``.
        passes: Number of stochastic forward passes.

    Returns:
        Tuple ``(mean, std)`` with aggregated predictions.
    """

    device = model_device(model)
    X = X.to(device)
    model.train()
    samples = []
    for _ in range(passes):
        _, _, _, tau = model(X)
        samples.append(tau)
    stacked = torch.stack(samples)
    mean = stacked.mean(dim=0)
    std = stacked.std(dim=0)
    model.eval()
    return mean, std


@torch.no_grad()
def predict_tau_ensemble(
    models: Tuple[ACX, ...] | list[ACX], X: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return mean and standard deviation of ensemble CATE predictions.

    Args:
        models: Sequence of trained ``ACX`` models.
        X: Covariates ``(n, p)`` used for prediction.

    Returns:
        Tuple ``(mean, std)`` over the ensemble predictions.
    """

    samples = []
    for model in models:
        device = model_device(model)
        was_training = model.training
        model.eval()
        _, _, _, tau = model(X.to(device))
        if was_training:
            model.train()
        samples.append(tau.cpu())
    stacked = torch.stack(samples)
    mean = stacked.mean(dim=0)
    std = stacked.std(dim=0)
    return mean, std
