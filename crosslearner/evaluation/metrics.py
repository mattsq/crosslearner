"""Evaluation metrics."""

import torch


def pehe(tau_hat: torch.Tensor, tau_true: torch.Tensor) -> float:
    """Return sqrt PEHE between predictions and ground truth.

    Args:
        tau_hat: Predicted treatment effects.
        tau_true: True treatment effects.

    Returns:
        Square-root PEHE.
    """
    return torch.mean((tau_hat - tau_true) ** 2).sqrt().item()
