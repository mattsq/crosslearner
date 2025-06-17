"""Evaluation metrics for treatment effect models."""

from __future__ import annotations

import numpy as np
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


def policy_risk(tau_hat: torch.Tensor, mu0: torch.Tensor, mu1: torch.Tensor) -> float:
    """Return the policy risk of using ``tau_hat`` for treatment decisions.

    The function compares the outcome under the policy implied by ``tau_hat``
    against always choosing the best potential outcome.
    """

    opt_outcome = torch.max(mu0, mu1)
    policy = (tau_hat > 0).float()
    pred_outcome = mu0 * (1.0 - policy) + mu1 * policy
    return torch.mean(opt_outcome - pred_outcome).item()


def ate_error(tau_hat: torch.Tensor, mu0: torch.Tensor, mu1: torch.Tensor) -> float:
    """Return estimation error for the Average Treatment Effect (ATE)."""

    ate_hat = torch.mean(tau_hat)
    ate_true = torch.mean(mu1 - mu0)
    return (ate_hat - ate_true).item()


def att_error(
    tau_hat: torch.Tensor, mu0: torch.Tensor, mu1: torch.Tensor, t: torch.Tensor
) -> float:
    """Return estimation error for the ATT (Average Treatment effect on Treated)."""

    mask = t.view(-1).bool()
    att_hat = torch.mean(tau_hat.view(-1)[mask])
    att_true = torch.mean((mu1 - mu0).view(-1)[mask])
    return (att_hat - att_true).item()


def bootstrap_ci(
    values: torch.Tensor, *, level: float = 0.95, n_boot: int = 1000
) -> tuple[float, float]:
    """Return a bootstrap confidence interval for the mean of ``values``."""

    arr = values.view(-1).cpu().numpy()
    samples = np.random.choice(arr, size=(n_boot, arr.size), replace=True)
    means = samples.mean(axis=1)
    alpha = 1.0 - level
    lower = float(np.percentile(means, 100 * alpha / 2))
    upper = float(np.percentile(means, 100 * (1 - alpha / 2)))
    return lower, upper
