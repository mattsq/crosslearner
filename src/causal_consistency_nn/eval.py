"""Dataset evaluation utilities."""

from __future__ import annotations

import torch
from torch.utils.data import TensorDataset

from .metrics import ate, gaussian_log_likelihood


def evaluate(
    mu0_pred: torch.Tensor,
    mu1_pred: torch.Tensor,
    dataset: TensorDataset,
    *,
    mu0_true: torch.Tensor | None = None,
    mu1_true: torch.Tensor | None = None,
    noise: float = 1.0,
) -> dict[str, float]:
    """Compute ATE and log-likelihood metrics for ``dataset``.

    Args:
        mu0_pred: Predicted outcome under control.
        mu1_pred: Predicted outcome under treatment.
        dataset: Dataset containing ``(X, T, Y)`` tensors.
        mu0_true: Optional true control outcomes.
        mu1_true: Optional true treated outcomes.
        noise: Noise standard deviation assumed for the outcomes.
    """

    _, T, Y = dataset.tensors
    ate_hat = ate(mu0_pred, mu1_pred)
    pred_y = torch.where(T.bool(), mu1_pred, mu0_pred)
    loglik = gaussian_log_likelihood(Y, pred_y, noise)

    metrics = {"ate": ate_hat, "log_likelihood": loglik}
    if mu0_true is not None and mu1_true is not None:
        metrics["ate_error"] = ate_hat - ate(mu0_true, mu1_true)
    return metrics
