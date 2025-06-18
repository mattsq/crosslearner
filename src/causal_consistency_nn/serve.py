"""Model serving helpers."""

from __future__ import annotations

import torch
from crosslearner.utils import model_device


def _forward(model: torch.nn.Module, X: torch.Tensor) -> tuple[torch.Tensor, ...]:
    device = model_device(model)
    X = X.to(device)
    model.eval()
    with torch.no_grad():
        out = model(X)
    if not isinstance(out, tuple):
        raise TypeError("model must return a tuple of tensors")
    return out


def predict_z(model: torch.nn.Module, X: torch.Tensor) -> torch.Tensor:
    """Return latent representation ``z`` for ``X``."""
    z, *_ = _forward(model, X)
    return z.cpu()


def counterfactual_z(
    model: torch.nn.Module, X: torch.Tensor, t: torch.Tensor
) -> torch.Tensor:
    """Return counterfactual outcome predictions."""
    _, mu0, mu1, _ = _forward(model, X)
    return torch.where(t.to(mu0.device).bool(), mu0, mu1).cpu()


def impute_y(
    model: torch.nn.Module,
    X: torch.Tensor,
    t: torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor:
    """Impute missing outcomes in ``y`` with model predictions."""
    device = model_device(model)
    X = X.to(device)
    t = t.to(device)
    mask = torch.isnan(y)
    if not mask.any():
        return y
    model.eval()
    with torch.no_grad():
        _, mu0, mu1, _ = model(X)
        pred = torch.where(t.bool(), mu1, mu0).cpu()
    y = y.clone()
    y[mask] = pred[mask]
    return y
