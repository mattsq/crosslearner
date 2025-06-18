from __future__ import annotations

from typing import Dict

import torch
from torch.utils.data import DataLoader

from .metrics import average_treatment_effect, gaussian_log_likelihood


def _device(model: torch.nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:  # pragma: no cover
        return torch.device("cpu")


def evaluate(
    model: torch.nn.Module, loader: DataLoader, *, device: str | None = None
) -> Dict[str, float]:
    """Compute ATE and log-likelihood metrics for ``model`` on ``loader``.

    The ``model`` is expected to return at least ``(mu0, mu1)`` when called on
    a batch of covariates. Additional outputs are ignored.
    """
    device = device or _device(model)
    mu0_list: list[torch.Tensor] = []
    mu1_list: list[torch.Tensor] = []
    ll_values: list[float] = []

    model.eval()
    with torch.no_grad():
        for xb, tb, yb in loader:
            xb = xb.to(device)
            tb = tb.to(device)
            yb = yb.to(device)

            out = model(xb)
            if isinstance(out, (tuple, list)):
                if len(out) >= 3:
                    mu0, mu1 = out[1], out[2]
                else:
                    mu0, mu1 = out[:2]
            else:
                raise ValueError(
                    "model output must be tuple/list containing mu0 and mu1"
                )

            mu0_list.append(mu0.cpu())
            mu1_list.append(mu1.cpu())
            pred_y = torch.where(tb.bool(), mu1, mu0)
            ll = gaussian_log_likelihood(yb, pred_y, torch.ones_like(pred_y))
            ll_values.append(ll)

    mu0_all = torch.cat(mu0_list)
    mu1_all = torch.cat(mu1_list)
    ate = average_treatment_effect(mu0_all, mu1_all)
    ll_mean = float(sum(ll_values) / len(ll_values)) if ll_values else float("nan")
    return {"ate": ate, "log_likelihood": ll_mean}
