from __future__ import annotations

import torch


def _device(model: torch.nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:  # pragma: no cover
        return torch.device("cpu")


def predict_z(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Return the latent representation ``z`` for ``x``."""
    device = _device(model)
    model.eval()
    with torch.no_grad():
        out = model(x.to(device))
        if isinstance(out, (tuple, list)):
            z = out[0]
        else:
            z = out
    return z.cpu()


def counterfactual_z(
    model: torch.nn.Module, x: torch.Tensor, t: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return representation and counterfactual outcome for ``x``."""
    device = _device(model)
    model.eval()
    with torch.no_grad():
        out = model(x.to(device))
        if isinstance(out, (tuple, list)) and len(out) >= 3:
            z, mu0, mu1 = out[:3]
        else:
            raise ValueError("model must output (z, mu0, mu1)")
        ycf = torch.where(t.to(device).bool(), mu0, mu1)
    return z.cpu(), ycf.cpu()


def impute_y(model: torch.nn.Module, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Return outcome prediction for treatment ``t``."""
    device = _device(model)
    model.eval()
    with torch.no_grad():
        out = model(x.to(device))
        if isinstance(out, (tuple, list)) and len(out) >= 3:
            _, mu0, mu1 = out[:3]
        else:
            raise ValueError("model must output (z, mu0, mu1)")
        ypred = torch.where(t.to(device).bool(), mu1, mu0)
    return ypred.cpu()
