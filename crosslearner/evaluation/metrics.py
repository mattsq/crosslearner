import torch


def pehe(tau_hat: torch.Tensor, tau_true: torch.Tensor) -> float:
    """Return sqrt PEHE between predictions and ground truth."""
    return torch.mean((tau_hat - tau_true) ** 2).sqrt().item()
