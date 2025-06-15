"""Utility functions for visualising training and predictions."""

import torch
from matplotlib import pyplot as plt

from .models.acx import ACX
from .training.history import History


def plot_losses(history: History):
    """Return a matplotlib Figure showing loss curves.

    Args:
        history: List of :class:`~crosslearner.training.history.EpochStats`.

    Returns:
        Matplotlib figure with the loss trajectories.
    """
    epochs = [h.epoch for h in history]
    fig, ax = plt.subplots()
    ax.plot(epochs, [h.loss_d for h in history], label="discriminator")
    ax.plot(epochs, [h.loss_g for h in history], label="generator")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.legend()
    fig.tight_layout()
    return fig


def scatter_tau(model: ACX, X: torch.Tensor, mu0: torch.Tensor, mu1: torch.Tensor):
    """Return a Figure comparing true and predicted treatment effects.

    Args:
        model: Trained ``ACX`` model.
        X: Covariate matrix.
        mu0: True control outcomes.
        mu1: True treated outcomes.

    Returns:
        Matplotlib figure with a scatter plot of true vs predicted ``tau``.
    """
    model.eval()
    with torch.no_grad():
        _, _, _, tau_hat = model(X)
    tau_true = mu1 - mu0
    fig, ax = plt.subplots()
    ax.scatter(tau_true.cpu(), tau_hat.cpu(), alpha=0.5)
    maxv = float(torch.max(torch.abs(torch.cat([tau_true, tau_hat]))))
    ax.plot([-maxv, maxv], [-maxv, maxv], "r--", linewidth=1)
    ax.set_xlabel("true tau")
    ax.set_ylabel("predicted tau")
    fig.tight_layout()
    return fig
