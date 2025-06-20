"""Utility functions for visualising training and predictions."""

import torch
from matplotlib import pyplot as plt

from .models.acx import ACX
from .training.history import History
from .utils import model_device


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
    device = model_device(model)
    X = X.to(device)
    mu0 = mu0.to(device)
    mu1 = mu1.to(device)

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


def plot_tau_distribution(tau: torch.Tensor, bins: int = 30):
    """Return a Figure with the distribution of treatment effects.

    Args:
        tau: Tensor of treatment effect estimates.
        bins: Number of histogram bins.

    Returns:
        Matplotlib figure containing the histogram of ``tau``.
    """
    fig, ax = plt.subplots()
    ax.hist(tau.cpu().view(-1), bins=bins, alpha=0.75)
    ax.set_xlabel("tau")
    ax.set_ylabel("count")
    fig.tight_layout()
    return fig


def plot_covariate_balance(X: torch.Tensor, T: torch.Tensor):
    """Return a Figure summarising covariate balance.

    The function plots standardised mean differences for each feature
    between treated and control units.

    Args:
        X: Covariate matrix ``(n, p)``.
        T: Binary treatment indicator ``(n, 1)`` or ``(n,)``.

    Returns:
        Matplotlib figure with a bar plot of standardised differences.
    """
    T = T.view(-1).bool()
    X_t = X[T]
    X_c = X[~T]
    mean_t = X_t.mean(0)
    mean_c = X_c.mean(0)
    var_t = X_t.var(0, unbiased=False)
    var_c = X_c.var(0, unbiased=False)
    smd = (mean_t - mean_c).abs() / (0.5 * (var_t + var_c)).sqrt()
    fig, ax = plt.subplots()
    ax.bar(range(X.shape[1]), smd.cpu())
    ax.set_xlabel("feature")
    ax.set_ylabel("standardised mean diff")
    fig.tight_layout()
    return fig


def plot_propensity_overlap(propensity: torch.Tensor, T: torch.Tensor, bins: int = 20):
    """Return a Figure showing propensity score overlap.

    Args:
        propensity: Estimated propensity scores ``(n, 1)`` or ``(n,)``.
        T: Binary treatment indicator ``(n, 1)`` or ``(n,)``.
        bins: Number of histogram bins.

    Returns:
        Matplotlib figure with overlaid histograms of propensities.
    """
    propensity = propensity.view(-1)
    T = T.view(-1).bool()
    fig, ax = plt.subplots()
    ax.hist(propensity[T].cpu(), bins=bins, alpha=0.5, label="treated")
    ax.hist(propensity[~T].cpu(), bins=bins, alpha=0.5, label="control")
    ax.set_xlabel("propensity score")
    ax.set_ylabel("count")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_residuals(y_true: torch.Tensor, y_pred: torch.Tensor):
    """Return a residual analysis scatter plot.

    Args:
        y_true: Ground truth outcomes.
        y_pred: Model predictions.

    Returns:
        Matplotlib figure with residuals vs predictions.
    """
    residuals = y_true.view(-1) - y_pred.view(-1)
    fig, ax = plt.subplots()
    ax.scatter(y_pred.view(-1).cpu(), residuals.cpu(), alpha=0.5)
    ax.axhline(0.0, color="r", linestyle="--", linewidth=1)
    ax.set_xlabel("prediction")
    ax.set_ylabel("residual")
    fig.tight_layout()
    return fig


def plot_cate_calibration(
    tau_hat: torch.Tensor, tau_true: torch.Tensor, bins: int = 10
) -> plt.Figure:
    """Return a calibration curve for CATE estimates."""

    tau_hat = tau_hat.view(-1)
    tau_true = tau_true.view(-1)
    edges = torch.linspace(tau_hat.min(), tau_hat.max(), bins + 1)
    bin_idx = torch.bucketize(tau_hat, edges, right=True)
    pred_means = []
    true_means = []
    for i in range(1, len(edges)):
        mask = bin_idx == i
        if mask.any():
            pred_means.append(float(tau_hat[mask].mean()))
            true_means.append(float(tau_true[mask].mean()))
    fig, ax = plt.subplots()
    ax.plot(pred_means, true_means, marker="o")
    minv = float(min(pred_means + true_means))
    maxv = float(max(pred_means + true_means))
    ax.plot([minv, maxv], [minv, maxv], "r--", linewidth=1)
    ax.set_xlabel("predicted tau")
    ax.set_ylabel("true tau")
    fig.tight_layout()
    return fig


def plot_partial_dependence(
    model: ACX,
    X: torch.Tensor,
    feature: int,
    *,
    grid_points: int = 20,
) -> plt.Figure:
    """Return a partial dependence plot for the CATE predictions."""

    device = model_device(model)
    X = X.to(device)
    vals = torch.linspace(X[:, feature].min(), X[:, feature].max(), grid_points)
    vals = vals.to(device)
    pdp = []
    model.eval()
    with torch.no_grad():
        for v in vals:
            Xv = X.clone()
            Xv[:, feature] = v
            _, _, _, tau = model(Xv)
            pdp.append(tau.mean())
    pdp = torch.stack(pdp)
    fig, ax = plt.subplots()
    ax.plot(vals.cpu(), pdp.cpu())
    ax.set_xlabel(f"feature {feature}")
    ax.set_ylabel("predicted tau")
    fig.tight_layout()
    return fig


def plot_ice(
    model: ACX,
    X: torch.Tensor,
    feature: int,
    *,
    grid_points: int = 20,
    sample_limit: int | None = None,
) -> plt.Figure:
    """Return an Individual Conditional Expectation (ICE) plot for the CATE."""

    device = model_device(model)
    X = X.to(device)
    vals = torch.linspace(X[:, feature].min(), X[:, feature].max(), grid_points)
    vals = vals.to(device)
    curves = []
    model.eval()
    with torch.no_grad():
        for v in vals:
            Xv = X.clone()
            Xv[:, feature] = v
            _, _, _, tau = model(Xv)
            curves.append(tau.view(-1))
    ice = torch.stack(curves)
    if sample_limit is not None:
        ice = ice[:, :sample_limit]
    fig, ax = plt.subplots()
    for i in range(ice.shape[1]):
        ax.plot(vals.cpu(), ice[:, i].cpu(), color="gray", alpha=0.3)
    ax.set_xlabel(f"feature {feature}")
    ax.set_ylabel("predicted tau")
    fig.tight_layout()
    return fig
