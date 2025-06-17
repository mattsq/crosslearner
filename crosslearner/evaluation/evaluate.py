"""Evaluation utilities."""

import torch
from crosslearner.evaluation.metrics import pehe
from crosslearner.models.acx import ACX


def _forward_to_device(
    model: ACX, X: torch.Tensor
) -> tuple[tuple[torch.Tensor, ...], torch.device]:
    """Return ``model(X)`` with all tensors on the model's device."""

    device = next(model.parameters()).device
    X = X.to(device)
    model.eval()
    with torch.no_grad():
        out = model(X)
    return out, device


def evaluate(
    model: ACX, X: torch.Tensor, mu0: torch.Tensor, mu1: torch.Tensor
) -> float:
    """Compute PEHE of a model on given data.

    Args:
        model: Trained ``ACX`` model.
        X: Covariates ``(n, p)``.
        mu0: Counterfactual outcome under control.
        mu1: Counterfactual outcome under treatment.

    Returns:
        The square-root PEHE value.
    """

    (_, _, _, tau_hat), device = _forward_to_device(model, X)
    mu0 = mu0.to(device)
    mu1 = mu1.to(device)
    tau_true = mu1 - mu0
    return pehe(tau_hat, tau_true)


def evaluate_ipw(
    model: ACX,
    X: torch.Tensor,
    T: torch.Tensor,
    Y: torch.Tensor,
    propensity: torch.Tensor,
) -> float:
    """Return IPW tau risk for a dataset without counterfactuals.

    The function computes an inverse-propensity weighted pseudo-outcome that is
    unbiased for the true treatment effect and measures the PEHE between the
    model predictions and this pseudo-outcome.

    Args:
        model: Trained ``ACX`` model.
        X: Covariates ``(n, p)``.
        T: Treatment indicators ``(n, 1)``.
        Y: Observed outcomes ``(n, 1)``.
        propensity: Propensity scores ``(n, 1)`` for receiving treatment.

    Returns:
        Estimated square-root PEHE using IPW pseudo-outcomes.
    """

    (_, _, _, tau_hat), device = _forward_to_device(model, X)
    T = T.to(device)
    Y = Y.to(device)
    propensity = propensity.to(device)
    pseudo = Y * (T / propensity - (1.0 - T) / (1.0 - propensity))
    return pehe(tau_hat, pseudo)


def evaluate_dr(
    model: ACX,
    X: torch.Tensor,
    T: torch.Tensor,
    Y: torch.Tensor,
    propensity: torch.Tensor,
) -> float:
    """Return doubly-robust tau risk for observational datasets.

    This estimator compares the model's CATE predictions against a doubly robust
    pseudo-outcome constructed from outcome and propensity models. It reduces to
    PEHE when true counterfactual outcomes are available but can be applied when
    only observed outcomes are known.

    Args:
        model: Trained ``ACX`` model.
        X: Covariates ``(n, p)``.
        T: Treatment indicators ``(n, 1)``.
        Y: Observed outcomes ``(n, 1)``.
        propensity: Propensity scores ``(n, 1)`` for treatment.

    Returns:
        Estimated square-root PEHE using the doubly robust pseudo-outcomes.
    """

    (_, mu0_hat, mu1_hat, tau_hat), device = _forward_to_device(model, X)
    T = T.to(device)
    Y = Y.to(device)
    propensity = propensity.to(device)
    mu_hat = T * mu1_hat + (1.0 - T) * mu0_hat
    pseudo = (
        (T - propensity) / (propensity * (1.0 - propensity)) * (Y - mu_hat)
        + mu1_hat
        - mu0_hat
    )
    return pehe(tau_hat, pseudo)
