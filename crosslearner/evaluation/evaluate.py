"""Evaluation utilities."""

import torch
from crosslearner.evaluation.metrics import pehe
from crosslearner.models.acx import ACX


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

    model.eval()
    with torch.no_grad():
        _, _, _, tau_hat = model(X)
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

    model.eval()
    with torch.no_grad():
        _, _, _, tau_hat = model(X)
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

    model.eval()
    with torch.no_grad():
        _, mu0_hat, mu1_hat, tau_hat = model(X)
    mu_hat = T * mu1_hat + (1.0 - T) * mu0_hat
    pseudo = (
        (T - propensity) / (propensity * (1.0 - propensity)) * (Y - mu_hat)
        + mu1_hat
        - mu0_hat
    )
    return pehe(tau_hat, pseudo)
