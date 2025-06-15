"""Doubly Robust (DR) learner baseline implementation."""

from __future__ import annotations

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPRegressor


class DRLearner:
    """Doubly Robust meta-learner."""

    def __init__(self, p: int) -> None:
        """Create the learner.

        Args:
            p: Number of covariates.
        """

        self.model_mu0 = MLPRegressor(hidden_layer_sizes=(64, 64), max_iter=100)
        self.model_mu1 = MLPRegressor(hidden_layer_sizes=(64, 64), max_iter=100)
        self.model_tau = MLPRegressor(hidden_layer_sizes=(64, 64), max_iter=100)
        self.model_e = LogisticRegression(max_iter=100)
        self.p = p

    def fit(self, X: np.ndarray, T: np.ndarray, Y: np.ndarray) -> None:
        """Fit outcome, propensity and pseudo-outcome models.

        Args:
            X: Covariates ``(n, p)``.
            T: Treatment indicators ``(n, 1)``.
            Y: Outcomes ``(n, 1)``.
        """

        T = T.ravel()
        self.model_mu0.fit(X, Y.ravel())
        self.model_mu1.fit(X, Y.ravel())
        self.model_e.fit(X, T)
        mu0_hat = self.model_mu0.predict(X)
        mu1_hat = self.model_mu1.predict(X)
        e_hat = self.model_e.predict_proba(X)[:, 1]
        tau_tilde = (
            mu1_hat
            - mu0_hat
            + (T - e_hat)
            / (e_hat * (1 - e_hat))
            * (Y.ravel() - np.where(T == 1, mu1_hat, mu0_hat))
        )
        self.model_tau.fit(X, tau_tilde)

    def predict_tau(self, X: np.ndarray) -> torch.Tensor:
        """Predict treatment effects.

        Args:
            X: Covariate matrix ``(n, p)``.

        Returns:
            Predicted treatment effects.
        """

        tau = self.model_tau.predict(X)
        return torch.tensor(tau, dtype=torch.float32)
