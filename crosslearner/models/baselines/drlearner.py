"""Doubly Robust (DR) learner baseline implementation."""

from __future__ import annotations

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

from .base import make_mlp_regressor


class DRLearner:
    """Doubly Robust meta-learner."""

    def __init__(self, p: int) -> None:
        """Create the learner.

        Args:
            p: Number of covariates.
        """

        self.model_mu0 = make_mlp_regressor()
        self.model_mu1 = make_mlp_regressor()
        self.model_tau = make_mlp_regressor()
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
        mask_t = T == 1
        mask_c = ~mask_t

        # fit outcome models on their respective subsets
        if mask_c.any():
            self.model_mu0.fit(X[mask_c], Y[mask_c].ravel())
        else:
            self.model_mu0.fit(X, Y.ravel())
        if mask_t.any():
            self.model_mu1.fit(X[mask_t], Y[mask_t].ravel())
        else:
            self.model_mu1.fit(X, Y.ravel())
        if np.unique(T).size < 2:
            e_hat = np.full_like(T, fill_value=T.mean(), dtype=float)
        else:
            self.model_e.fit(X, T)
            e_hat = self.model_e.predict_proba(X)[:, 1]
        e_hat = np.clip(e_hat, 1e-3, 1 - 1e-3)
        mu0_hat = self.model_mu0.predict(X)
        mu1_hat = self.model_mu1.predict(X)
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
