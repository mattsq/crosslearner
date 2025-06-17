"""Base class for simple meta-learners."""

from __future__ import annotations

import numpy as np
import torch
from sklearn.neural_network import MLPRegressor


def make_mlp_regressor(
    *, hidden_layer_sizes: tuple[int, ...] = (64, 64), max_iter: int = 100, **kwargs
) -> MLPRegressor:
    """Return ``MLPRegressor`` with shared default parameters."""

    return MLPRegressor(
        hidden_layer_sizes=hidden_layer_sizes, max_iter=max_iter, **kwargs
    )


class BaseTauLearner:
    """Base class for learners that estimate treatment effects."""

    p: int

    def predict_tau(self, X: np.ndarray) -> torch.Tensor:
        """Predict treatment effects as ``mu1 - mu0``.

        Args:
            X: Covariate matrix ``(n, p)``.

        Returns:
            Tensor of predicted treatment effects.
        """
        mu1 = self._predict_mu1(X)
        mu0 = self._predict_mu0(X)
        return torch.tensor(mu1 - mu0, dtype=torch.float32)

    def _predict_mu1(self, X: np.ndarray) -> np.ndarray:
        """Predict outcomes under treatment."""
        raise NotImplementedError

    def _predict_mu0(self, X: np.ndarray) -> np.ndarray:
        """Predict outcomes under control."""
        raise NotImplementedError
