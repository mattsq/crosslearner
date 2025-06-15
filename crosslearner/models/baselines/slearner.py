"""Implementation of the single-head S-learner baseline."""

import numpy as np
import torch
from sklearn.neural_network import MLPRegressor


class SLearner:
    """Single-head learner that fits one model for both treatments."""

    def __init__(self, p: int) -> None:
        """Initialize the learner.

        Args:
            p: Number of covariates.
        """

        self.model = MLPRegressor(hidden_layer_sizes=(64, 64), max_iter=100)
        self.p = p

    def fit(self, X: np.ndarray, T: np.ndarray, Y: np.ndarray) -> None:
        """Fit the outcome model.

        Args:
            X: Covariate matrix of shape ``(n, p)``.
            T: Treatment indicators with shape ``(n, 1)``.
            Y: Observed outcomes with shape ``(n, 1)``.
        """

        XT = np.concatenate([X, T], axis=1)
        self.model.fit(XT, Y.ravel())

    def predict_tau(self, X: np.ndarray) -> torch.Tensor:
        """Predict treatment effects for new samples.

        Args:
            X: Covariate matrix with shape ``(n, p)``.

        Returns:
            Predicted treatment effects as a tensor of shape ``(n,)``.
        """

        X1 = np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)
        X0 = np.concatenate([X, np.zeros((X.shape[0], 1))], axis=1)
        mu1 = self.model.predict(X1)
        mu0 = self.model.predict(X0)
        return torch.tensor(mu1 - mu0, dtype=torch.float32)
