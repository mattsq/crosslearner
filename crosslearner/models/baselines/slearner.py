"""Implementation of the single-head S-learner baseline."""

import numpy as np

from .base import BaseTauLearner, make_mlp_regressor


class SLearner(BaseTauLearner):
    """Single-head learner that fits one model for both treatments."""

    def __init__(self, p: int) -> None:
        """Initialize the learner.

        Args:
            p: Number of covariates.
        """

        self.model = make_mlp_regressor()
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

    def _predict_mu1(self, X: np.ndarray) -> np.ndarray:
        """Predict outcomes under treatment."""
        X1 = np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)
        return self.model.predict(X1)

    def _predict_mu0(self, X: np.ndarray) -> np.ndarray:
        """Predict outcomes under control."""
        X0 = np.concatenate([X, np.zeros((X.shape[0], 1))], axis=1)
        return self.model.predict(X0)
