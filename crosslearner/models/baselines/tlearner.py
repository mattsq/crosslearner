"""Two-head T-learner baseline."""

import numpy as np
import torch
from sklearn.neural_network import MLPRegressor


class TLearner:
    """Separate models for treated and control units."""

    def __init__(self, p: int) -> None:
        """Create the learner.

        Args:
            p: Number of covariates.
        """

        self.model_t = MLPRegressor(hidden_layer_sizes=(64, 64), max_iter=100)
        self.model_c = MLPRegressor(hidden_layer_sizes=(64, 64), max_iter=100)
        self.p = p
        self._fitted_t = False
        self._fitted_c = False

    def fit(self, X: np.ndarray, T: np.ndarray, Y: np.ndarray) -> None:
        """Fit treatment and control models separately.

        Args:
            X: Covariate matrix ``(n, p)``.
            T: Treatment indicators ``(n, 1)``.
            Y: Outcomes ``(n, 1)``.
        """

        Xt = X[T.squeeze() == 1]
        Yt = Y[T.squeeze() == 1]
        Xc = X[T.squeeze() == 0]
        Yc = Y[T.squeeze() == 0]
        if len(Xt):
            self.model_t.fit(Xt, Yt.ravel())
            self._fitted_t = True
        else:
            self._fitted_t = False
        if len(Xc):
            self.model_c.fit(Xc, Yc.ravel())
            self._fitted_c = True
        else:
            self._fitted_c = False

    def predict_tau(self, X: np.ndarray) -> torch.Tensor:
        """Predict treatment effects.

        Args:
            X: Covariate matrix ``(n, p)``.

        Returns:
            Predicted treatment effects.
        """

        mu1 = self.model_t.predict(X) if self._fitted_t else np.zeros(len(X))
        mu0 = self.model_c.predict(X) if self._fitted_c else np.zeros(len(X))
        return torch.tensor(mu1 - mu0, dtype=torch.float32)
