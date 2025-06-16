"""Implementation of the X-learner baseline."""

import numpy as np
import torch
from sklearn.neural_network import MLPRegressor

from .tlearner import TLearner


class XLearner:
    """Meta-learner that combines T-learner and imputed pseudo-outcomes."""

    def __init__(self, p: int) -> None:
        """Create the learner.

        Args:
            p: Number of covariates.
        """

        self.t = TLearner(p)
        self.model_tau_t = MLPRegressor(hidden_layer_sizes=(64, 64), max_iter=100)
        self.model_tau_c = MLPRegressor(hidden_layer_sizes=(64, 64), max_iter=100)
        self.p = p
        self._fitted_tau_t = False
        self._fitted_tau_c = False

    def fit(self, X: np.ndarray, T: np.ndarray, Y: np.ndarray) -> None:
        """Fit the base learners and pseudo-outcome regressors.

        Args:
            X: Covariates ``(n, p)``.
            T: Treatment indicators ``(n, 1)``.
            Y: Outcomes ``(n, 1)``.
        """

        self.t.fit(X, T, Y)
        Xt = X[T.squeeze() == 1]
        Yt = Y[T.squeeze() == 1]
        Xc = X[T.squeeze() == 0]
        Yc = Y[T.squeeze() == 0]

        mu0_t = (
            self.t.model_c.predict(Xt)
            if self.t._fitted_c and len(Xt)
            else np.zeros(len(Xt))
        )
        d1 = Yt.ravel() - mu0_t
        mu1_c = (
            self.t.model_t.predict(Xc)
            if self.t._fitted_t and len(Xc)
            else np.zeros(len(Xc))
        )
        d0 = mu1_c - Yc.ravel()

        if len(Xt):
            self.model_tau_t.fit(Xt, d1)
            self._fitted_tau_t = True
        else:
            self._fitted_tau_t = False
        if len(Xc):
            self.model_tau_c.fit(Xc, d0)
            self._fitted_tau_c = True
        else:
            self._fitted_tau_c = False
        self.prop = T.mean()

    def predict_tau(self, X: np.ndarray) -> torch.Tensor:
        """Predict treatment effects using imputed models.

        Args:
            X: Covariate matrix ``(n, p)``.

        Returns:
            Tensor of predicted treatment effects.
        """

        tau_t = self.model_tau_t.predict(X) if self._fitted_tau_t else np.zeros(len(X))
        tau_c = self.model_tau_c.predict(X) if self._fitted_tau_c else np.zeros(len(X))
        return torch.tensor(
            (1 - self.prop) * tau_t + self.prop * tau_c, dtype=torch.float32
        )
