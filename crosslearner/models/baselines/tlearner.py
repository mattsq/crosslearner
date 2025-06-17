"""Two-head T-learner baseline."""

import numpy as np

from .base import BaseTauLearner, make_mlp_regressor


class TLearner(BaseTauLearner):
    """Separate models for treated and control units."""

    def __init__(self, p: int) -> None:
        """Create the learner.

        Args:
            p: Number of covariates.
        """

        self.model_t = make_mlp_regressor()
        self.model_c = make_mlp_regressor()
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

    def _predict_mu1(self, X: np.ndarray) -> np.ndarray:
        """Predict outcomes under treatment."""
        return self.model_t.predict(X) if self._fitted_t else np.zeros(len(X))

    def _predict_mu0(self, X: np.ndarray) -> np.ndarray:
        """Predict outcomes under control."""
        return self.model_c.predict(X) if self._fitted_c else np.zeros(len(X))
