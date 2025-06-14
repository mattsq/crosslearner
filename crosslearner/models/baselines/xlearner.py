import numpy as np
import torch
from sklearn.neural_network import MLPRegressor

from .tlearner import TLearner


class XLearner:
    def __init__(self, p: int):
        self.t = TLearner(p)
        self.model_tau_t = MLPRegressor(hidden_layer_sizes=(64, 64), max_iter=100)
        self.model_tau_c = MLPRegressor(hidden_layer_sizes=(64, 64), max_iter=100)
        self.p = p

    def fit(self, X: np.ndarray, T: np.ndarray, Y: np.ndarray):
        self.t.fit(X, T, Y)
        Xt = X[T.squeeze() == 1]
        Yt = Y[T.squeeze() == 1]
        Xc = X[T.squeeze() == 0]
        Yc = Y[T.squeeze() == 0]
        mu0_t = self.t.model_c.predict(Xt) if len(Xt) else np.zeros(len(Xt))
        d1 = Yt.ravel() - mu0_t
        mu1_c = self.t.model_t.predict(Xc) if len(Xc) else np.zeros(len(Xc))
        d0 = mu1_c - Yc.ravel()
        if len(Xt):
            self.model_tau_t.fit(Xt, d1)
        if len(Xc):
            self.model_tau_c.fit(Xc, d0)
        self.prop = T.mean()

    def predict_tau(self, X: np.ndarray) -> torch.Tensor:
        tau_t = self.model_tau_t.predict(X)
        tau_c = self.model_tau_c.predict(X)
        return torch.tensor(
            (1 - self.prop) * tau_t + self.prop * tau_c, dtype=torch.float32
        )
