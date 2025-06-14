import numpy as np
import torch
from sklearn.neural_network import MLPRegressor


class TLearner:
    def __init__(self, p: int):
        self.model_t = MLPRegressor(hidden_layer_sizes=(64, 64), max_iter=100)
        self.model_c = MLPRegressor(hidden_layer_sizes=(64, 64), max_iter=100)
        self.p = p

    def fit(self, X: np.ndarray, T: np.ndarray, Y: np.ndarray):
        Xt = X[T.squeeze() == 1]
        Yt = Y[T.squeeze() == 1]
        Xc = X[T.squeeze() == 0]
        Yc = Y[T.squeeze() == 0]
        if len(Xt):
            self.model_t.fit(Xt, Yt.ravel())
        if len(Xc):
            self.model_c.fit(Xc, Yc.ravel())

    def predict_tau(self, X: np.ndarray) -> torch.Tensor:
        mu1 = self.model_t.predict(X)
        mu0 = self.model_c.predict(X)
        return torch.tensor(mu1 - mu0, dtype=torch.float32)
