from typing import Tuple
import numpy as np
import torch
from sklearn.neural_network import MLPRegressor


class SLearner:
    def __init__(self, p: int):
        self.model = MLPRegressor(hidden_layer_sizes=(64, 64), max_iter=100)
        self.p = p

    def fit(self, X: np.ndarray, T: np.ndarray, Y: np.ndarray):
        XT = np.concatenate([X, T], axis=1)
        self.model.fit(XT, Y.ravel())

    def predict_tau(self, X: np.ndarray) -> torch.Tensor:
        X1 = np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)
        X0 = np.concatenate([X, np.zeros((X.shape[0], 1))], axis=1)
        mu1 = self.model.predict(X1)
        mu0 = self.model.predict(X0)
        return torch.tensor(mu1 - mu0, dtype=torch.float32)
