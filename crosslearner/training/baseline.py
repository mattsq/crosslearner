from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPRegressor
from torch.utils.data import DataLoader

from crosslearner.evaluation.propensity import estimate_propensity
from crosslearner.utils import set_seed


@dataclass
class BaselineConfig:
    p: int
    folds: int = 5
    epochs: int = 1
    lr: float = 1e-3
    epsilon_prop: float = 1e-3
    seed: int = 0


def _split_indices(n: int, folds: int, seed: int) -> Tuple[torch.Tensor, torch.Tensor]:
    kf = KFold(n_splits=folds, shuffle=True, random_state=seed)
    train_idx, val_idx = next(kf.split(range(n)))
    return torch.tensor(train_idx), torch.tensor(val_idx)


def pseudo_outcome(
    mu0: torch.Tensor,
    mu1: torch.Tensor,
    T: torch.Tensor,
    Y: torch.Tensor,
    e_hat: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    e = e_hat.clamp(min=eps, max=1 - eps)
    mu = T * mu1 + (1.0 - T) * mu0
    return mu1 - mu0 + (T - e) / (e * (1.0 - e)) * (Y - mu)


def train_baseline(
    loader: DataLoader,
    mu0: torch.Tensor,
    mu1: torch.Tensor,
    cfg: BaselineConfig,
    device: str = "cpu",
) -> Tuple[MLPRegressor, list[float]]:
    set_seed(cfg.seed)
    device_t = torch.device(device)
    X = torch.cat([b[0] for b in loader]).to(device_t)
    T = torch.cat([b[1] for b in loader]).to(device_t)
    Y = torch.cat([b[2] for b in loader]).to(device_t)
    mu0 = mu0.to(device_t)
    mu1 = mu1.to(device_t)

    e_hat = estimate_propensity(
        X.cpu(), T.cpu(), folds=cfg.folds, seed=cfg.seed, eps=cfg.epsilon_prop
    )
    e_hat = e_hat.to(device_t)
    D = pseudo_outcome(mu0, mu1, T, Y, e_hat, cfg.epsilon_prop)

    n = X.size(0)
    train_idx, val_idx = _split_indices(n, cfg.folds, cfg.seed)
    X_train = X[train_idx].cpu().numpy()
    y_train = D[train_idx].cpu().numpy().ravel()
    X_val = X[val_idx].cpu().numpy()
    y_val = D[val_idx].cpu().numpy().ravel()

    val_pred0 = np.zeros_like(y_val)
    init_loss = float(np.sqrt(np.mean((val_pred0 - y_val) ** 2)))

    model = MLPRegressor(
        hidden_layer_sizes=(256, 128, 64),
        max_iter=cfg.epochs,
        random_state=cfg.seed,
        learning_rate_init=cfg.lr,
    )
    model.fit(X_train, y_train)
    val_pred = model.predict(X_val)
    final_loss = float(np.sqrt(np.mean((val_pred - y_val) ** 2)))
    return model, [init_loss, final_loss]
