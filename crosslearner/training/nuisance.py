"""Utilities for cross-fitted nuisance estimation."""

from typing import Iterable

import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold

from crosslearner.utils import set_seed


def _make_regressor(inp: int, hid: Iterable[int] = (64, 64)) -> nn.Sequential:
    """Return a simple fully connected regressor."""
    layers: list[nn.Module] = []
    d = inp
    for h in hid:
        layers += [nn.Linear(d, h), nn.ReLU()]
        d = h
    layers.append(nn.Linear(d, 1))
    return nn.Sequential(*layers)


def _make_propensity_net(inp: int, hid: Iterable[int] = (64, 64)) -> nn.Sequential:
    """Return a sigmoid-activated regressor for propensity scores."""
    net = _make_regressor(inp, hid)
    net.add_module("sigmoid", nn.Sigmoid())
    return net


def estimate_nuisances(
    X: torch.Tensor,
    T: torch.Tensor,
    Y: torch.Tensor,
    *,
    folds: int = 5,
    lr: float = 1e-3,
    batch: int = 256,
    propensity_epochs: int = 500,
    outcome_epochs: int = 3,
    early_stop: int = 10,
    device: str,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return cross-fitted propensity and outcome predictions.

    ``early_stop`` controls the number of epochs with no improvement
    before training is halted. Passing a non-positive value disables
    early stopping entirely.
    """
    bce = nn.BCELoss()
    mse = nn.MSELoss()

    patience = float("inf") if early_stop <= 0 else early_stop

    set_seed(seed)
    kfold = StratifiedKFold(folds, shuffle=True, random_state=seed)
    e_hat = torch.empty_like(T, device=device)
    mu0_hat = torch.empty_like(Y, device=device)
    mu1_hat = torch.empty_like(Y, device=device)

    for train_idx, val_idx in kfold.split(X.cpu(), T.cpu()):
        Xtr, Ttr, Ytr = X[train_idx], T[train_idx], Y[train_idx]
        Xva = X[val_idx]

        # split fold again for early stopping
        n = Xtr.shape[0]
        split = int(0.8 * n)
        X_train, X_val = Xtr[:split], Xtr[split:]
        T_train, T_val = Ttr[:split], Ttr[split:]
        Y_train, Y_val = Ytr[:split], Ytr[split:]

        prop = _make_propensity_net(X.shape[1]).to(device)
        opt_p = torch.optim.Adam(prop.parameters(), lr)
        best_state = {k: v.detach().clone() for k, v in prop.state_dict().items()}
        best_loss = float("inf")
        no_improve = 0
        for _ in range(propensity_epochs):
            pred = prop(X_train)
            loss = bce(pred, T_train)
            opt_p.zero_grad()
            loss.backward()
            opt_p.step()
            val_loss = bce(prop(X_val), T_val).item()
            if val_loss < best_loss - 1e-6:
                best_loss = val_loss
                best_state = {
                    k: v.detach().clone() for k, v in prop.state_dict().items()
                }
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    break
        prop.load_state_dict(best_state)
        e_hat[val_idx] = prop(Xva).detach()

        mu0 = _make_regressor(X.shape[1]).to(device)
        mu1 = _make_regressor(X.shape[1]).to(device)
        opt_mu = torch.optim.Adam(list(mu0.parameters()) + list(mu1.parameters()), lr)
        ds = torch.utils.data.TensorDataset(X_train, T_train, Y_train)
        val_ds = torch.utils.data.TensorDataset(X_val, T_val, Y_val)
        loader = torch.utils.data.DataLoader(ds, batch_size=batch, shuffle=True)
        best_mu0 = {k: v.detach().clone() for k, v in mu0.state_dict().items()}
        best_mu1 = {k: v.detach().clone() for k, v in mu1.state_dict().items()}
        best_val = float("inf")
        no_improve = 0
        for _ in range(outcome_epochs):
            for xb, tb, yb in loader:
                pred0, pred1 = mu0(xb), mu1(xb)
                loss = torch.tensor(0.0, device=device)
                mask0 = tb == 0
                mask1 = tb == 1
                if mask0.any():
                    loss = loss + mse(pred0[mask0], yb[mask0])
                if mask1.any():
                    loss = loss + mse(pred1[mask1], yb[mask1])
                opt_mu.zero_grad()
                loss.backward()
                opt_mu.step()
            with torch.no_grad():
                val_loss = 0.0
                count = 0
                for xb, tb, yb in torch.utils.data.DataLoader(val_ds, batch_size=batch):
                    pred0, pred1 = mu0(xb), mu1(xb)
                    mask0 = tb == 0
                    mask1 = tb == 1
                    loss = torch.tensor(0.0, device=device)
                    if mask0.any():
                        loss = loss + mse(pred0[mask0], yb[mask0])
                    if mask1.any():
                        loss = loss + mse(pred1[mask1], yb[mask1])
                    val_loss += loss.item()
                    count += 1
                val_loss /= max(count, 1)
                if val_loss < best_val - 1e-6:
                    best_val = val_loss
                    best_mu0 = {
                        k: v.detach().clone() for k, v in mu0.state_dict().items()
                    }
                    best_mu1 = {
                        k: v.detach().clone() for k, v in mu1.state_dict().items()
                    }
                    no_improve = 0
                else:
                    no_improve += 1
                    if no_improve >= patience:
                        break
        mu0.load_state_dict(best_mu0)
        mu1.load_state_dict(best_mu1)
        mu0_hat[val_idx] = mu0(Xva).detach()
        mu1_hat[val_idx] = mu1(Xva).detach()

    return e_hat, mu0_hat, mu1_hat
