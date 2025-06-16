"""Propensity score estimation helpers."""

from __future__ import annotations

import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold


def estimate_propensity(
    X: torch.Tensor, T: torch.Tensor, *, folds: int = 5, seed: int = 0
) -> torch.Tensor:
    """Return cross-fitted propensity scores via logistic regression.

    Args:
        X: Covariates ``(n, p)``.
        T: Binary treatment indicators ``(n, 1)`` or ``(n,)``.
        folds: Number of cross-fitting folds.
        seed: Random seed controlling the CV split.

    Returns:
        Estimated propensity scores ``(n, 1)``.
    """

    X_np = X.detach().cpu().numpy()
    T_np = T.view(-1).detach().cpu().numpy()
    kf = KFold(n_splits=folds, shuffle=True, random_state=seed)
    prop = torch.empty(X.shape[0], dtype=torch.float32)
    for train_idx, test_idx in kf.split(X_np):
        model = LogisticRegression(max_iter=1000)
        model.fit(X_np[train_idx], T_np[train_idx])
        pred = model.predict_proba(X_np[test_idx])[:, 1]
        prop[test_idx] = torch.tensor(pred, dtype=torch.float32)
    return prop.unsqueeze(-1)
