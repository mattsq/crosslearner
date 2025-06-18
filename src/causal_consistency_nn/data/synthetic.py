"""Synthetic dataset generator following the SCM described in Prompt.txt."""

from __future__ import annotations

from typing import Tuple

import torch
from torch.utils.data import DataLoader, TensorDataset

from ..config import SyntheticDataConfig


def generate_synthetic(
    config: SyntheticDataConfig,
) -> Tuple[TensorDataset, Tuple[torch.Tensor, torch.Tensor]]:
    """Generate synthetic data according to the simple SCM.

    Returns a ``TensorDataset`` containing ``X``, ``T`` and the possibly
    missing outcome ``Y`` as well as the true potential outcomes.
    """
    gen = (
        torch.Generator().manual_seed(config.seed) if config.seed is not None else None
    )

    X = torch.randn(config.n_samples, config.p, generator=gen)
    pi = torch.sigmoid(X[:, :2].sum(-1))
    T = torch.bernoulli(pi, generator=gen).float()

    mu0 = (X[:, 0] - X[:, 1]).unsqueeze(-1)
    mu1 = mu0 + 2.0 * torch.tanh(X[:, 2]).unsqueeze(-1)

    Y = torch.where(T.bool().unsqueeze(-1), mu1, mu0)
    Y = Y + config.noise * torch.randn(Y.shape, generator=gen)

    if config.missing_y_prob > 0.0:
        mask = torch.rand(Y.shape, generator=gen) < config.missing_y_prob
        Y = Y.clone()
        Y[mask] = float("nan")

    dataset = TensorDataset(X, T.unsqueeze(-1), Y)
    return dataset, (mu0, mu1)


def get_synthetic_dataloader(
    config: SyntheticDataConfig,
    *,
    batch_size: int = 256,
    shuffle: bool = True,
) -> Tuple[DataLoader, Tuple[torch.Tensor, torch.Tensor]]:
    """Return a ``DataLoader`` for the generated synthetic dataset."""

    dset, mu = generate_synthetic(config)
    loader = DataLoader(dset, batch_size=batch_size, shuffle=shuffle)
    return loader, mu
