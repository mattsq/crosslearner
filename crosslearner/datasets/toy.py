"""Synthetic toy dataset."""

import torch
from torch.utils.data import TensorDataset, DataLoader


def get_toy_dataloader(
    batch_size: int = 256,
    n: int = 8000,
    p: int = 10,
    seed: int | None = None,
) -> tuple[DataLoader, tuple[torch.Tensor, torch.Tensor]]:
    """Return ``DataLoader`` with simple synthetic data.

    Args:
        batch_size: Size of the mini-batches.
        n: Number of samples.
        p: Number of covariates.

    Returns:
        Tuple ``(loader, (mu0, mu1))`` where ``mu0`` and ``mu1`` are the true
        potential outcomes.
    """
    gen = torch.Generator().manual_seed(seed) if seed is not None else None
    X = torch.randn(n, p, generator=gen)
    pi = torch.sigmoid(X[:, :2].sum(-1))
    T = torch.bernoulli(pi, generator=gen).float()
    mu0 = (X[:, 0] - X[:, 1]).unsqueeze(-1)
    mu1 = mu0 + 2.0 * torch.tanh(X[:, 2]).unsqueeze(-1)
    t_unsq = T.unsqueeze(-1)
    Y = torch.where(t_unsq.bool(), mu1, mu0) + 0.5 * torch.randn(n, 1, generator=gen)

    dset = TensorDataset(X, t_unsq, Y)
    loader = DataLoader(dset, batch_size=batch_size, shuffle=True)
    return loader, (mu0, mu1)
