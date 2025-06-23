"""Synthetic dataset emphasising discriminator tricks."""

import torch
from torch.utils.data import DataLoader, TensorDataset


def get_tricky_dataloader(
    batch_size: int = 256,
    n: int = 1000,
    p: int = 8,
    confounding: float = 1.0,
    seed: int | None = None,
):
    """Return DataLoader for a small imbalanced synthetic dataset.

    The setup deliberately introduces strong confounding and class imbalance so
    the discriminator quickly overfits without stabilisation tricks.
    """
    gen = torch.Generator().manual_seed(seed) if seed is not None else None
    X = torch.randn(n, p, generator=gen)
    U = torch.randn(n, generator=gen)

    logit = 2.5 * X[:, 0] - 2.5 * X[:, 1] + confounding * U - 1.0
    T = torch.bernoulli(torch.sigmoid(logit), generator=gen).float()

    mu0 = X[:, 0] - X[:, 1] + 0.5 * confounding * U
    mu1 = mu0 + torch.tanh(X[:, 2]) + 1.0 * (X[:, 3] > 0).float()

    Y = torch.where(T.bool(), mu1, mu0) + 0.3 * torch.randn(n, generator=gen)

    loader = DataLoader(
        TensorDataset(X, T.unsqueeze(-1), Y.unsqueeze(-1)),
        batch_size=batch_size,
        shuffle=True,
    )
    return loader, (mu0.unsqueeze(-1), mu1.unsqueeze(-1))
