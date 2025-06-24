"""Synthetic data generator with configurable confounding."""

import torch
from torch.utils.data import TensorDataset, DataLoader


def get_confounding_dataloader(
    batch_size: int = 256,
    n: int = 8000,
    p: int = 10,
    confounding: float = 0.0,
    seed: int | None = None,
):
    """Return synthetic dataloader with adjustable confounding strength.

    Args:
        batch_size: Size of each mini-batch.
        n: Number of samples to generate.
        p: Number of covariates.
        confounding: Strength of the unobserved confounder.
        seed: Optional random seed for reproducibility.

    Returns:
        Tuple ``(loader, (mu0, mu1))`` with the ``DataLoader`` and true
        potential outcomes.
    """
    gen = torch.Generator().manual_seed(seed) if seed is not None else None
    X = torch.randn(n, p, generator=gen)
    U = torch.randn(n, generator=gen)
    logit = X[:, :2].sum(-1) + confounding * U
    T = torch.bernoulli(torch.sigmoid(logit), generator=gen).float()
    mu0 = X[:, 0] + confounding * U
    mu1 = mu0 + torch.tanh(X[:, 1] + confounding * U)
    Y = torch.where(T.bool(), mu1, mu0) + 0.5 * torch.randn(n, generator=gen)
    loader = DataLoader(
        TensorDataset(X, T.unsqueeze(-1), Y.unsqueeze(-1)),
        batch_size=batch_size,
        shuffle=True,
    )
    return loader, (mu0.unsqueeze(-1), mu1.unsqueeze(-1))
