import torch
from torch.utils.data import TensorDataset, DataLoader


def get_complex_dataloader(
    batch_size: int = 256, n: int = 8000, p: int = 20, seed: int | None = None
):
    """Return DataLoader with a more complex synthetic dataset."""
    if seed is not None:
        gen = torch.Generator().manual_seed(seed)
    else:
        gen = None
    X = torch.randn(n, p, generator=gen)
    logit = X[:, :3].prod(-1) + torch.sin(X[:, 3]) - X[:, 4] ** 2
    pi = torch.sigmoid(logit)
    T = torch.bernoulli(pi, generator=gen).float()
    mu0 = X[:, 0] + torch.cos(X[:, 1]) + 0.5 * (X[:, 2] ** 2)
    mu1 = mu0 + torch.tanh(X[:, 3] + X[:, 4]) + X[:, 5]
    Y = torch.where(T.bool(), mu1, mu0) + 0.5 * torch.randn(n, generator=gen)
    return DataLoader(
        TensorDataset(X, T.unsqueeze(-1), Y.unsqueeze(-1)),
        batch_size=batch_size,
        shuffle=True,
    ), (
        mu0.unsqueeze(-1),
        mu1.unsqueeze(-1),
    )
