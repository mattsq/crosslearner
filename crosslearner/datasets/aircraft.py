"""Synthetic aircraft dataset based on the Breguet range equation."""

import torch
from torch.utils.data import DataLoader, TensorDataset


def get_aircraft_dataloader(
    batch_size: int = 256, n: int = 8000, seed: int | None = None
):
    """Return ``DataLoader`` with aircraft performance data.

    Covariates are aircraft weight, cruise range, thrust specific fuel
    consumption, lift-to-drag ratio and cruise speed. Treatment models a
    modification that increases the lift-to-drag ratio by 10% which in
    turn reduces fuel burn according to the simplified Breguet range
    equation.

    Args:
        batch_size: Mini-batch size.
        n: Number of samples.
        seed: Optional random seed for reproducibility.

    Returns:
        Tuple ``(loader, (mu0, mu1))`` with true potential outcomes.
    """
    gen = torch.Generator().manual_seed(seed) if seed is not None else None

    weight = 40000 + 40000 * torch.rand(n, generator=gen)  # kg
    rng = 1000 + 4000 * torch.rand(n, generator=gen)  # km
    tsfc = 0.5 + 0.2 * torch.rand(n, generator=gen)  # 1/h
    ld_ratio = 10 + 10 * torch.rand(n, generator=gen)
    speed = 200 + 50 * torch.rand(n, generator=gen)  # m/s
    X = torch.stack([weight, rng, tsfc, ld_ratio, speed], dim=1)

    logit = (
        1e-4 * (weight - 60000)
        + 0.2 * (ld_ratio - 15) / 10
        - 5.0 * (tsfc - 0.6)
        + 0.0002 * (rng - 2500)
    )
    pi = torch.sigmoid(logit)
    T = torch.bernoulli(pi, generator=gen).float()

    c = tsfc / 3600  # convert to 1/s
    R = rng * 1000  # convert to m
    mu0_val = weight * (1 - torch.exp(-R * c / (speed * ld_ratio)))
    mu1_val = weight * (1 - torch.exp(-R * c / (speed * (ld_ratio * 1.1))))
    mu0 = mu0_val.unsqueeze(-1)
    mu1 = mu1_val.unsqueeze(-1)

    Y = torch.where(T.bool(), mu1_val, mu0_val)
    Y = Y + 0.05 * Y * torch.randn(n, generator=gen)

    loader = DataLoader(
        TensorDataset(X, T.unsqueeze(-1), Y.unsqueeze(-1)),
        batch_size=batch_size,
        shuffle=True,
    )
    return loader, (mu0, mu1)
