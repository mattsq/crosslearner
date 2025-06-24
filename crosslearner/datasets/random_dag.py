"""Random DAG-based synthetic dataset generator."""

from __future__ import annotations
import random
from typing import Callable

import torch
from torch.utils.data import DataLoader, TensorDataset


_Function = Callable[[torch.Tensor], torch.Tensor]


_FUNCTIONS: list[_Function] = [
    lambda x: x,
    torch.tanh,
    torch.sin,
    lambda x: x**2,
]


_DISTROS = ["normal", "uniform", "exponential"]


def _sample_distribution(
    name: str, n: int, gen: torch.Generator | None
) -> torch.Tensor:
    if name == "normal":
        return torch.randn(n, generator=gen)
    if name == "uniform":
        return 2 * torch.rand(n, generator=gen) - 1
    if name == "exponential":
        r = torch.rand(n, generator=gen)
        return -torch.log1p(-r)
    raise ValueError(f"unknown distribution {name}")


def get_random_dag_dataloader(
    batch_size: int = 256,
    n: int = 8000,
    p: int = 10,
    seed: int | None = None,
) -> tuple[DataLoader, tuple[torch.Tensor, torch.Tensor]]:
    """Return DataLoader for a random acyclic structural equation model."""
    gen = torch.Generator().manual_seed(seed) if seed is not None else None
    if seed is not None:
        random.seed(seed)

    order = list(range(p))
    random.shuffle(order)
    X: list[torch.Tensor] = [torch.zeros(n) for _ in range(p)]
    for idx, j in enumerate(order):
        dist = random.choice(_DISTROS)
        x = _sample_distribution(dist, n, gen)
        for i in order[:idx]:
            if torch.rand(1, generator=gen).item() < 0.3:
                func = random.choice(_FUNCTIONS)
                bias = 0.1 * torch.randn(1, generator=gen)
                weight = torch.randn(1, generator=gen)
                x = x + weight * func(X[i] + bias)
        X[j] = x
    X_tensor = torch.stack(X, dim=1)

    # Treatment model
    logit = 0.1 * torch.randn(1, generator=gen)
    for j in range(p):
        if torch.rand(1, generator=gen).item() < 0.5:
            func = random.choice(_FUNCTIONS)
            bias = 0.1 * torch.randn(1, generator=gen)
            weight = torch.randn(1, generator=gen)
            logit = logit + weight * func(X[j] + bias)
    pi = torch.sigmoid(logit)
    T = torch.bernoulli(pi, generator=gen).float()

    # Outcome model
    base = 0.1 * torch.randn(1, generator=gen)
    for j in range(p):
        if torch.rand(1, generator=gen).item() < 0.5:
            func = random.choice(_FUNCTIONS)
            bias = 0.1 * torch.randn(1, generator=gen)
            weight = torch.randn(1, generator=gen)
            base = base + weight * func(X[j] + bias)

    func_ty = random.choice(_FUNCTIONS)
    bias_ty = 0.1 * torch.randn(1, generator=gen)
    weight_ty = torch.randn(1, generator=gen)

    mu0_val = base + weight_ty * func_ty(torch.zeros_like(T) + bias_ty)
    mu1_val = base + weight_ty * func_ty(torch.ones_like(T) + bias_ty)

    Y = torch.where(T.bool(), mu1_val, mu0_val)
    Y = Y + 0.1 * torch.randn(n, generator=gen)

    loader = DataLoader(
        TensorDataset(X_tensor, T.unsqueeze(-1), Y.unsqueeze(-1)),
        batch_size=batch_size,
        shuffle=True,
    )
    return loader, (mu0_val.unsqueeze(-1), mu1_val.unsqueeze(-1))
