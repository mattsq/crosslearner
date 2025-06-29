"""Loader for the ACIC 2016 benchmark dataset."""

from typing import Tuple

from causallib.datasets.data_loader import load_acic16

import torch
from torch.utils.data import DataLoader, TensorDataset


def get_acic2016_dataloader(
    seed: int = 0, batch_size: int = 256, *, data_dir: str | None = None
) -> Tuple[DataLoader, Tuple[torch.Tensor, torch.Tensor]]:
    """Return ACIC 2016 dataloader for the given replication index.

    Args:
        seed: Replication index from the benchmark dataset.
        batch_size: Mini-batch size.
        data_dir: Optional directory to cache the file.

    Returns:
        Tuple ``(loader, (mu0, mu1))`` with the dataloader and counterfactuals.
    """
    data = load_acic16(instance=seed + 1)
    X = torch.tensor(data.X.values, dtype=torch.float32)
    T = torch.tensor(data.a.values, dtype=torch.float32).unsqueeze(-1)
    Y = torch.tensor(data.y.values, dtype=torch.float32).unsqueeze(-1)
    mu0 = torch.tensor(data.po.iloc[:, 0].values, dtype=torch.float32).unsqueeze(-1)
    mu1 = torch.tensor(data.po.iloc[:, 1].values, dtype=torch.float32).unsqueeze(-1)
    dset = TensorDataset(X, T, Y)
    loader = DataLoader(dset, batch_size=batch_size, shuffle=True)
    return loader, (mu0, mu1)
