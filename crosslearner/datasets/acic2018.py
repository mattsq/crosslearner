"""Loader for the ACIC 2018 benchmark dataset."""

import os
from typing import Tuple

from .utils import download_if_missing

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

URL_2018 = "https://raw.githubusercontent.com/py-why/BenchmarkDatasets/master/acic2018/acic2018.npz"


def get_acic2018_dataloader(
    seed: int = 0, batch_size: int = 256, *, data_dir: str | None = None
) -> Tuple[DataLoader, Tuple[torch.Tensor, torch.Tensor]]:
    """Return ACIC 2018 dataloader for the given replication index.

    Args:
        seed: Replication index from the benchmark dataset.
        batch_size: Mini-batch size.
        data_dir: Optional directory to cache the file.

    Returns:
        Tuple ``(loader, (mu0, mu1))`` with dataloader and counterfactuals.
    """
    data_dir = data_dir or os.path.join(os.path.dirname(__file__), "_data")
    os.makedirs(data_dir, exist_ok=True)
    fpath = download_if_missing(URL_2018, os.path.join(data_dir, "acic2018.npz"))
    data = np.load(fpath)
    X = data["x"][:, :, seed]
    T = data["t"][:, seed]
    Y = data["yf"][:, seed]
    mu0 = data["mu0"][:, seed]
    mu1 = data["mu1"][:, seed]
    loader = DataLoader(
        TensorDataset(
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(T, dtype=torch.float32).unsqueeze(-1),
            torch.tensor(Y, dtype=torch.float32).unsqueeze(-1),
        ),
        batch_size=batch_size,
        shuffle=True,
    )
    return loader, (
        torch.tensor(mu0, dtype=torch.float32).unsqueeze(-1),
        torch.tensor(mu1, dtype=torch.float32).unsqueeze(-1),
    )
