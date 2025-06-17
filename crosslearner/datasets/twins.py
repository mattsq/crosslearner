"""Loader for the Twins dataset."""

import os
from typing import Tuple

from .utils import download_if_missing

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

URL_TWINS = (
    "https://raw.githubusercontent.com/py-why/BenchmarkDatasets/master/twins/twins.npz"
)


def get_twins_dataloader(
    batch_size: int = 256, *, data_dir: str | None = None
) -> Tuple[DataLoader, Tuple[torch.Tensor, torch.Tensor]]:
    """Return dataloader for the Twins dataset."""
    data_dir = data_dir or os.path.join(os.path.dirname(__file__), "_data")
    os.makedirs(data_dir, exist_ok=True)
    fpath = download_if_missing(URL_TWINS, os.path.join(data_dir, "twins.npz"))
    data = np.load(fpath)
    x = torch.tensor(data["x"], dtype=torch.float32)
    t = torch.tensor(data["t"], dtype=torch.float32).unsqueeze(-1)
    y = torch.tensor(data["yf"], dtype=torch.float32).unsqueeze(-1)
    mu0 = torch.tensor(data["mu0"], dtype=torch.float32).unsqueeze(-1)
    mu1 = torch.tensor(data["mu1"], dtype=torch.float32).unsqueeze(-1)
    loader = DataLoader(TensorDataset(x, t, y), batch_size=batch_size, shuffle=True)
    return loader, (mu0, mu1)
