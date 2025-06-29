"""Loader for the Twins dataset."""

from typing import Tuple

import torch
from torch.utils.data import DataLoader, TensorDataset


def get_twins_dataloader(
    batch_size: int = 256, *, data_dir: str | None = None
) -> Tuple[DataLoader, Tuple[torch.Tensor, torch.Tensor]]:
    """Return dataloader for the Twins dataset."""
    from causaldata import twins as twins_data

    df = twins_data.load_pandas().data
    x = torch.tensor(
        df.drop(columns=["t", "yf", "mu0", "mu1"]).values, dtype=torch.float32
    )
    t = torch.tensor(df["t"].values, dtype=torch.float32).unsqueeze(-1)
    y = torch.tensor(df["yf"].values, dtype=torch.float32).unsqueeze(-1)
    mu0 = torch.tensor(df["mu0"].values, dtype=torch.float32).unsqueeze(-1)
    mu1 = torch.tensor(df["mu1"].values, dtype=torch.float32).unsqueeze(-1)
    loader = DataLoader(TensorDataset(x, t, y), batch_size=batch_size, shuffle=True)
    return loader, (mu0, mu1)
