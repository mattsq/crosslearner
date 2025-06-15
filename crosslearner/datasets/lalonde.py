"""Loader for the original LaLonde dataset."""

import torch
from torch.utils.data import DataLoader, TensorDataset
from causaldata import nsw_mixtape


def get_lalonde_dataloader(batch_size: int = 256):
    """Return DataLoader for the LaLonde dataset."""
    df = nsw_mixtape.load_pandas().data
    y = torch.tensor(df["re78"].values, dtype=torch.float32).unsqueeze(-1)
    t = torch.tensor(df["treat"].values, dtype=torch.float32).unsqueeze(-1)
    x = torch.tensor(
        df.drop(columns=["re78", "treat", "data_id"]).values, dtype=torch.float32
    )
    loader = DataLoader(TensorDataset(x, t, y), batch_size=batch_size, shuffle=True)
    return loader, (None, None)
