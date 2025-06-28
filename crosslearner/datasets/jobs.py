"""Jobs dataset used for off-policy evaluation."""

import torch
from torch.utils.data import DataLoader, TensorDataset
from causaldata import nsw_mixtape


def get_jobs_dataloader(batch_size: int = 256):
    """Return DataLoader for the Jobs training dataset.

    Args:
        batch_size: Mini-batch size.

    Returns:
        Loader and ``(None, None)`` because counterfactuals are unavailable.
    """
    df = nsw_mixtape.load_pandas().data
    y = torch.tensor(df["re78"].values, dtype=torch.float32).unsqueeze(-1)
    t = torch.tensor(df["treat"].values, dtype=torch.float32).unsqueeze(-1)
    x = torch.tensor(
        df.drop(columns=["re78", "treat", "data_id"]).values, dtype=torch.float32
    )
    mean = x.mean(0, keepdim=True)
    std = x.std(0, unbiased=False, keepdim=True).clamp_min(1e-6)
    x = (x - mean) / std
    loader = DataLoader(TensorDataset(x, t, y), batch_size=batch_size, shuffle=True)
    return loader, (None, None)
