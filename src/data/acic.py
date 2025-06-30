from __future__ import annotations

import os
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import TensorDataset

from crosslearner.datasets.utils import download_if_missing

URL_2018 = "https://raw.githubusercontent.com/py-why/BenchmarkDatasets/master/acic2018/acic2018.npz"


def _load_2018(
    seed: int, path: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(path)
    x = data["x"][:, :, seed]
    t = data["t"][:, seed]
    y = data["yf"][:, seed]
    mu0 = data["mu0"][:, seed]
    mu1 = data["mu1"][:, seed]
    return x, t, y, mu0, mu1


def _load_2016(
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    from causallib.datasets.data_loader import load_acic16

    data = load_acic16(instance=seed + 1)
    x = data.X.values
    t = data.a.values
    y = data.y.values
    mu0 = data.po.iloc[:, 0].values
    mu1 = data.po.iloc[:, 1].values
    return x, t, y, mu0, mu1


def load_acic(
    *,
    year: int = 2018,
    seed: int = 0,
    val_fraction: float = 0.2,
    test_fraction: float = 0.2,
    data_dir: str | None = None,
) -> Tuple[
    Tuple[TensorDataset, TensorDataset, TensorDataset],
    Tuple[torch.Tensor, torch.Tensor],
]:
    """Load ACIC benchmark data with deterministic splits.

    Args:
        year: Which dataset variant to load (2016 or 2018).
        seed: Replication index / RNG seed.
        val_fraction: Fraction of samples used for validation.
        test_fraction: Fraction of samples used for testing.
        data_dir: Optional cache directory for downloads.

    Returns:
        ``((train, val, test), (mu0, mu1))`` tuple with three ``TensorDataset``
        splits and the true counterfactual outcomes.
    """

    if year not in {2016, 2018}:
        raise ValueError("year must be 2016 or 2018")

    cache_dir = data_dir or os.path.join(
        os.path.expanduser("~"), ".cache", "otxlearner", "acic"
    )
    os.makedirs(cache_dir, exist_ok=True)

    if year == 2018:
        fpath = os.path.join(cache_dir, "acic2018.npz")
        fpath = download_if_missing(URL_2018, fpath)
        x, t, y, mu0, mu1 = _load_2018(seed, fpath)
    else:
        x, t, y, mu0, mu1 = _load_2016(seed)

    rng = np.random.default_rng(seed)
    idx = np.arange(len(y))
    rng.shuffle(idx)
    train_end = int(len(y) * (1.0 - val_fraction - test_fraction))
    val_end = int(len(y) * (1.0 - test_fraction))
    train_idx = idx[:train_end]
    val_idx = idx[train_end:val_end]
    test_idx = idx[val_end:]

    def to_dataset(sel: np.ndarray) -> TensorDataset:
        return TensorDataset(
            torch.tensor(x[sel], dtype=torch.float32),
            torch.tensor(t[sel], dtype=torch.float32).reshape(-1, 1),
            torch.tensor(y[sel], dtype=torch.float32).reshape(-1, 1),
        )

    train_ds = to_dataset(train_idx)
    val_ds = to_dataset(val_idx)
    test_ds = to_dataset(test_idx)
    mu0_t = torch.tensor(mu0, dtype=torch.float32).reshape(-1, 1)
    mu1_t = torch.tensor(mu1, dtype=torch.float32).reshape(-1, 1)
    return (train_ds, val_ds, test_ds), (mu0_t, mu1_t)
