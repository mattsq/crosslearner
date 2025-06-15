"""Loader for the IHDP semi-synthetic benchmark."""

import os
import urllib.request
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

URL_TRAIN = "https://www.fredjo.com/files/ihdp_npci_1-100.train.npz"
URL_TEST = "https://www.fredjo.com/files/ihdp_npci_1-100.test.npz"


def _download(url: str, path: str) -> str:
    if not os.path.exists(path):
        urllib.request.urlretrieve(url, path)
    return path


def get_ihdp_dataloader(
    seed: int = 0, batch_size: int = 256, *, data_dir: str | None = None
) -> Tuple[DataLoader, Tuple[torch.Tensor, torch.Tensor]]:
    """Return IHDP dataloader for the given replication index.

    Args:
        seed: Replication index from 0 to 99.
        batch_size: Size of mini-batches.
        data_dir: Optional directory to cache the dataset.

    Returns:
        Data loader and tuple ``(mu0, mu1)`` with true outcomes.
    """
    data_dir = data_dir or os.path.join(os.path.dirname(__file__), "_data")
    os.makedirs(data_dir, exist_ok=True)
    f_train = _download(URL_TRAIN, os.path.join(data_dir, "ihdp_train.npz"))
    f_test = _download(URL_TEST, os.path.join(data_dir, "ihdp_test.npz"))

    train = np.load(f_train)
    test = np.load(f_test)
    X = np.concatenate([train["x"][:, :, seed], test["x"][:, :, seed]], axis=0)
    T = np.concatenate([train["t"][:, seed], test["t"][:, seed]], axis=0)
    Y = np.concatenate([train["yf"][:, seed], test["yf"][:, seed]], axis=0)
    mu0 = np.concatenate([train["mu0"][:, seed], test["mu0"][:, seed]], axis=0)
    mu1 = np.concatenate([train["mu1"][:, seed], test["mu1"][:, seed]], axis=0)

    dset = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(T, dtype=torch.float32).unsqueeze(-1),
        torch.tensor(Y, dtype=torch.float32).unsqueeze(-1),
    )
    loader = DataLoader(dset, batch_size=batch_size, shuffle=True)
    return loader, (
        torch.tensor(mu0, dtype=torch.float32).unsqueeze(-1),
        torch.tensor(mu1, dtype=torch.float32).unsqueeze(-1),
    )
