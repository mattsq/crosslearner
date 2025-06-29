"""Helper functions for downloading datasets and preparing dataloaders."""

import os
import urllib.request
from typing import Iterable

import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset


def download_if_missing(url: str, path: str) -> str:
    """Download ``url`` to ``path`` if the file does not already exist.

    Args:
        url: Remote location to download.
        path: Destination file path.

    Returns:
        Local path to the downloaded file.

    Raises:
        RuntimeError: If the download fails.
    """
    if os.path.exists(path):
        return path
    try:
        urllib.request.urlretrieve(url, path)
    except Exception as exc:  # pragma: no cover - network errors
        raise RuntimeError(
            f"Failed to download dataset from {url}. "
            f"Please download the file manually and place it at {path}."
        ) from exc
    return path


def dataframe_to_dataloader(
    df: pd.DataFrame,
    treatment_col: str,
    outcome_col: str,
    batch_size: int = 256,
    drop: Iterable[str] | None = None,
) -> DataLoader:
    """Convert a ``pandas`` table to a ``DataLoader``.

    Categorical columns are one-hot encoded and missing values imputed with 0.

    Args:
        df: Data table with treatment and outcome columns.
        treatment_col: Column indicating treatment assignment.
        outcome_col: Observed outcome column.
        batch_size: Mini-batch size for the returned loader.
        drop: Additional columns to exclude from the covariates.

    Returns:
        Loader yielding ``(X, T, Y)`` tuples.
    """

    drop = list(drop or [])
    X = df.drop(columns=[treatment_col, outcome_col, *drop])
    X = pd.get_dummies(X).fillna(0.0)
    T = torch.tensor(
        df[treatment_col].astype(float).values, dtype=torch.float32
    ).unsqueeze(-1)
    Y = torch.tensor(
        df[outcome_col].astype(float).values, dtype=torch.float32
    ).unsqueeze(-1)
    X_t = torch.tensor(X.values, dtype=torch.float32)
    dset = TensorDataset(X_t, T, Y)
    return DataLoader(dset, batch_size=batch_size, shuffle=True)
