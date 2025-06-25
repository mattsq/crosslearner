"""Utility helpers for reproducibility and PyTorch model handling."""

import os
import random

import numpy as np
import torch
import torch.nn as nn

from .scheduler import MutableBatchSampler, GNSBatchScheduler


def set_seed(seed: int, *, deterministic: bool = False) -> None:
    """Set random seed for Python, NumPy and PyTorch.

    Args:
        seed: Seed value used for all RNGs.
        deterministic: If ``True`` enable deterministic CUDA operations.

    Returns:
        ``None``. The global random state is modified in-place.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def default_device() -> str:
    """Return the best available device string.

    Returns:
        ``"cuda"`` when a GPU is available, otherwise ``"cpu"``.
    """
    return "cuda" if torch.cuda.is_available() else "cpu"


def apply_spectral_norm(model: nn.Module) -> None:
    """Apply spectral normalization to all linear layers in ``model``.

    Args:
        model: Module whose ``nn.Linear`` layers should be normalised.

    Returns:
        ``None``. ``model`` is modified in-place.
    """
    for module in model.modules():
        if isinstance(module, nn.Linear):
            nn.utils.spectral_norm(module)


def model_device(model: nn.Module) -> torch.device:
    """Return the device of ``model`` or CPU if no parameters.

    Args:
        model: Neural network whose parameter device should be inspected.

    Returns:
        Device of the first parameter or ``cpu`` when the model has none.
    """
    try:
        return next(model.parameters()).device
    except StopIteration:  # pragma: no cover - unlikely
        return torch.device("cpu")


__all__ = [
    "set_seed",
    "default_device",
    "apply_spectral_norm",
    "model_device",
    "MutableBatchSampler",
    "GNSBatchScheduler",
]
