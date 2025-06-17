import os
import random

import numpy as np
import torch
import torch.nn as nn


def set_seed(seed: int, *, deterministic: bool = False) -> None:
    """Set random seed for Python, NumPy and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def default_device() -> str:
    """Return the best available device string."""

    return "cuda" if torch.cuda.is_available() else "cpu"


def apply_spectral_norm(model: nn.Module) -> None:
    """Apply spectral normalization to all linear layers in ``model``."""

    for module in model.modules():
        if isinstance(module, nn.Linear):
            nn.utils.spectral_norm(module)
