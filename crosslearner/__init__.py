"""Public API for the ``crosslearner`` package."""

from .datasets import get_toy_dataloader, get_complex_dataloader
from .training.train_acx import train_acx
from .training.history import EpochStats, History
from .evaluation.evaluate import evaluate
from .experiments import ExperimentManager, cross_validate_acx
from .experiments.sweep import run_sweep
from .utils import set_seed, default_device
from .visualization import (
    plot_losses,
    scatter_tau,
    plot_tau_distribution,
    plot_covariate_balance,
    plot_propensity_overlap,
    plot_residuals,
)

__all__ = [
    "get_toy_dataloader",
    "get_complex_dataloader",
    "train_acx",
    "EpochStats",
    "History",
    "evaluate",
    "plot_losses",
    "scatter_tau",
    "plot_tau_distribution",
    "plot_covariate_balance",
    "plot_propensity_overlap",
    "plot_residuals",
    "ExperimentManager",
    "cross_validate_acx",
    "run_sweep",
    "set_seed",
    "default_device",
]
