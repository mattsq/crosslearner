"""Public API for the ``crosslearner`` package."""

from .datasets import get_toy_dataloader, get_complex_dataloader
from .training.train_acx import train_acx
from .training.history import EpochStats, History
from .evaluation.evaluate import evaluate
from .evaluation.uncertainty import predict_tau_mc_dropout
from .experiments import ExperimentManager, cross_validate_acx
from .utils import set_seed, default_device
from .export import export_model
from .visualization import (
    plot_losses,
    scatter_tau,
    plot_tau_distribution,
    plot_covariate_balance,
    plot_propensity_overlap,
    plot_residuals,
    plot_partial_dependence,
    plot_ice,
)

__all__ = [
    "get_toy_dataloader",
    "get_complex_dataloader",
    "train_acx",
    "EpochStats",
    "History",
    "evaluate",
    "predict_tau_mc_dropout",
    "plot_losses",
    "scatter_tau",
    "plot_tau_distribution",
    "plot_covariate_balance",
    "plot_propensity_overlap",
    "plot_residuals",
    "plot_partial_dependence",
    "plot_ice",
    "ExperimentManager",
    "cross_validate_acx",
    "set_seed",
    "default_device",
    "export_model",
]
