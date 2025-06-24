"""Public API for the ``crosslearner`` package."""

from .datasets import get_toy_dataloader, get_complex_dataloader
from .training.train_acx import train_acx, train_acx_ensemble
from .training.history import EpochStats, History
from .evaluation.evaluate import evaluate
from .evaluation.uncertainty import (
    predict_tau_mc_dropout,
    predict_tau_ensemble,
    predict_tau_mc_ensemble,
)
from .models.stochastic import DropConnectLinear, BaseNet, StochasticEnsemble
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
    plot_grad_norms,
    plot_learning_rates,
    plot_partial_dependence,
    plot_ice,
)

__all__ = [
    "get_toy_dataloader",
    "get_complex_dataloader",
    "train_acx",
    "train_acx_ensemble",
    "EpochStats",
    "History",
    "evaluate",
    "predict_tau_mc_dropout",
    "predict_tau_ensemble",
    "predict_tau_mc_ensemble",
    "plot_losses",
    "scatter_tau",
    "plot_tau_distribution",
    "plot_covariate_balance",
    "plot_propensity_overlap",
    "plot_residuals",
    "plot_grad_norms",
    "plot_learning_rates",
    "plot_partial_dependence",
    "plot_ice",
    "ExperimentManager",
    "cross_validate_acx",
    "set_seed",
    "default_device",
    "export_model",
    "DropConnectLinear",
    "BaseNet",
    "StochasticEnsemble",
]
