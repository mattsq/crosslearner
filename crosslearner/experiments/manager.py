"""Experiment management utilities for AC-X models."""

from __future__ import annotations

import os
from typing import Callable, Optional, TYPE_CHECKING

import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold

from dataclasses import replace

from ..utils import set_seed, default_device

from ..training.train_acx import train_acx
from ..training import ModelConfig, TrainingConfig
from ..evaluation.evaluate import evaluate

if TYPE_CHECKING:  # pragma: no cover
    import optuna


def cross_validate_acx(
    loader: DataLoader,
    mu0: torch.Tensor,
    mu1: torch.Tensor,
    *,
    model_config: ModelConfig,
    training_config: TrainingConfig,
    folds: int = 5,
    device: Optional[str] = None,
    log_dir: Optional[str] = None,
    seed: int = 0,
) -> float:
    r"""Return mean validation PEHE across ``folds`` splits.

    Args:
        loader: Data loader providing ``(X, T, Y)`` tuples.
        mu0: Potential outcomes under control.
        mu1: Potential outcomes under treatment.
        model_config: Model configuration for the ACX model.
        training_config: Training configuration controlling optimisation.
        folds: Number of cross-validation folds.
        device: Device string passed to :func:`train_acx`.
        log_dir: Optional directory for TensorBoard logs. Each fold logs to
            ``log_dir/fold_i``.
        seed: Random seed for reproducible splits.

    Returns:
        Mean validation :math:`\sqrt{\mathrm{PEHE}}` across folds.
    """
    device = device or default_device()
    set_seed(seed)
    dataset: TensorDataset = loader.dataset  # type: ignore[arg-type]
    X, T, Y = dataset.tensors
    kf = KFold(n_splits=folds, shuffle=True, random_state=seed)
    metrics = []
    batch_size = loader.batch_size or 32

    for i, (train_idx, val_idx) in enumerate(kf.split(X)):
        train_ds = TensorDataset(X[train_idx], T[train_idx], Y[train_idx])
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

        val_data = (
            X[val_idx].to(device),
            mu0[val_idx].to(device),
            mu1[val_idx].to(device),
        )
        fold_dir = os.path.join(log_dir, f"fold_{i}") if log_dir else None
        cfg = replace(
            training_config,
            val_data=val_data,
            tensorboard_logdir=fold_dir,
        )
        model = train_acx(
            train_loader,
            model_config,
            cfg,
            device=device,
            seed=seed,
        )
        metric = evaluate(model, *val_data)
        metrics.append(metric)
    return float(sum(metrics) / len(metrics))


class ExperimentManager:
    """Run cross-validation and Optuna searches for AC-X models."""

    def __init__(
        self,
        loader: DataLoader,
        mu0: torch.Tensor,
        mu1: torch.Tensor,
        p: int,
        *,
        folds: int = 5,
        device: Optional[str] = None,
        log_dir: Optional[str] = None,
        seed: int = 0,
    ) -> None:
        self.loader = loader
        self.mu0 = mu0
        self.mu1 = mu1
        self.p = p
        self.folds = folds
        self.device = device
        self.log_dir = log_dir
        self.seed = seed

    def cross_validate(
        self,
        model_config: ModelConfig,
        training_config: TrainingConfig,
    ) -> float:
        """Cross-validate ``train_acx`` with stored data."""
        return cross_validate_acx(
            self.loader,
            self.mu0,
            self.mu1,
            model_config=model_config,
            training_config=training_config,
            folds=self.folds,
            device=self.device,
            log_dir=self.log_dir,
            seed=self.seed,
        )

    def optimize(
        self,
        space_fn: Callable[["optuna.Trial"], dict],
        *,
        n_trials: int = 50,
        direction: str = "minimize",
    ) -> "optuna.Study":
        """Run an Optuna search over ``n_trials`` using ``space_fn``.

        ``space_fn`` should take an :class:`optuna.Trial` and return a dict
        of parameters passed to :func:`train_acx`.
        """
        import optuna

        def objective(trial: "optuna.Trial") -> float:
            params = space_fn(trial)
            trial_dir = (
                os.path.join(self.log_dir, f"trial_{trial.number}")
                if self.log_dir
                else None
            )
            model_cfg = ModelConfig(
                p=self.p,
                **{
                    k: v
                    for k, v in params.items()
                    if k in ModelConfig.__dataclass_fields__
                },
            )
            train_params = {
                k: v
                for k, v in params.items()
                if k in TrainingConfig.__dataclass_fields__
            }
            train_cfg = TrainingConfig(**train_params)
            return cross_validate_acx(
                self.loader,
                self.mu0,
                self.mu1,
                model_config=model_cfg,
                training_config=replace(
                    train_cfg,
                    tensorboard_logdir=trial_dir,
                ),
                folds=self.folds,
                device=self.device,
                log_dir=trial_dir,
                seed=self.seed,
            )

        study = optuna.create_study(direction=direction)
        study.optimize(objective, n_trials=n_trials)
        return study
