"""Command-line hyperparameter sweep with Optuna."""

from __future__ import annotations

import argparse
from typing import Callable, Dict, Iterable, Tuple

import optuna
import torch
from torch.utils.data import DataLoader

from .datasets import (
    get_toy_dataloader,
    get_complex_dataloader,
    get_ihdp_dataloader,
    get_jobs_dataloader,
    get_acic2016_dataloader,
    get_acic2018_dataloader,
    get_twins_dataloader,
    get_lalonde_dataloader,
    get_confounding_dataloader,
)
from .training.train_acx import train_acx
from .training import ModelConfig, TrainingConfig
from .evaluation.evaluate import evaluate, evaluate_dr
from .utils import default_device, set_seed


DATASET_LOADERS: Dict[
    str, Callable[[], Tuple[DataLoader, Tuple[torch.Tensor, torch.Tensor]]]
] = {
    "toy": get_toy_dataloader,
    "complex": get_complex_dataloader,
    "ihdp": get_ihdp_dataloader,
    "jobs": get_jobs_dataloader,
    "acic2016": get_acic2016_dataloader,
    "acic2018": get_acic2018_dataloader,
    "twins": get_twins_dataloader,
    "lalonde": get_lalonde_dataloader,
    "confounded": get_confounding_dataloader,
}


def _parse_args(args: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optuna hyperparameter sweep")
    parser.add_argument(
        "dataset",
        choices=DATASET_LOADERS.keys(),
        help="dataset to use",
    )
    parser.add_argument(
        "--trials", type=int, default=20, help="number of Optuna trials"
    )
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    return parser.parse_args(args)


def _space(trial: optuna.Trial) -> dict:
    rep_dim = trial.suggest_int("rep_dim", 32, 128)

    phi_width = trial.suggest_int("phi_width", 64, 256)
    phi_layers = trial.suggest_int("phi_layers", 1, 3)
    phi_layers = [phi_width] * phi_layers

    head_width = trial.suggest_int("head_width", 32, 128)
    head_layers = trial.suggest_int("head_layers", 1, 3)
    head_layers = [head_width] * head_layers

    disc_width = trial.suggest_int("disc_width", 32, 128)
    disc_layers = trial.suggest_int("disc_layers", 1, 3)
    disc_layers = [disc_width] * disc_layers

    params = {
        "rep_dim": rep_dim,
        "phi_layers": phi_layers,
        "head_layers": head_layers,
        "disc_layers": disc_layers,
        "lr_g": trial.suggest_float("lr_g", 1e-4, 1e-2, log=True),
        "lr_d": trial.suggest_float("lr_d", 1e-4, 1e-2, log=True),
        "alpha_out": trial.suggest_float("alpha_out", 0.5, 2.0),
        "beta_cons": trial.suggest_float("beta_cons", 1.0, 20.0),
        "gamma_adv": trial.suggest_float("gamma_adv", 0.0, 2.0),
        "phi_dropout": trial.suggest_float("phi_dropout", 0.0, 0.5),
        "head_dropout": trial.suggest_float("head_dropout", 0.0, 0.5),
        "disc_dropout": trial.suggest_float("disc_dropout", 0.0, 0.5),
        "epochs": trial.suggest_int("epochs", 10, 50),
        "batch_norm": trial.suggest_categorical("batch_norm", [True, False]),
        "spectral_norm": trial.suggest_categorical("spectral_norm", [True, False]),
    }
    return params


def main(argv: Iterable[str] | None = None) -> None:
    args = _parse_args(argv)
    set_seed(args.seed)
    loader_fn = DATASET_LOADERS[args.dataset]
    loader, (mu0, mu1) = loader_fn()
    p = loader.dataset.tensors[0].size(1)
    device = default_device()

    X = torch.cat([b[0] for b in loader])
    T_all = torch.cat([b[1] for b in loader])
    Y_all = torch.cat([b[2] for b in loader])

    def objective(trial: optuna.Trial) -> float:
        params = _space(trial)
        model_cfg = ModelConfig(
            p=p,
            rep_dim=params["rep_dim"],
            phi_layers=params["phi_layers"],
            head_layers=params["head_layers"],
            disc_layers=params["disc_layers"],
            batch_norm=params["batch_norm"],
        )
        train_cfg = TrainingConfig(
            epochs=params["epochs"],
            lr_g=params["lr_g"],
            lr_d=params["lr_d"],
            alpha_out=params["alpha_out"],
            beta_cons=params["beta_cons"],
            gamma_adv=params["gamma_adv"],
            phi_dropout=params["phi_dropout"],
            head_dropout=params["head_dropout"],
            disc_dropout=params["disc_dropout"],
            spectral_norm=params["spectral_norm"],
            verbose=False,
        )
        model = train_acx(loader, model_cfg, train_cfg, device=device)
        if mu0 is not None and mu1 is not None:
            return evaluate(model, X, mu0, mu1)
        propensity = torch.full_like(T_all, T_all.float().mean().item())
        return evaluate_dr(model, X, T_all, Y_all, propensity)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=args.trials)
    print("Best value", study.best_value)
    print("Best params", study.best_params)


if __name__ == "__main__":
    main()
