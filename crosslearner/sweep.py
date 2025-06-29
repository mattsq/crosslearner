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
    get_aircraft_dataloader,
    get_tricky_dataloader,
    get_random_dag_dataloader,
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
    "aircraft": get_aircraft_dataloader,
    "tricky": get_tricky_dataloader,
    "random_dag": get_random_dag_dataloader,
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


def _space(trial: optuna.Trial, dataset_size: int) -> dict:
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

    epistemic_consistency = trial.suggest_categorical(
        "epistemic_consistency", [True, False]
    )
    tau_heads = trial.suggest_int("tau_heads", 2 if epistemic_consistency else 1, 4)

    upper_B = min(512, dataset_size)
    lower_B = min(32, upper_B)
    params = {
        "rep_dim": rep_dim,
        "phi_layers": phi_layers,
        "head_layers": head_layers,
        "disc_layers": disc_layers,
        # Newly exposed model parameters
        "embed_dim": trial.suggest_int("embed_dim", 4, 16),
        "activation": trial.suggest_categorical("activation", ["relu", "gelu", "tanh"]),
        "lr_g": trial.suggest_float("lr_g", 1e-4, 1e-2, log=True),
        "lr_d": trial.suggest_float("lr_d", 1e-4, 1e-2, log=True),
        "alpha_out": trial.suggest_float("alpha_out", 0.5, 2.0),
        "beta_cons": trial.suggest_float("beta_cons", 1.0, 20.0),
        "gamma_adv": trial.suggest_float("gamma_adv", 0.0, 2.0),
        "phi_dropout": trial.suggest_float("phi_dropout", 0.0, 0.5),
        "head_dropout": trial.suggest_float("head_dropout", 0.0, 0.5),
        "disc_dropout": trial.suggest_float("disc_dropout", 0.0, 0.5),
        "epochs": trial.suggest_int("epochs", 10, 50),
        "normalization": trial.suggest_categorical(
            "normalization", [None, "batch", "layer", "group"]
        ),
        "spectral_norm": trial.suggest_categorical("spectral_norm", [True, False]),
        # Residual connections per subnetwork
        "residual": trial.suggest_categorical("residual", [True, False]),
        "phi_residual": trial.suggest_categorical("phi_residual", [True, False]),
        "head_residual": trial.suggest_categorical("head_residual", [True, False]),
        "disc_residual": trial.suggest_categorical("disc_residual", [True, False]),
        "disc_pack": trial.suggest_int("disc_pack", 1, 4),
        "moe_experts": trial.suggest_int("moe_experts", 1, 4),
        "tau_heads": tau_heads,
        "tau_bias": trial.suggest_categorical("tau_bias", [True, False]),
        # Newly exposed training parameters
        "warm_start": trial.suggest_int("warm_start", 0, 5),
        "adv_loss": trial.suggest_categorical(
            "adv_loss", ["bce", "hinge", "ls", "rgan"]
        ),
        "feature_matching": trial.suggest_categorical(
            "feature_matching", [True, False]
        ),
        "label_smoothing": trial.suggest_categorical("label_smoothing", [True, False]),
        "instance_noise": trial.suggest_categorical("instance_noise", [True, False]),
        "gradient_reversal": trial.suggest_categorical(
            "gradient_reversal", [True, False]
        ),
        "disc_steps": trial.suggest_int("disc_steps", 1, 3),
        "disc_aug_prob": trial.suggest_float("disc_aug_prob", 0.0, 0.5),
        "disc_aug_noise": trial.suggest_float("disc_aug_noise", 0.0, 0.1),
        "mmd_weight": trial.suggest_float("mmd_weight", 0.0, 1.0),
        "mmd_sigma": trial.suggest_float("mmd_sigma", 0.1, 2.0),
        "lambda_gp": trial.suggest_float("lambda_gp", 0.0, 20.0),
        "r1_gamma": trial.suggest_float("r1_gamma", 0.0, 2.0),
        "r2_gamma": trial.suggest_float("r2_gamma", 0.0, 2.0),
        "adaptive_reg": trial.suggest_categorical("adaptive_reg", [True, False]),
        "unrolled_steps": trial.suggest_int("unrolled_steps", 0, 5),
        "unrolled_steps_epochs": trial.suggest_int("unrolled_steps_epochs", 0, 10),
        "ema_decay": trial.suggest_float("ema_decay", 0.9, 0.999),
        "use_wgan_gp": trial.suggest_categorical("use_wgan_gp", [True, False]),
        "ttur": trial.suggest_categorical("ttur", [True, False]),
        "grl_weight": trial.suggest_float("grl_weight", 0.5, 2.0),
        "contrastive_weight": trial.suggest_float("contrastive_weight", 0.0, 1.0),
        "contrastive_margin": trial.suggest_float("contrastive_margin", 0.5, 2.0),
        "contrastive_noise": trial.suggest_float("contrastive_noise", 0.0, 0.1),
        "delta_prop": trial.suggest_float("delta_prop", 0.0, 1.0),
        "lambda_dr": trial.suggest_float("lambda_dr", 0.0, 1.0),
        "noise_std": trial.suggest_float("noise_std", 0.0, 0.1),
        "noise_consistency_weight": trial.suggest_float(
            "noise_consistency_weight", 0.0, 1.0
        ),
        "moe_entropy_weight": trial.suggest_float("moe_entropy_weight", 0.0, 1.0),
        "disentangle": trial.suggest_categorical("disentangle", [True, False]),
        "adv_t_weight": trial.suggest_float("adv_t_weight", 0.0, 1.0),
        "adv_y_weight": trial.suggest_float("adv_y_weight", 0.0, 1.0),
        "epistemic_consistency": epistemic_consistency,
        "rep_consistency_weight": trial.suggest_float(
            "rep_consistency_weight", 0.0, 1.0
        ),
        "rep_momentum": trial.suggest_float("rep_momentum", 0.9, 0.999),
        "pretrain_epochs": trial.suggest_int("pretrain_epochs", 0, 10),
        "pretrain_mask_prob": trial.suggest_float("pretrain_mask_prob", 0.1, 0.3),
        "pretrain_lr": trial.suggest_float("pretrain_lr", 1e-4, 1e-2, log=True),
        "finetune_lr": trial.suggest_float("finetune_lr", 1e-4, 1e-2, log=True),
        "freeze_phi_epoch": trial.suggest_int("freeze_phi_epoch", 0, 20),
        "weight_clip": trial.suggest_float("weight_clip", 0.0, 0.1),
        "log_grad_norms": trial.suggest_categorical("log_grad_norms", [True, False]),
        "log_learning_rate": trial.suggest_categorical(
            "log_learning_rate", [True, False]
        ),
        "log_weight_histograms": trial.suggest_categorical(
            "log_weight_histograms", [True, False]
        ),
        "optimizer": trial.suggest_categorical("optimizer", ["adam", "adamw", "sgd"]),
        "lr_scheduler": trial.suggest_categorical(
            "lr_scheduler", [None, "cosine", "step"]
        ),
        "adaptive_batch": trial.suggest_categorical("adaptive_batch", [True, False]),
        "gns_target": trial.suggest_float("gns_target", 0.5, 2.0),
        "gns_band": trial.suggest_float("gns_band", 0.5, 1.0),
        "gns_growth_factor": trial.suggest_int("gns_growth_factor", 1, 4),
        "gns_check_every": trial.suggest_int("gns_check_every", 50, 400),
        "gns_plateau_patience": trial.suggest_int("gns_plateau_patience", 1, 5),
        "gns_ema": trial.suggest_float("gns_ema", 0.5, 0.99),
        "gns_max_batch": trial.suggest_int("gns_max_batch", 64, 512),
        "use_gradnorm": trial.suggest_categorical("use_gradnorm", [True, False]),
        "gradnorm_alpha": trial.suggest_float("gradnorm_alpha", 0.5, 1.5),
        "gradnorm_lr": trial.suggest_float("gradnorm_lr", 1e-4, 1e-2, log=True),
        "batch_size": trial.suggest_int("batch_size", lower_B, upper_B),
    }

    if params["disentangle"]:
        params.update(
            rep_dim_c=trial.suggest_int("rep_dim_c", 16, 64),
            rep_dim_a=trial.suggest_int("rep_dim_a", 16, 64),
            rep_dim_i=trial.suggest_int("rep_dim_i", 16, 64),
        )
    return params


def main(argv: Iterable[str] | None = None) -> None:
    args = _parse_args(argv)
    set_seed(args.seed)
    loader_fn = DATASET_LOADERS[args.dataset]
    loader, (mu0, mu1) = loader_fn()
    dataset = loader.dataset
    dataset_size = len(dataset)
    p = dataset.tensors[0].size(1)
    device = default_device()

    X, T_all, Y_all = dataset.tensors  # type: ignore[misc]

    def objective(trial: optuna.Trial) -> float:
        params = _space(trial, dataset_size)
        model_cfg = ModelConfig(
            p=p,
            **{
                k: v for k, v in params.items() if k in ModelConfig.__dataclass_fields__
            },
        )
        train_params = {
            k: v for k, v in params.items() if k in TrainingConfig.__dataclass_fields__
        }
        train_cfg = TrainingConfig(**train_params, verbose=False)
        batch_size = params.get("batch_size", loader.batch_size or 1)
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        model = train_acx(train_loader, model_cfg, train_cfg, device=device)
        if mu0 is not None and mu1 is not None:
            return evaluate(model, X, mu0, mu1)
        propensity = torch.full_like(T_all, T_all.float().mean().item())
        return evaluate_dr(model, X, T_all, Y_all, propensity)

    study = optuna.create_study(direction="minimize")
    if args.trials < 1:
        print("No trials executed")
        return
    study.optimize(objective, n_trials=args.trials)
    print("Best value", study.best_value)
    print("Best params", study.best_params)


if __name__ == "__main__":
    main()
