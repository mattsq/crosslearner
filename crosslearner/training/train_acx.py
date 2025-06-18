import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Callable, Iterable, Optional, Tuple, Type

from .config import ModelConfig, TrainingConfig
from .trainer import ACXTrainer
from .history import History
from ..models.acx import ACX


def train_acx(
    loader: DataLoader,
    p: int,
    *,
    model_config: ModelConfig | None = None,
    training_config: TrainingConfig | None = None,
    rep_dim: int = 64,
    phi_layers: Iterable[int] | None = (128,),
    head_layers: Iterable[int] | None = (64,),
    disc_layers: Iterable[int] | None = (64,),
    activation: str | Callable[[], nn.Module] = "relu",
    phi_dropout: float = 0.0,
    head_dropout: float = 0.0,
    disc_dropout: float = 0.0,
    residual: bool = False,
    phi_residual: bool | None = None,
    head_residual: bool | None = None,
    disc_residual: bool | None = None,
    device: Optional[str] = None,
    seed: int | None = None,
    epochs: int = 30,
    alpha_out: float = 1.0,
    beta_cons: float = 10.0,
    gamma_adv: float = 1.0,
    lr_g: float = 1e-3,
    lr_d: float = 1e-3,
    optimizer: str | Type[torch.optim.Optimizer] = "adam",
    opt_g_kwargs: dict | None = None,
    opt_d_kwargs: dict | None = None,
    lr_scheduler: str | Type[torch.optim.lr_scheduler._LRScheduler] | None = None,
    sched_g_kwargs: dict | None = None,
    sched_d_kwargs: dict | None = None,
    grad_clip: float = 2.0,
    warm_start: int = 0,
    use_wgan_gp: bool = False,
    spectral_norm: bool = False,
    feature_matching: bool = False,
    label_smoothing: bool = False,
    instance_noise: bool = False,
    gradient_reversal: bool = False,
    ttur: bool = False,
    lambda_gp: float = 10.0,
    eta_fm: float = 5.0,
    grl_weight: float = 1.0,
    tensorboard_logdir: Optional[str] = None,
    weight_clip: Optional[float] = None,
    val_data: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
    risk_data: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
    risk_folds: int = 5,
    nuisance_propensity_epochs: int = 500,
    nuisance_outcome_epochs: int = 3,
    nuisance_early_stop: int = 10,
    patience: int = 0,
    verbose: bool = True,
    return_history: bool = False,
) -> ACX | tuple[ACX, History]:
    """Train AC-X model with optional GAN tricks.

    This function preserves the original ``train_acx`` signature but delegates
    all heavy lifting to :class:`ACXTrainer`. When ``model_config`` or
    ``training_config`` are supplied, the respective keyword arguments are
    ignored.
    """

    if model_config is None:
        model_config = ModelConfig(
            p=p,
            rep_dim=rep_dim,
            phi_layers=phi_layers,
            head_layers=head_layers,
            disc_layers=disc_layers,
            activation=activation,
            phi_dropout=phi_dropout,
            head_dropout=head_dropout,
            disc_dropout=disc_dropout,
            residual=residual,
            phi_residual=phi_residual,
            head_residual=head_residual,
            disc_residual=disc_residual,
        )
    elif model_config.p != p:
        raise ValueError("p does not match model_config.p")

    if training_config is None:
        training_config = TrainingConfig(
            epochs=epochs,
            alpha_out=alpha_out,
            beta_cons=beta_cons,
            gamma_adv=gamma_adv,
            lr_g=lr_g,
            lr_d=lr_d,
            optimizer=optimizer,
            opt_g_kwargs=opt_g_kwargs or {},
            opt_d_kwargs=opt_d_kwargs or {},
            lr_scheduler=lr_scheduler,
            sched_g_kwargs=sched_g_kwargs or {},
            sched_d_kwargs=sched_d_kwargs or {},
            grad_clip=grad_clip,
            warm_start=warm_start,
            use_wgan_gp=use_wgan_gp,
            spectral_norm=spectral_norm,
            feature_matching=feature_matching,
            label_smoothing=label_smoothing,
            instance_noise=instance_noise,
            gradient_reversal=gradient_reversal,
            ttur=ttur,
            lambda_gp=lambda_gp,
            eta_fm=eta_fm,
            grl_weight=grl_weight,
            tensorboard_logdir=tensorboard_logdir,
            weight_clip=weight_clip,
            val_data=val_data,
            risk_data=risk_data,
            risk_folds=risk_folds,
            nuisance_propensity_epochs=nuisance_propensity_epochs,
            nuisance_outcome_epochs=nuisance_outcome_epochs,
            nuisance_early_stop=nuisance_early_stop,
            patience=patience,
            verbose=verbose,
            return_history=return_history,
        )

    trainer = ACXTrainer(model_config, training_config, device=device, seed=seed)
    return trainer.train(loader)
