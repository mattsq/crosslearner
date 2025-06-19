from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional, Tuple, Type

import torch
import torch.nn as nn


@dataclass
class ModelConfig:
    """Configuration for the :class:`ACX` architecture."""

    p: int
    rep_dim: int = 64
    disentangle: bool = False
    rep_dim_c: int | None = None
    rep_dim_a: int | None = None
    rep_dim_i: int | None = None
    phi_layers: Iterable[int] | None = (128,)
    head_layers: Iterable[int] | None = (64,)
    disc_layers: Iterable[int] | None = (64,)
    activation: str | Callable[[], nn.Module] = "relu"
    phi_dropout: float = 0.0
    head_dropout: float = 0.0
    disc_dropout: float = 0.0
    residual: bool = False
    phi_residual: bool | None = None
    head_residual: bool | None = None
    disc_residual: bool | None = None
    disc_pack: int = (
        1  #: Number of samples concatenated for PacGAN-style discriminator.
    )
    batch_norm: bool = False


@dataclass
class TrainingConfig:
    """Hyperparameters controlling the training process."""

    epochs: int = 30
    alpha_out: float = 1.0
    beta_cons: float = 10.0
    gamma_adv: float = 1.0
    lr_g: float = 1e-3
    lr_d: float = 1e-3
    optimizer: str | Type[torch.optim.Optimizer] = "adam"
    opt_g_kwargs: dict = field(default_factory=dict)
    opt_d_kwargs: dict = field(default_factory=dict)
    lr_scheduler: str | Type[torch.optim.lr_scheduler._LRScheduler] | None = None
    sched_g_kwargs: dict = field(default_factory=dict)
    sched_d_kwargs: dict = field(default_factory=dict)
    grad_clip: float = 2.0
    warm_start: int = 0
    use_wgan_gp: bool = False
    adv_loss: str = "bce"
    ema_decay: Optional[float] = None
    spectral_norm: bool = False
    feature_matching: bool = False
    label_smoothing: bool = False
    instance_noise: bool = False
    gradient_reversal: bool = False
    ttur: bool = False
    disc_steps: int = 1
    disc_aug_prob: float = 0.0
    disc_aug_noise: float = 0.0
    mmd_weight: float = 0.0
    mmd_sigma: float = 1.0
    lambda_gp: float = 10.0
    r1_gamma: float = 0.0
    r2_gamma: float = 0.0
    adaptive_reg: bool = False
    d_reg_lower: float = 0.3
    d_reg_upper: float = 0.7
    reg_factor: float = 1.1
    lambda_gp_min: float = 1e-3
    lambda_gp_max: float = 100.0
    unrolled_steps: int = 0
    eta_fm: float = 5.0
    grl_weight: float = 1.0
    contrastive_weight: float = 0.0
    contrastive_margin: float = 1.0
    contrastive_noise: float = 0.0
    delta_prop: float = 0.0
    lambda_dr: float = 0.0
    noise_std: float = 0.0
    noise_consistency_weight: float = 0.0
    rep_consistency_weight: float = 0.0
    rep_momentum: float = 0.99
    adv_t_weight: float = 0.0
    adv_y_weight: float = 0.0
    tensorboard_logdir: Optional[str] = None
    weight_clip: Optional[float] = None
    val_data: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None
    risk_data: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None
    risk_folds: int = 5
    nuisance_propensity_epochs: int = 500
    nuisance_outcome_epochs: int = 3
    nuisance_early_stop: int = 10
    patience: int = 0
    verbose: bool = True
    return_history: bool = False
