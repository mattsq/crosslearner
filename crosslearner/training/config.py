from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional, Tuple, Type

import torch
import torch.nn as nn


@dataclass
class ModelConfig:
    """Configuration for the :class:`ACX` architecture."""

    p: int
    cat_dims: Iterable[int] | None = None  #: Cardinalities of categorical features.
    embed_dim: int = 8  #: Embedding dimension for each categorical variable.
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
    moe_experts: int = 1  #: Number of expert pairs for mixture-of-experts heads.
    tau_heads: int = 1  #: Number of effect heads for epistemic ensembling.
    tau_bias: bool = True  #: If ``False`` freeze the effect head biases at zero.


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
    warm_start: int = (
        0
        #: Number of initial epochs using only the outcome loss before
        #: adversarial objectives kick in.  Setting this to ``>0`` can
        #: stabilise training on small or noisy datasets.
    )
    use_wgan_gp: bool = False
    adv_loss: str = "bce"
    ema_decay: Optional[float] = (
        None
        #: Exponential moving average decay for generator parameters.
        #: When set, a detached copy of the model is updated after each
        #: optimisation step via ``p_ema = ema_decay * p_ema + (1 - ema_decay) * p``.
        #: The EMA weights are used for evaluation and returned at the end of training.
    )
    spectral_norm: bool = False
    feature_matching: bool = False
    label_smoothing: bool = False
    instance_noise: bool = (
        False
        #: Add decaying Gaussian noise to discriminator targets to regularise early training.
    )
    gradient_reversal: bool = False
    ttur: bool = (
        False
        #: Freeze discriminator updates when its loss drops below ``0.3``
        #: to implement a two time-scale update rule.
    )
    disc_steps: int = 1
    disc_aug_prob: float = 0.0
    disc_aug_noise: float = 0.0
    mmd_weight: float = 0.0
    mmd_sigma: float = 1.0
    lambda_gp: float = 10.0
    r1_gamma: float = (
        0.0
        #: Coefficient for the R1 gradient penalty on real samples.
    )
    r2_gamma: float = (
        0.0
        #: Coefficient for the R2 gradient penalty on fake samples.
    )
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
    delta_prop: float = 0.0  #: Weight for the propensity head cross-entropy loss.
    lambda_dr: float = 0.0  #: Weight for the doubly robust loss term.
    noise_std: float = 0.0
    noise_consistency_weight: float = 0.0
    epistemic_consistency: bool = False  #: Down-weight consistency loss by uncertainty.
    rep_consistency_weight: float = 0.0
    moe_entropy_weight: float = 0.0  #: Weight for gating entropy regularization.
    rep_momentum: float = 0.99
    pretrain_epochs: int = (
        0
        #: Number of epochs to pretrain the encoder using masked feature
        #: reconstruction. ``0`` disables pretraining.
    )
    pretrain_mask_prob: float = (
        0.15
        #: Fraction of input features randomly masked during representation
        #: pretraining.
    )
    pretrain_lr: float | None = (
        None
        #: Optional learning rate for the encoder during pretraining. ``None``
        #: uses ``lr_g``.
    )
    finetune_lr: float | None = (
        None
        #: Optional learning rate for the encoder after pretraining. ``None``
        #: scales ``lr_g`` by ``0.1``.
    )
    adv_t_weight: float = (
        0.0
        #: Weight for predicting treatment from the confounder and
        #: outcome representations when ``disentangle=True``.
    )
    adv_y_weight: float = (
        0.0
        #: Weight for predicting the observed outcome from the confounder and
        #: instrument representations when ``disentangle=True``.
    )
    tensorboard_logdir: Optional[str] = (
        None
        #: Directory in which to write TensorBoard event files during training.
    )
    log_grad_norms: bool = (
        False
        #: Record gradient norms to TensorBoard when ``tensorboard_logdir`` is set.
    )
    log_learning_rate: bool = (
        False
        #: Log the learning rate of each optimiser at the end of every epoch.
    )
    log_weight_histograms: bool = (
        False
        #: Write histograms of model parameters to TensorBoard once per epoch.
    )
    weight_clip: Optional[float] = (
        None
        #: Clip discriminator weights to ``[-weight_clip, weight_clip]`` after
        #: each update.  Useful for Wasserstein-style training.  ``None``
        #: disables clipping.
    )
    val_data: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None
    risk_data: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None
    risk_folds: int = 5
    nuisance_propensity_epochs: int = 500
    nuisance_outcome_epochs: int = 3
    nuisance_early_stop: int = 10
    patience: int = 0
    early_stop_metric: str = (
        "auto"
        #: Metric for early stopping.
        #: ``auto`` uses validation PEHE when ``val_data`` is provided,
        #: orthogonal risk when ``risk_data`` is provided and the generator
        #: loss otherwise. Set to ``pehe`` or ``risk`` to override.
    )
    verbose: bool = True
    return_history: bool = False
    active_aug_freq: int = (
        0
        #: Epoch interval for active counterfactual data augmentation.
        #: ``0`` disables the feature.
    )
    active_aug_samples: int = 0
    active_aug_steps: int = 10
    active_aug_lr: float = 0.1
    adaptive_batch: bool = (
        False
        #: Enable the gradient noise scale batch scheduler.
    )
    gns_target: float = (
        1.0
        #: Target gradient noise scale triggering batch growth.
    )
    gns_band: float = (
        0.7
        #: Multiplicative tolerance around ``gns_target`` before growing the
        #: batch size.
    )
    gns_growth_factor: int = (
        2
        #: Factor by which to multiply the batch size when ``gns_target`` is
        #: reached.
    )
    gns_check_every: int = (
        200
        #: Number of training steps between gradient noise scale evaluations.
    )
    gns_plateau_patience: int = (
        3
        #: Force a batch growth step after this many evaluations without
        #: improvement in the validation loss.
    )
    gns_ema: float = (
        0.9
        #: Exponential moving average factor for smoothing the gradient noise
        #: scale estimates.
    )
    gns_max_batch: Optional[int] = (
        None
        #: Optional hard limit on the batch size reached by the scheduler.
    )
