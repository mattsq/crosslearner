# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
- Fixed `crosslearner-sweep` crash when `disentangle=True` by sampling
  `rep_dim_c`, `rep_dim_a` and `rep_dim_i`
- Extended ``plot_losses`` to visualise validation losses and identify risk-based metrics
- Added `get_random_dag_dataloader` for generating random DAG-based synthetic datasets
- Initial creation of CHANGELOG
- Added `crosslearner-benchmark` command comparing ACX to baseline models
- Added GradNorm adaptive loss balancing via ``use_gradnorm`` configuration
- Logged GradNorm weights when ``use_gradnorm`` and ``log_grad_norms`` are enabled
- Added ``freeze_phi_epoch`` option to ``TrainingConfig`` to stop updating the
  representation network after a chosen epoch
- Added PacGAN-style discriminator packing via `disc_pack` configuration
- Unified benchmark CLI and baseline comparison
- Added mixture-of-experts heads via ``moe_experts`` and ``moe_entropy_weight``
- Baseline MLPRegressors now train until convergence
- Added hinge and least-squares GAN losses via `adv_loss` option
- Added relativistic GAN loss via `adv_loss='rgan'`
- Exposed `train_acx_ensemble` in `crosslearner.training`
- Added ``unrolled_steps_epochs`` option to disable unrolled discriminator
  updates after a set number of epochs
- Added exponential moving average (`ema_decay`) for generator parameters
- Added R1/R2 regularization and unrolled discriminator updates
- Refactored unrolled discriminator logic to use stateless functional calls
- Refined unrolled discriminator updates to use torch.func.functional_call and
  in-place parameter updates
- Fixed `set_seed` to skip `torch.cuda.manual_seed_all` when CUDA is unavailable
- Added `crosslearner-sweep` command for Optuna hyperparameter search
- Fixed dropout parameters in `crosslearner-sweep` to apply to `ModelConfig`
  instead of `TrainingConfig`
- Extended `crosslearner-sweep` to sample all configuration options
- Added synthetic data generation utilities with configurable noise and missing outcomes
- MLP and ACX are now TorchScript and ONNX exportable via `export_model`
- Added optional batch normalization through `batch_norm` flag in `ModelConfig`
- Extended `ModelConfig` with `normalization` option supporting layer or group
  normalisation in place of batch norm
- Expanded documentation: added setup instructions, dataset descriptions and
  testing guide
- Added optional contrastive loss via `contrastive_weight` for balanced
  covariates
- Added Monte Carlo dropout utility `predict_tau_mc_dropout` for uncertainty
  estimation
- Added propensity head and doubly robust training objective via `delta_prop`
  and `lambda_dr` weights
- Documented additional evaluation utilities including `policy_risk`,
  `ate_error`, `att_error`, `bootstrap_ci` and `estimate_propensity`
- Added optional noise injection and input consistency regularization via
  `noise_std` and `noise_consistency_weight`
- Added optional representation disentanglement with adversarial training
  through `disentangle` and `adv_t_weight`/`adv_y_weight` options
- Added optional discriminator data augmentation, MMD penalty and multiple
  discriminator update steps for improved GAN stability
- Added adaptive regularization controlled by `adaptive_reg` for tuning
  gradient penalty strength based on discriminator loss
- Added optional representation drift regularization via
  `rep_consistency_weight` and `rep_momentum`

- Documented the ``gradient_reversal`` training option with usage guidance
- Documented risk-based early stopping with `risk_data` option
- Documented the ``spectral_norm`` training option with usage guidance
- Added `plot_partial_dependence` and `plot_ice` visualisations for exploring
  how predicted treatment effects change with a single feature
- Replaced pairwise distance computation in `_mmd_rbf` with `torch.cdist` and
  added regression test
- Removed `retain_graph=True` from gradient penalty computation
- Vectorised R1/R2 gradient penalty computation
- Optimized optimizer resets using `zero_grad(set_to_none=True)` in `ACXTrainer`
- Replaced pairwise-mask logic in `_sample_negatives` with index lists for each
  treatment group
- Cached a zero tensor per epoch and replaced redundant `torch.tensor(0.0)`
  constructions
- Reused a single `torch.cdist` to compute all pairwise distances in `_mmd_rbf`
- Switched `_mmd_rbf` to an unbiased estimator using the kernel trick and added
  a regression test comparing to the previous implementation
- Documented the ``instance_noise`` training option with motivation and usage details
- Documented optional discriminator feature augmentation via ``disc_aug_prob`` and ``disc_aug_noise``
- Documented representation disentanglement with ``adv_t_weight`` and ``adv_y_weight``
- Documented optional MMD regularisation via ``mmd_weight`` and ``mmd_sigma``
- Documented TensorBoard logging via ``tensorboard_logdir`` option
- Documented ``warm_start`` option for pretraining the generator
- Documented ``opt_g_kwargs`` and ``opt_d_kwargs`` for custom optimizer
  arguments
- Added ``opt_phi_kwargs``, ``opt_head_kwargs`` and ``opt_disc_kwargs`` for
  overriding optimiser parameters of individual network components
- Added options to log gradient norms, learning rates and weight histograms for
  improved training diagnostics
- Extended visualisation utilities to plot auxiliary losses, gradient norms and
  learning rate schedules
- Detached MOE gating weights to avoid memory leaks and added unit tests for
  ``MOEHeads`` gating behaviour
- Added epistemic-aware consistency loss via ``epistemic_consistency`` and
  ``tau_heads`` options
- Passed ``tau_heads`` from ``ModelConfig`` to ``ACX`` and validate when
  ``epistemic_consistency`` is enabled
- Added ``effect_consistency_weight`` property and tests for tau head validation
- Stopped tracking gradients for ``tau_variance`` to reduce memory usage
- Added active counterfactual data augmentation via ``active_aug_freq`` and
  related options
- Active augmentation now preserves ``DataLoader`` settings when appending
  pseudo data
- Active augmentation now handles optional ``DataLoader`` arguments for
  compatibility with older PyTorch versions
- Disabled gradient tracking for model parameters during active counterfactual
  search to reduce overhead
- Added optional representation pre-training via masked feature reconstruction
- Documented representation pretraining options and updated config comments
- Added ``tau_bias`` flag to freeze effect head biases for more stable training
- Added ``train_acx_ensemble`` and ``predict_tau_ensemble`` helpers for model
  ensembling
- Added ``get_tricky_dataloader`` providing a small imbalanced dataset for
  testing discriminator stabilisation tricks
- Logged validation outcome, consistency and adversarial losses when validation
  data is provided
- Added ``early_stop_metric`` option to ``TrainingConfig`` for choosing the
  validation metric used for early stopping
- Documented ``pehe`` evaluation helper in usage examples
- Added optional categorical embeddings via ``cat_dims`` and ``embed_dim`` parameters to ``ACX``
- Extended ``ModelConfig`` with ``cat_dims`` and ``embed_dim`` fields and
  enabled representation pretraining with categorical inputs

- Added missing docstrings across several modules to improve code clarity
- Improved adaptive batch scheduler with unified autocast and new unit tests
- Exposed additional adaptive batch parameters (`gns_band`, `gns_growth_factor`,
  `gns_check_every`, `gns_plateau_patience`, `gns_ema`, `gns_max_batch`) via
  ``TrainingConfig``
- Logged reconstruction loss during representation pretraining
- Logged batch progress with a progress bar when ``verbose`` is enabled during
  training
- ``gns_max_batch`` now defaults to the dataset size when using adaptive
  batching
