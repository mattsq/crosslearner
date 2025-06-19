# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
- Initial creation of CHANGELOG
- Added `crosslearner-benchmark` command comparing ACX to baseline models
- Added PacGAN-style discriminator packing via `disc_pack` configuration
- Unified benchmark CLI and baseline comparison
- Baseline MLPRegressors now train until convergence
- Added hinge and least-squares GAN losses via `adv_loss` option
- Added exponential moving average (`ema_decay`) for generator parameters
- Added R1/R2 regularization and unrolled discriminator updates
- Fixed `set_seed` to skip `torch.cuda.manual_seed_all` when CUDA is unavailable
- Added synthetic data generation utilities with configurable noise and missing outcomes
- MLP and ACX are now TorchScript and ONNX exportable via `export_model`
- Added optional batch normalization through `batch_norm` flag in `ModelConfig`
- Expanded documentation: added setup instructions, dataset descriptions and
  testing guide
- Added optional contrastive loss via `contrastive_weight` for balanced
  covariates
- Added Monte Carlo dropout utility `predict_tau_mc_dropout` for uncertainty
  estimation
- Added propensity head and doubly robust training objective via `delta_prop`
  and `lambda_dr` weights
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
- Documented risk-based early stopping with `risk_data` option
