# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
- Initial creation of CHANGELOG
- Added `crosslearner-benchmark` command comparing ACX to baseline models
- Added configurable weight initialisation schemes for ACX MLPs
- Added PacGAN-style discriminator packing via `disc_pack` configuration
- Unified benchmark CLI and baseline comparison
- Baseline MLPRegressors now train until convergence
- Added hinge and least-squares GAN losses via `adv_loss` option
- Added exponential moving average (`ema_decay`) for generator parameters
- Added R1/R2 regularization and unrolled discriminator updates
- Fixed `set_seed` to skip `torch.cuda.manual_seed_all` when CUDA is unavailable
- Added synthetic data generation utilities with configurable noise and missing outcomes
- MLP and ACX are now TorchScript and ONNX exportable via `export_model`
