# crosslearner

[![CI](https://github.com/mattsq/crosslearner/actions/workflows/ci.yml/badge.svg)](https://github.com/mattsq/crosslearner/actions/workflows/ci.yml) [![Docs](https://github.com/mattsq/crosslearner/actions/workflows/docs.yml/badge.svg)](https://mattsq.github.io/crosslearner/)

`crosslearner` implements the **Adversarial–Consistency X-learner (AC‑X)**. This variant of the X‑learner augments outcome models with an adversarial consistency term. The library is geared towards reproducible research and ships with many GAN tricks and benchmarking utilities.

## Background

The X-learner approach estimates conditional treatment effects by combining outcome regressions for the treated and control groups with imputed pseudo-outcomes. See Küenzel *et&nbsp;al.* (2019) for details on the X-learner methodology.

Recent work has explored adversarial training for counterfactual inference. In particular, Yoon, Jordon and van&nbsp;der Schaar (2018) introduced GANITE, demonstrating how generative adversarial networks can estimate individual treatment effects. AC‑X builds on these ideas by enforcing a consistency constraint between potential outcome heads and a dedicated treatment‑effect head.

## AC‑X Objective

Let $x$ denote covariates, $t \in \{0,1\}$ a binary treatment indicator and $y$ the observed outcome. The model predicts potential outcomes $\hat\mu_0(x)$ and $\hat\mu_1(x)$ as well as an explicit effect $\hat\tau(x)$. AC‑X minimises

$$
\mathcal{L} 
= \sum_i \ell\bigl(y_i,\hat\mu_{t_i}(x_i)\bigr)
  + \beta\_{\mathrm{cons}} \lVert \hat\tau(x_i) - (\hat\mu_1(x_i)-\hat\mu_0(x_i)) \rVert^2
  + \gamma\_{\mathrm{adv}} \, \mathcal{L}\_{\mathrm{adv}},
$$

where $\ell$ is the squared error and $\mathcal{L}\_{\mathrm{adv}}$ is the standard discriminator loss encouraging generated counterfactual pairs to be indistinguishable from real data.

## Installation

Install from source:

```bash
pip install .
```

To run a quick training loop:

```bash
crosslearner-train
```

This command trains an AC‑X model on a small synthetic dataset and prints the final $\sqrt{\mathrm{PEHE}}$ metric. Loss histories are optionally logged to TensorBoard for experiment tracking.

## Benchmarking

To benchmark across available datasets:

```bash
crosslearner-benchmarks all --replicates 1 --epochs 1
```

This command trains a small model on several built-in datasets and reports the
mean $\sqrt{\mathrm{PEHE}}$ for each task. In this environment the benchmark
uses the ``toy``, ``complex``, ``iris``, ``ihdp`` and ``confounded`` datasets.

To additionally compare with baseline models, pass ``--baselines`` (or use the
``crosslearner-benchmark`` alias):

```bash
crosslearner-benchmarks toy --baselines --replicates 1 --epochs 1
```

Sample output with ``--replicates 1 --epochs 1``:

| dataset    | $\sqrt{\mathrm{PEHE}}$ |
|------------|-----------------------:|
| ``toy``    | **1.26** |
| ``complex``| **1.27** |
| ``iris``   | **0.85** |
| ``ihdp``   | **4.14** |
| ``confounded`` | **0.81** |

## Hyperparameter Sweeps

`optuna` can automate tuning of the many options exposed by
`train_acx`. Define an objective that trains a model with parameters
sampled from a search space and returns the validation
\(\sqrt{\mathrm{PEHE}}\). Running the study explores different
configurations and reports the best one:

```python
import optuna
import torch
from crosslearner.datasets.toy import get_toy_dataloader
from crosslearner.training.train_acx import train_acx
from crosslearner.training import ModelConfig, TrainingConfig
from crosslearner.evaluation.evaluate import evaluate

loader, (mu0, mu1) = get_toy_dataloader()
X = torch.cat([b[0] for b in loader])
mu0_all = mu0
mu1_all = mu1

def objective(trial):
    model_cfg = ModelConfig(
        p=10,
        rep_dim=trial.suggest_int("rep_dim", 32, 128),
    )
    train_cfg = TrainingConfig(
        epochs=30,
        lr_g=trial.suggest_loguniform("lr_g", 1e-4, 1e-2),
        lr_d=trial.suggest_loguniform("lr_d", 1e-4, 1e-2),
        beta_cons=trial.suggest_float("beta_cons", 1.0, 20.0),
    )
    return evaluate(
        train_acx(loader, model_cfg, train_cfg),
        X,
        mu0_all,
        mu1_all,
    )

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=50)
```

See :doc:`hyperparameter_sweeps` in the documentation for more details.

## Experiment Manager

To streamline large-scale experiments the package ships with
``ExperimentManager`` which marries cross-validation, Optuna searches and
TensorBoard logging. It takes a data loader and true potential outcomes and
automatically performs repeated training and evaluation:

```python
from crosslearner.datasets.toy import get_toy_dataloader
from crosslearner.experiments import ExperimentManager
import optuna

loader, (mu0, mu1) = get_toy_dataloader()
manager = ExperimentManager(loader, mu0, mu1, p=10, folds=3, log_dir="runs")

def space(trial: optuna.Trial) -> dict:
    return {"rep_dim": trial.suggest_int("rep_dim", 32, 64), "epochs": 5}

study = manager.optimize(space, n_trials=10)
print("best PEHE", study.best_value)
```

Each fold and trial is logged under ``log_dir`` to make results reproducible.

Using the default training hyper-parameters (``rep_dim=64``, ``lr=1e-3`` and a
single hidden layer of size 128), running ``ExperimentManager`` for two epochs
with three cross-validation folds yields the following mean
cross-validated $\sqrt{\mathrm{PEHE}}$ across the built-in synthetic datasets:

| dataset            | $p$ | confounding | $\sqrt{\mathrm{PEHE}}$ |
|--------------------|----:|------------:|-----------------------:|
| ``toy``            | 10 | -- | **0.69** |
| ``complex``        | 20 | -- | **0.95** |
| ``confounding=0.5``| 10 | 0.5 | **0.59** |
| ``aircraft``       | 5  | -- | **430.87** |

## Repository Layout

- `crosslearner/models/` – model definitions including `ACX`.
- `crosslearner/datasets/` – data loaders for IHDP, Jobs, ACIC, Twins, LaLonde and synthetic generators.
  The ACIC loaders attempt to download `.npz` files from GitHub. If the URLs are
  unavailable, download the files manually and place them under
  `crosslearner/datasets/_data`.
  See `docs/datasets.rst` for a description of each loader.
- `crosslearner/training/` – training utilities and GAN tricks.
- `crosslearner/evaluation/` – metrics such as PEHE.
- `crosslearner/configs/` – YAML configs with hyper‑parameters.

The training code exposes options for Wasserstein loss with gradient penalty, hinge and least‑squares objectives, spectral normalisation, feature matching, MMD regularisation, exponential moving average, instance noise, PacGAN-style discriminator packing, gradient reversal and two‑time‑scale update rule (TTUR). Early stopping on validation $\sqrt{\mathrm{PEHE}}$ is also provided. Dropout can be configured separately for the representation, head and discriminator networks.

Use the config file as a starting point for your own experiments on IHDP, ACIC or other datasets.

## Visualisation

The `History` returned by `train_acx` can be passed to
`crosslearner.visualization.plot_losses` to plot generator, discriminator and
auxiliary losses. Gradient norms and learning rates can be visualised with
`plot_grad_norms` and `plot_learning_rates`. The module also provides several
utilities for exploring model
behaviour:

- `scatter_tau` produces a scatter plot of predicted versus true treatment
  effects.
- `plot_tau_distribution` shows the distribution of estimated effects.
- `plot_covariate_balance` visualises covariate balance using standardised mean
  differences.
- `plot_propensity_overlap` plots propensity score overlap.
- `plot_residuals` displays residuals against predictions.
- `plot_partial_dependence` visualises how the predicted effect varies with a
  single covariate by averaging over the others.
- `plot_ice` plots individual conditional expectation curves for a feature.

## Model Export

Models can be scripted or exported to ONNX with
`crosslearner.export.export_model`:

```python
from crosslearner.models.acx import ACX
from crosslearner.export import export_model
import torch

model = ACX(p=10)
x = torch.randn(1, 10)
export_model(model, x, "acx.pt")
export_model(model, x, "acx.onnx", onnx=True)
```

## Uncertainty Estimation

To quantify prediction confidence you can perform Monte Carlo dropout at
inference time. The helper
`predict_tau_mc_dropout` runs multiple forward passes with dropout enabled and
returns the mean and standard deviation of the treatment effect:

```python
from crosslearner.evaluation import predict_tau_mc_dropout

mean, std = predict_tau_mc_dropout(model, X, passes=50)
```

Alternatively you can train multiple models and average their predictions:

```python
from crosslearner.training import train_acx_ensemble
from crosslearner.evaluation import predict_tau_ensemble

models = train_acx_ensemble(loader, model_cfg, train_cfg, n_models=5)
mean, std = predict_tau_ensemble(models, X)
```

## Documentation

Hosted documentation is available at [https://mattsq.github.io/crosslearner/](https://mattsq.github.io/crosslearner/).

API documentation is built with Sphinx. Run the following commands to generate
HTML docs in `docs/_build/html`:

```bash
pip install sphinx
make -C docs html
```

## References

- J. Küenzel, J. Sekhon, P. Bickel, and B. Yu. *The X-Learner for Estimating Individualized Treatment Effects*. (2019).
- J. Yoon, M. Jordon, and M. van der Schaar. *GANITE: Estimation of Individualized Treatment Effects using Generative Adversarial Nets*. (2018).
