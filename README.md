# crosslearner

`crosslearner` implements the **Adversarial–Consistency X-learner (AC‑X)**, a variant of the X-learner that augments outcome models with an adversarial consistency term. The package is designed for reproducible research with numerous GAN tricks and benchmarking utilities.

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

Run a toy training loop:

```bash
crosslearner-train
```

The script trains an AC‑X model on a small synthetic dataset and prints the final $\sqrt{\mathrm{PEHE}}$ metric. Loss histories are optionally logged to TensorBoard for experiment tracking.

## Benchmarking

To benchmark across available datasets:

```bash
crosslearner-benchmarks all --replicates 1 --epochs 1
```

This downloads the IHDP and Jobs datasets and prints the mean $\sqrt{\mathrm{PEHE}}$ for each task.

## Repository Layout

- `crosslearner/models/` – model definitions including `ACX`.
- `crosslearner/datasets/` – data loaders (currently a toy synthetic generator).
- `crosslearner/training/` – training utilities and GAN tricks.
- `crosslearner/evaluation/` – metrics such as PEHE.
- `crosslearner/configs/` – YAML configs with hyper‑parameters.

The training code exposes options for Wasserstein loss with gradient penalty, spectral normalisation, feature matching, instance noise, gradient reversal and two‑time‑scale update rule (TTUR). Early stopping on validation $\sqrt{\mathrm{PEHE}}$ is also provided.

Use the config file as a starting point for your own experiments on IHDP, ACIC or other datasets.

## Visualisation

The `History` returned by `train_acx` can be passed to `crosslearner.visualization.plot_losses` to plot generator and discriminator losses. The module also provides `crosslearner.visualization.scatter_tau` for a scatter plot of predicted versus true treatment effects.

## References

- J. Küenzel, J. Sekhon, P. Bickel, and B. Yu. *The X-Learner for Estimating Individualized Treatment Effects*. (2019).
- J. Yoon, M. Jordon, and M. van der Schaar. *GANITE: Estimation of Individualized Treatment Effects using Generative Adversarial Nets*. (2018).
