# crosslearner

This project provides a research friendly implementation of the **Adversarial–Consistency X-learner (AC‑X)** described in `Prompt.txt`. The code is written in PyTorch and is structured for easy experimentation with adversarial tricks and benchmarking datasets.

## Getting Started

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the toy example training loop:

```bash
python train.py
```

The script trains an AC‑X model on a synthetic dataset and prints the final \sqrt{PEHE}.

## Benchmarking

To run a small benchmark across multiple datasets use:

```bash
python -m crosslearner.benchmarks.run_benchmarks toy --replicates 3
```

Replace `toy` with `complex` for a harder synthetic task or `iris` to automatically download a tiny external dataset and evaluate on it.

## Repository Layout

- `crosslearner/models/` – model definitions including `ACX`.
- `crosslearner/datasets/` – data loaders (currently a toy synthetic generator).
- `crosslearner/training/` – training utilities and GAN tricks.
- `crosslearner/evaluation/` – metrics such as PEHE.
- `crosslearner/configs/` – YAML configs with hyper‑parameters.

The training code exposes toggles for Wasserstein loss, spectral normalisation, feature matching, instance noise, gradient reversal and two‑time‑scale update rule (TTUR) as described in `Prompt.txt`.

Use the config file as a starting point for your own experiments on IHDP, ACIC or other datasets.
