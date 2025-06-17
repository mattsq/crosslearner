"""Command-line entry points for running benchmarks."""

import argparse
import os
import urllib.request
from typing import Callable, Dict, List

import torch
from torch.utils.data import DataLoader, TensorDataset

from crosslearner.datasets.toy import get_toy_dataloader
from crosslearner.datasets.complex import get_complex_dataloader
from crosslearner.datasets.ihdp import get_ihdp_dataloader
from crosslearner.datasets.jobs import get_jobs_dataloader
from crosslearner.datasets.acic2016 import get_acic2016_dataloader
from crosslearner.datasets.acic2018 import get_acic2018_dataloader
from crosslearner.datasets.twins import get_twins_dataloader
from crosslearner.datasets.lalonde import get_lalonde_dataloader
from crosslearner.datasets.synthetic import get_confounding_dataloader
from crosslearner.training.train_acx import train_acx
from crosslearner.evaluation.evaluate import evaluate, evaluate_dr
from crosslearner.evaluation.metrics import (
    policy_risk,
    ate_error,
    att_error,
    bootstrap_ci,
)
from crosslearner.utils import set_seed


def load_external_iris(batch_size: int = 256, seed: int | None = None):
    """Download the Iris dataset and create a synthetic causal task.

    Args:
        batch_size: Size of the batches.
        seed: Optional random seed for reproducibility.

    Returns:
        Data loader and tuple of true potential outcomes.
    """
    url = (
        "https://raw.githubusercontent.com/scikit-learn/"
        "scikit-learn/main/sklearn/datasets/data/iris.csv"
    )
    fname = os.path.join(os.path.dirname(__file__), "iris.csv")
    if not os.path.exists(fname):
        urllib.request.urlretrieve(url, fname)
    data = []
    with open(fname) as f:
        next(f)  # header
        for line in f:
            parts = line.strip().split(",")
            data.append([float(p) for p in parts[:4]])
    X = torch.tensor(data)
    if seed is not None:
        gen = torch.Generator().manual_seed(seed)
    else:
        gen = None
    T_prob = torch.sigmoid(X[:, 0] / 2)
    T = torch.bernoulli(T_prob, generator=gen).float().unsqueeze(-1)
    mu0 = X.sum(1, keepdim=True) / 10
    mu1 = mu0 + torch.sin(X[:, 1:3].sum(1, keepdim=True))
    Y = torch.where(T.bool(), mu1, mu0) + 0.1 * torch.randn(X.size(0), 1, generator=gen)
    loader = DataLoader(TensorDataset(X, T, Y), batch_size=batch_size, shuffle=True)
    return loader, (mu0, mu1)


# Mapping from dataset name to loader callable accepting a seed and returning
# ``(loader, (mu0, mu1))``. Datasets without a notion of seed simply ignore the
# argument.
DATASET_LOADERS: Dict[
    str, Callable[[int], tuple[DataLoader, tuple[torch.Tensor, torch.Tensor]]]
] = {
    "toy": lambda seed: get_toy_dataloader(),
    "complex": lambda seed: get_complex_dataloader(seed=seed),
    "iris": lambda seed: load_external_iris(seed=seed),
    "ihdp": lambda seed: get_ihdp_dataloader(seed=seed),
    "jobs": lambda seed: get_jobs_dataloader(),
    "acic2016": lambda seed: get_acic2016_dataloader(seed=seed),
    "acic2018": lambda seed: get_acic2018_dataloader(seed=seed),
    "twins": lambda seed: get_twins_dataloader(),
    "lalonde": lambda seed: get_lalonde_dataloader(),
    "confounded": lambda seed: get_confounding_dataloader(seed=seed),
}

# Subset of datasets used when requesting ``"all"``.
ALL_DATASETS = ["toy", "complex", "iris", "ihdp", "confounded"]


def run(dataset: str, replicates: int = 3, epochs: int = 30) -> List[Dict[str, float]]:
    """Run the benchmark for the given dataset and return metrics per replicate.

    Args:
        dataset: One of ``toy``, ``complex``, ``iris``, ``ihdp`` or ``jobs``.
        replicates: Number of random seeds to evaluate.
        epochs: Training epochs per replicate.

    Returns:
        List of dictionaries with metrics for each replicate.
    """
    if dataset == "all":
        summary = []
        for ds in ALL_DATASETS:
            res = run(ds, replicates, epochs)
            mean_pehe_ds = sum(r["pehe"] for r in res) / len(res)
            summary.append((ds, mean_pehe_ds))
        for ds, val in summary:
            print(f"{ds}\t{val:.3f}")
        return [v for _, v in summary]

    if dataset not in DATASET_LOADERS:
        raise ValueError(f"Unknown dataset {dataset}")

    results: List[Dict[str, float]] = []
    loader_fn = DATASET_LOADERS[dataset]
    for seed in range(replicates):
        set_seed(seed)
        loader, (mu0, mu1) = loader_fn(seed)
        p = loader.dataset.tensors[0].size(1)
        model = train_acx(loader, p=p, epochs=epochs, seed=seed)
        X = torch.cat([b[0] for b in loader])
        T_all = torch.cat([b[1] for b in loader])
        Y_all = torch.cat([b[2] for b in loader])
        with torch.no_grad():
            _, _, _, tau_hat = model(X)

        if mu0 is None or mu1 is None:
            # Off-policy dataset without counterfactual outcomes
            propensity = torch.full_like(T_all, T_all.float().mean().item())
            pehe_val = evaluate_dr(model, X, T_all, Y_all, propensity)
            risk_val = float("nan")
            ate_err = float("nan")
            att_err = float("nan")
            coverage = float("nan")
            print(f"replicate {seed}: DR sqrt PEHE {pehe_val:.3f}")
        else:
            mu0_all = mu0
            mu1_all = mu1
            tau_true = mu1_all - mu0_all
            pehe_val = evaluate(model, X, mu0_all, mu1_all)
            risk_val = policy_risk(tau_hat, mu0_all, mu1_all)
            ate_err = ate_error(tau_hat, mu0_all, mu1_all)
            att_err = att_error(tau_hat, mu0_all, mu1_all, T_all)
            ci_low, ci_high = bootstrap_ci(tau_hat)
            coverage = float(ci_low <= tau_true.mean().item() <= ci_high)
            print(
                f"replicate {seed}: sqrt PEHE {pehe_val:.3f} policy risk {risk_val:.3f}"
            )

        results.append(
            {
                "pehe": pehe_val,
                "policy_risk": risk_val,
                "ate_error": ate_err,
                "att_error": att_err,
                "coverage": coverage,
            }
        )
    mean_pehe = sum(r["pehe"] for r in results) / len(results)
    mean_risk = sum(r["policy_risk"] for r in results) / len(results)
    mean_ate_err = sum(r["ate_error"] for r in results) / len(results)
    mean_att_err = sum(r["att_error"] for r in results) / len(results)
    coverage_rate = sum(r["coverage"] for r in results) / len(results)
    print(
        "mean sqrt PEHE: {:.3f} policy risk: {:.3f} ATE err: {:.3f} ATT err: {:.3f} coverage: {:.2f}".format(
            mean_pehe, mean_risk, mean_ate_err, mean_att_err, coverage_rate
        )
    )
    return results


def main():
    parser = argparse.ArgumentParser(description="Run CrossLearner benchmarks")
    parser.add_argument(
        "dataset",
        choices=[
            "toy",
            "complex",
            "iris",
            "ihdp",
            "jobs",
            "acic2016",
            "acic2018",
            "twins",
            "lalonde",
            "confounded",
            "all",
        ],
        help="dataset to benchmark or 'all'",
    )
    parser.add_argument("--replicates", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=30)
    args = parser.parse_args()
    run(args.dataset, args.replicates, args.epochs)


if __name__ == "__main__":
    main()
