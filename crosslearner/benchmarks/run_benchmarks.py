import argparse
import os
import urllib.request
from typing import List

import torch
from torch.utils.data import DataLoader, TensorDataset

from crosslearner.datasets.toy import get_toy_dataloader
from crosslearner.datasets.complex import get_complex_dataloader
from crosslearner.datasets.ihdp import get_ihdp_dataloader
from crosslearner.datasets.jobs import get_jobs_dataloader
from crosslearner.training.train_acx import train_acx
from crosslearner.evaluation.evaluate import evaluate


def load_external_iris(batch_size: int = 256, seed: int | None = None):
    """Download the Iris dataset and create a synthetic causal task."""
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


def run(dataset: str, replicates: int = 3, epochs: int = 30) -> List[float]:
    """Run the benchmark for the given dataset and return list of PEHE values."""
    results = []
    for seed in range(replicates):
        if dataset == "toy":
            loader, (mu0, mu1) = get_toy_dataloader()
            p = 10
        elif dataset == "complex":
            loader, (mu0, mu1) = get_complex_dataloader(seed=seed)
            p = 20
        elif dataset == "iris":
            loader, (mu0, mu1) = load_external_iris(seed=seed)
            p = 4
        elif dataset == "ihdp":
            loader, (mu0, mu1) = get_ihdp_dataloader(seed=seed)
            p = loader.dataset.tensors[0].size(1)
        elif dataset == "jobs":
            loader, (mu0, mu1) = get_jobs_dataloader()
            p = loader.dataset.tensors[0].size(1)
        elif dataset == "all":
            all_ds = ["toy", "complex", "iris", "ihdp", "jobs"]
            summary = []
            for ds in all_ds:
                res = run(ds, replicates, epochs)
                summary.append((ds, sum(res) / len(res)))
            for ds, val in summary:
                print(f"{ds}\t{val:.3f}")
            return [v for _, v in summary]
        else:
            raise ValueError(f"Unknown dataset {dataset}")
        model = train_acx(loader, p=p, epochs=epochs)
        X = torch.cat([b[0] for b in loader])
        mu0_all = mu0
        mu1_all = mu1
        pehe_val = evaluate(model, X, mu0_all, mu1_all)
        print(f"replicate {seed}: sqrt PEHE {pehe_val:.3f}")
        results.append(pehe_val)
    mean_pehe = sum(results) / len(results)
    print(f"mean sqrt PEHE: {mean_pehe:.3f}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Run CrossLearner benchmarks")
    parser.add_argument(
        "dataset",
        choices=["toy", "complex", "iris", "ihdp", "jobs", "all"],
        help="dataset to benchmark or 'all'",
    )
    parser.add_argument("--replicates", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=30)
    args = parser.parse_args()
    run(args.dataset, args.replicates, args.epochs)


if __name__ == "__main__":
    main()
