"""Benchmark ACX and baseline models on built-in datasets."""

from __future__ import annotations

import argparse
from typing import Dict, List, Type

import numpy as np
import torch

from .run_benchmarks import DATASET_LOADERS, ALL_DATASETS
from crosslearner.training.train_acx import train_acx
from crosslearner.models.baselines import DRLearner, SLearner, TLearner, XLearner
from crosslearner.evaluation.evaluate import evaluate, evaluate_dr
from crosslearner.evaluation.metrics import (
    policy_risk,
    ate_error,
    att_error,
    bootstrap_ci,
    pehe,
)
from crosslearner.utils import set_seed

BASELINES: Dict[str, Type] = {
    "slearner": SLearner,
    "tlearner": TLearner,
    "xlearner": XLearner,
    "drlearner": DRLearner,
}


def _baseline_mus(model, X: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    if isinstance(model, SLearner) or isinstance(model, TLearner):
        mu1 = model._predict_mu1(X)
        mu0 = model._predict_mu0(X)
    elif isinstance(model, XLearner):
        mu1 = model.t.model_t.predict(X)
        mu0 = model.t.model_c.predict(X)
    elif isinstance(model, DRLearner):
        mu0 = model.model_mu0.predict(X)
        mu1 = model.model_mu1.predict(X)
    else:  # pragma: no cover - unexpected
        raise TypeError("Unknown model type")
    mu0_t = torch.tensor(mu0, dtype=torch.float32).reshape(-1, 1)
    mu1_t = torch.tensor(mu1, dtype=torch.float32).reshape(-1, 1)
    return mu0_t, mu1_t


def _evaluate_baseline(
    model, X: np.ndarray, T: torch.Tensor, Y: torch.Tensor, mu0, mu1
) -> Dict[str, float]:
    tau_hat = model.predict_tau(X)
    if mu0 is None or mu1 is None:
        mu0_hat, mu1_hat = _baseline_mus(model, X)
        propensity = torch.full_like(T, T.float().mean().item())
        mu_hat = T * mu1_hat + (1.0 - T) * mu0_hat
        pseudo = (
            (T - propensity) / (propensity * (1.0 - propensity)) * (Y - mu_hat)
            + mu1_hat
            - mu0_hat
        )
        pehe_val = pehe(tau_hat, pseudo)
        risk_val = float("nan")
        ate_err = float("nan")
        att_err = float("nan")
        coverage = float("nan")
    else:
        tau_true = mu1 - mu0
        pehe_val = pehe(tau_hat, tau_true)
        risk_val = policy_risk(tau_hat, mu0, mu1)
        ate_err = ate_error(tau_hat, mu0, mu1)
        att_err = att_error(tau_hat, mu0, mu1, T)
        ci_low, ci_high = bootstrap_ci(tau_hat)
        coverage = float(ci_low <= tau_true.mean().item() <= ci_high)
    return {
        "pehe": pehe_val,
        "policy_risk": risk_val,
        "ate_error": ate_err,
        "att_error": att_err,
        "coverage": coverage,
    }


def run(
    dataset: str, replicates: int = 3, epochs: int = 30
) -> List[Dict[str, Dict[str, float]]]:
    """Run benchmarks comparing ACX to baseline models."""
    if dataset == "all":
        summary = []
        for ds in ALL_DATASETS:
            res = run(ds, replicates, epochs)
            mean_pehe_ds = sum(r["acx"]["pehe"] for r in res) / len(res)
            summary.append((ds, mean_pehe_ds))
        for ds, val in summary:
            print(f"{ds}\t{val:.3f}")
        return [v for _, v in summary]

    if dataset not in DATASET_LOADERS:
        raise ValueError(f"Unknown dataset {dataset}")

    loader_fn = DATASET_LOADERS[dataset]
    results: List[Dict[str, Dict[str, float]]] = []
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
            propensity = torch.full_like(T_all, T_all.float().mean().item())
            pehe_val = evaluate_dr(model, X, T_all, Y_all, propensity)
            risk_val = float("nan")
            ate_err = float("nan")
            att_err = float("nan")
            coverage = float("nan")
            print(f"replicate {seed}: ACX DR sqrt PEHE {pehe_val:.3f}")
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
                f"replicate {seed}: ACX sqrt PEHE {pehe_val:.3f} policy risk {risk_val:.3f}"
            )

        metrics = {
            "acx": {
                "pehe": pehe_val,
                "policy_risk": risk_val,
                "ate_error": ate_err,
                "att_error": att_err,
                "coverage": coverage,
            }
        }

        X_np = X.numpy()
        T_np = T_all.numpy()
        Y_np = Y_all.numpy()
        for name, cls in BASELINES.items():
            baseline = cls(p=p)
            baseline.fit(X_np, T_np, Y_np)
            b_metrics = _evaluate_baseline(baseline, X_np, T_all, Y_all, mu0, mu1)
            print(f"replicate {seed}: {name} sqrt PEHE {b_metrics['pehe']:.3f}")
            metrics[name] = b_metrics

        results.append(metrics)

    # print means
    models = results[0].keys()
    for m in models:
        mean_pehe = sum(r[m]["pehe"] for r in results) / len(results)
        mean_risk = sum(r[m]["policy_risk"] for r in results) / len(results)
        mean_ate_err = sum(r[m]["ate_error"] for r in results) / len(results)
        mean_att_err = sum(r[m]["att_error"] for r in results) / len(results)
        coverage_rate = sum(r[m]["coverage"] for r in results) / len(results)
        print(
            f"{m}: mean sqrt PEHE {mean_pehe:.3f} policy risk {mean_risk:.3f} ATE err {mean_ate_err:.3f} ATT err {mean_att_err:.3f} coverage {coverage_rate:.2f}"
        )

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run CrossLearner benchmark with baselines"
    )
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


if __name__ == "__main__":  # pragma: no cover
    main()
