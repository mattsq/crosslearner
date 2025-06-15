"""Example script for running a toy experiment."""

import torch

from crosslearner.datasets.toy import get_toy_dataloader
from crosslearner.training.train_acx import train_acx
from crosslearner.evaluation.evaluate import evaluate


def main():
    """Train a toy model and print the PEHE."""

    loader, (mu0, mu1) = get_toy_dataloader()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = train_acx(loader, p=10, device=device)
    X = torch.cat([b[0] for b in loader]).to(device)
    mu0_all = mu0.to(device)
    mu1_all = mu1.to(device)
    metric = evaluate(model, X, mu0_all, mu1_all)
    print("\n\u221aPEHE (lower is better):", metric)


if __name__ == "__main__":
    main()
