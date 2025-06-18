"""Example script for running a toy experiment."""

import torch

from crosslearner.utils import set_seed, default_device
from crosslearner.datasets.toy import get_toy_dataloader
from crosslearner.training.train_acx import train_acx
from crosslearner.training import ModelConfig, TrainingConfig
from crosslearner.evaluation.evaluate import evaluate


def main():
    """Train a toy model and print the PEHE."""

    set_seed(0)
    loader, (mu0, mu1) = get_toy_dataloader()
    device = default_device()
    model_cfg = ModelConfig(p=10)
    train_cfg = TrainingConfig()
    model = train_acx(loader, model_cfg, train_cfg, device=device)
    X = torch.cat([b[0] for b in loader]).to(device)
    mu0_all = mu0.to(device)
    mu1_all = mu1.to(device)
    metric = evaluate(model, X, mu0_all, mu1_all)
    print("\n\u221aPEHE (lower is better):", metric)


if __name__ == "__main__":
    main()
