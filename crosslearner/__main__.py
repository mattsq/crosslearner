"""Command-line interface for quick experimentation."""

import torch

from crosslearner.utils import set_seed

from crosslearner.datasets.toy import get_toy_dataloader
from crosslearner.training.train_acx import train_acx
from crosslearner.training.config import ModelConfig, TrainingConfig
from crosslearner.evaluation.evaluate import evaluate


def main() -> None:
    """Run the toy training loop from the command line."""
    set_seed(0)
    loader, (mu0, mu1) = get_toy_dataloader()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_cfg = ModelConfig(p=10)
    train_cfg = TrainingConfig()
    model = train_acx(loader, model_cfg, train_cfg, device=device)
    X = torch.cat([b[0] for b in loader]).to(device)
    mu0_all = mu0.to(device)
    mu1_all = mu1.to(device)
    metric = evaluate(model, X, mu0_all, mu1_all)
    print("\nsqrt(PEHE) (lower is better):", metric)


if __name__ == "__main__":
    main()
