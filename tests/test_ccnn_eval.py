import torch

from causal_consistency_nn.config import SyntheticDataConfig
from causal_consistency_nn.data.synthetic import generate_synthetic
from causal_consistency_nn.metrics import ate
from causal_consistency_nn.eval import evaluate
from causal_consistency_nn.serve import impute_y
from crosslearner.models.acx import ACX


def test_ate_function():
    cfg = SyntheticDataConfig(n_samples=8, p=3, noise=0.1, seed=0)
    dataset, (mu0, mu1) = generate_synthetic(cfg)
    est = ate(mu0, mu1)
    assert abs(est - (mu1 - mu0).mean().item()) < 1e-6


def test_evaluate_synthetic():
    cfg = SyntheticDataConfig(n_samples=20, p=3, noise=0.1, seed=0)
    dataset, (mu0, mu1) = generate_synthetic(cfg)
    good = evaluate(mu0, mu1, dataset, mu0_true=mu0, mu1_true=mu1, noise=cfg.noise)
    zeros = torch.zeros_like(mu0)
    bad = evaluate(zeros, zeros, dataset, mu0_true=mu0, mu1_true=mu1, noise=cfg.noise)
    assert abs(good["ate_error"]) < 1e-6
    assert good["log_likelihood"] > bad["log_likelihood"]


def test_impute_y_replaces_missing():
    cfg = SyntheticDataConfig(n_samples=10, p=3, noise=0.1, missing_y_prob=1.0, seed=0)
    dataset, _ = generate_synthetic(cfg)
    X, T, Y = dataset.tensors
    model = ACX(p=cfg.p)
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()
    out = impute_y(model, X, T, Y)
    assert torch.isnan(out).sum() == 0
    assert torch.allclose(out, torch.zeros_like(out))
