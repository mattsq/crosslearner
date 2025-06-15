import torch
from crosslearner.evaluation.metrics import (
    pehe,
    policy_risk,
    ate_error,
    att_error,
    bootstrap_ci,
)


def test_pehe():
    tau_hat = torch.tensor([0.0, 1.0])
    tau_true = torch.tensor([0.0, 0.0])
    result = pehe(tau_hat, tau_true)
    expected = (0.5) ** 0.5
    assert abs(result - expected) < 1e-6


def test_additional_metrics():
    mu0 = torch.zeros(4, 1)
    mu1 = torch.ones(4, 1)
    tau_hat = torch.tensor([[0.5], [0.5], [-0.5], [-0.5]])
    risk = policy_risk(tau_hat, mu0, mu1)
    assert risk >= 0.0
    t = torch.tensor([[0], [1], [0], [1]])
    tau_hat = torch.ones(4, 1)
    assert abs(ate_error(tau_hat, mu0, mu1)) < 1e-6
    assert abs(att_error(tau_hat, mu0, mu1, t)) < 1e-6
    low, high = bootstrap_ci(tau_hat.view(-1), n_boot=10)
    assert low <= high
