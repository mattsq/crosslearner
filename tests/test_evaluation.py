import torch
import pytest

from crosslearner.evaluation.metrics import pehe
from crosslearner.evaluation.evaluate import evaluate, evaluate_ipw, evaluate_dr
from crosslearner.models.acx import ACX


def test_pehe_simple():
    tau_hat = torch.tensor([0.0, 1.0])
    tau_true = torch.tensor([0.0, 0.0])
    expected = (0.5) ** 0.5
    assert abs(pehe(tau_hat, tau_true) - expected) < 1e-6


def test_evaluate_zero_error():
    model = ACX(p=2)
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()
        model.tau.net[-1].bias.fill_(1.0)
    X = torch.zeros(3, 2)
    mu0 = torch.zeros(3, 1)
    mu1 = torch.ones(3, 1)
    metric = evaluate(model, X, mu0, mu1)
    assert metric < 1e-6


def test_evaluate_ipw_zero_error():
    model = ACX(p=2)
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()
        model.mu1.net[-1].bias.fill_(1.0)
        model.tau.net[-1].bias.fill_(1.0)
    X = torch.zeros(4, 2)
    T = torch.tensor([[0], [1], [0], [1]], dtype=torch.float32)
    Y = T.clone().float()
    propensity = torch.full((4, 1), 0.5)
    metric = evaluate_ipw(model, X, T, Y, propensity)
    assert metric >= 0.0


def test_evaluate_dr_zero_error():
    model = ACX(p=2)
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()
        model.mu1.net[-1].bias.fill_(1.0)
        model.tau.net[-1].bias.fill_(1.0)
    X = torch.zeros(4, 2)
    T = torch.tensor([[0], [1], [0], [1]], dtype=torch.float32)
    Y = T.clone().float()
    propensity = torch.full((4, 1), 0.5)
    metric = evaluate_dr(model, X, T, Y, propensity)
    assert metric < 1e-6


def test_evaluate_device_mismatch():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    model = ACX(p=2).cuda()
    X = torch.zeros(2, 2)
    mu0 = torch.zeros(2, 1)
    mu1 = torch.ones(2, 1)
    metric = evaluate(model, X, mu0, mu1)
    assert isinstance(metric, float)
