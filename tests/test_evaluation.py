import torch

from crosslearner.evaluation.metrics import pehe
from crosslearner.evaluation.evaluate import evaluate
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
