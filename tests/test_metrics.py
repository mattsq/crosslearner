import torch
from crosslearner.evaluation.metrics import pehe


def test_pehe():
    tau_hat = torch.tensor([0.0, 1.0])
    tau_true = torch.tensor([0.0, 0.0])
    result = pehe(tau_hat, tau_true)
    expected = (0.5) ** 0.5
    assert abs(result - expected) < 1e-6
