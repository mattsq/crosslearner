import torch

from crosslearner.models.acx import ACX
from crosslearner.evaluation.uncertainty import predict_tau_mc_dropout


def test_mc_dropout_shapes_and_variance():
    model = ACX(p=2, phi_dropout=0.5, head_dropout=0.5)
    X = torch.randn(5, 2)
    mean, std = predict_tau_mc_dropout(model, X, passes=20)
    assert mean.shape == (5, 1)
    assert std.shape == (5, 1)
    assert torch.any(std > 0)
