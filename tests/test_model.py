import torch
from crosslearner.models.acx import ACX


def test_acx_forward_shapes():
    model = ACX(p=5)
    X = torch.randn(2, 5)
    h, m0, m1, tau = model(X)
    assert h.shape == (2, 64)
    assert m0.shape == (2, 1)
    assert m1.shape == (2, 1)
    assert tau.shape == (2, 1)
