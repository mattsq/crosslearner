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


def test_acx_custom_architecture():
    model = ACX(
        p=3,
        rep_dim=32,
        phi_layers=(16, 16),
        head_layers=(8,),
        disc_layers=(8,),
        activation="tanh",
    )
    X = torch.randn(4, 3)
    h, *_ = model(X)
    assert h.shape == (4, 32)
