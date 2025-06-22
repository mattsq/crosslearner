import torch
import torch.nn as nn
import pytest
from crosslearner.models.acx import ACX
from crosslearner.training.config import ModelConfig


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


def test_acx_dropout_layers():
    model = ACX(p=2, phi_dropout=0.1, head_dropout=0.2, disc_dropout=0.3)
    assert any(isinstance(m, nn.Dropout) for m in model.phi.net.modules())
    assert any(isinstance(m, nn.Dropout) for m in model.mu0.net.modules())
    assert any(isinstance(m, nn.Dropout) for m in model.disc.net.modules())


def test_acx_residual_option():
    model = ACX(p=3, residual=True)
    assert model.phi.residual is True
    assert model.mu0.residual is True
    X = torch.randn(2, 3)
    h, _, _, _ = model(X)
    assert h.shape[0] == 2


def test_acx_partial_residual():
    model = ACX(p=3, phi_residual=True, head_residual=False, disc_residual=False)
    assert model.phi.residual is True
    assert model.mu0.residual is False
    assert model.disc.residual is False


def test_acx_disc_pack():
    model = ACX(p=3, disc_pack=2)
    assert model.disc_pack == 2
    assert model.disc.net[0][0].in_features == 2 * (64 + 2)


def test_acx_moe_forward():
    model = ACX(p=3, moe_experts=2)
    X = torch.randn(2, 3)
    h, m0, m1, tau = model(X)
    assert h.shape == (2, 64)
    assert m0.shape == (2, 1)
    assert m1.shape == (2, 1)
    assert tau.shape == (2, 1)
    assert model.use_moe
    assert model.moe.gates.shape == (2, 2)


def test_acx_tau_variance():
    model = ACX(p=2, tau_heads=3)
    X = torch.randn(4, 2)
    _ = model(X)
    var = model.tau_variance
    assert var.shape == (4, 1)
    assert torch.any(var >= 0)


def test_model_config_invalid_tau_heads():
    with pytest.raises(ValueError):
        ModelConfig(p=2, tau_heads=0)
