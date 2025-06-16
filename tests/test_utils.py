import pytest
import torch
import torch.nn as nn

from crosslearner.models.acx import MLP, _get_activation


def test_get_activation_invalid_name():
    with pytest.raises(ValueError):
        _get_activation("invalid")


def test_get_activation_callable_roundtrip():
    fn = _get_activation(nn.ReLU)
    assert fn is nn.ReLU


def test_mlp_dropout_range_errors():
    with pytest.raises(ValueError):
        MLP(4, 2, dropout=-0.1)
    with pytest.raises(ValueError):
        MLP(4, 2, dropout=1.0)


def test_mlp_residual_forward():
    mlp = MLP(4, 4, hidden=(4,), residual=True)
    x = torch.randn(2, 4)
    y = mlp(x)
    assert y.shape == (2, 4)
