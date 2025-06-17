import pytest
import torch
import torch.nn as nn
import numpy as np
import random

from crosslearner.models.acx import MLP, _get_activation
from crosslearner.utils import set_seed, default_device


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


def test_mlp_forward_matches_sequential_without_residual():
    mlp = MLP(3, 2, hidden=(5,), residual=False)
    x = torch.randn(4, 3)
    y_seq = mlp.net(x)
    y = mlp(x)
    assert torch.allclose(y, y_seq)


def test_set_seed_reproducibility():
    set_seed(123)
    r1 = random.random()
    n1 = np.random.rand(1)
    t1 = torch.rand(1)
    set_seed(123)
    assert random.random() == r1
    assert np.allclose(np.random.rand(1), n1)
    assert torch.allclose(torch.rand(1), t1)


def test_default_device_returns_valid_string():
    dev = default_device()
    assert dev in {"cuda", "cpu"}
    assert isinstance(dev, str)
