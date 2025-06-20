import torch

from crosslearner.training.trainer import _mmd_rbf


def _mmd_rbf_manual(
    x: torch.Tensor, y: torch.Tensor, sigma: float = 1.0
) -> torch.Tensor:
    diff_x = x.unsqueeze(1) - x.unsqueeze(0)
    diff_y = y.unsqueeze(1) - y.unsqueeze(0)
    diff_xy = x.unsqueeze(1) - y.unsqueeze(0)
    k_xx = torch.exp(-diff_x.pow(2).sum(2) / (2 * sigma**2))
    k_yy = torch.exp(-diff_y.pow(2).sum(2) / (2 * sigma**2))
    k_xy = torch.exp(-diff_xy.pow(2).sum(2) / (2 * sigma**2))
    return k_xx.mean() + k_yy.mean() - 2 * k_xy.mean()


def test_mmd_rbf_matches_manual():
    torch.manual_seed(0)
    x = torch.randn(5, 3)
    y = torch.randn(6, 3)
    expected = _mmd_rbf_manual(x, y)
    actual = _mmd_rbf(x, y)
    assert torch.allclose(actual, expected, atol=1e-6)


def test_mmd_rbf_empty_inputs():
    x = torch.empty(0, 3)
    y = torch.randn(4, 3)
    assert _mmd_rbf(x, y) == 0
    assert _mmd_rbf(y, torch.empty(0, 3)) == 0
