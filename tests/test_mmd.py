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
    n_x = x.size(0)
    n_y = y.size(0)
    if n_x > 1:
        k_xx.fill_diagonal_(0)
        term_x = k_xx.sum() / (n_x * (n_x - 1))
    else:
        term_x = 0.0
    if n_y > 1:
        k_yy.fill_diagonal_(0)
        term_y = k_yy.sum() / (n_y * (n_y - 1))
    else:
        term_y = 0.0
    return term_x + term_y - 2 * k_xy.mean()


def _mmd_rbf_old(x: torch.Tensor, y: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
    xy = torch.cat([x, y])
    dist = torch.cdist(xy, xy).pow(2)
    k = torch.exp(-dist / (2 * sigma**2))
    n_x = x.size(0)
    k_xx = k[:n_x, :n_x]
    k_yy = k[n_x:, n_x:]
    k_xy = k[:n_x, n_x:]
    return k_xx.mean() + k_yy.mean() - 2 * k_xy.mean()


def test_mmd_rbf_matches_manual():
    torch.manual_seed(0)
    x = torch.randn(5, 3)
    y = torch.randn(6, 3)
    expected = _mmd_rbf_manual(x, y)
    actual = _mmd_rbf(x, y)
    assert torch.allclose(actual, expected, atol=1e-6)


def test_mmd_rbf_non_default_sigma():
    torch.manual_seed(1)
    x = torch.randn(8, 2)
    y = torch.randn(7, 2)
    sigma = 2.5
    expected = _mmd_rbf_manual(x, y, sigma)
    actual = _mmd_rbf(x, y, sigma)
    assert torch.allclose(actual, expected, atol=1e-6)


def test_mmd_rbf_empty_inputs():
    x = torch.empty(0, 3)
    y = torch.randn(4, 3)
    assert _mmd_rbf(x, y) == 0
    assert _mmd_rbf(y, torch.empty(0, 3)) == 0


def test_mmd_rbf_matches_old_for_small_batch():
    torch.manual_seed(2)
    x = torch.randn(3, 4)
    y = torch.randn(2, 4)
    new = _mmd_rbf(x, y)
    old = _mmd_rbf_old(x, y)
    assert new <= old
