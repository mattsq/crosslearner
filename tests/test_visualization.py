import torch
import matplotlib
import pytest

matplotlib.use("Agg")

from crosslearner.visualization import (
    plot_losses,
    scatter_tau,
    plot_tau_distribution,
    plot_covariate_balance,
    plot_propensity_overlap,
    plot_residuals,
    plot_grad_norms,
    plot_learning_rates,
    plot_cate_calibration,
    plot_partial_dependence,
    plot_ice,
)
from crosslearner.training.history import EpochStats
from crosslearner.models.acx import ACX


def test_plot_losses_returns_figure():
    hist = [
        EpochStats(
            epoch=0, loss_d=1.0, loss_g=2.0, loss_y=0.0, loss_cons=0.0, loss_adv=0.0
        )
    ]
    fig = plot_losses(hist)
    assert fig is not None
    matplotlib.pyplot.close(fig)


def test_plot_losses_with_validation_losses():
    hist = [
        EpochStats(
            epoch=0,
            loss_d=0.0,
            loss_g=0.0,
            loss_y=0.0,
            loss_cons=0.0,
            loss_adv=0.0,
            val_pehe=1.0,
            val_loss_y=0.1,
            val_loss_cons=0.2,
            val_loss_adv=0.3,
        )
    ]
    fig = plot_losses(hist)
    assert fig is not None
    matplotlib.pyplot.close(fig)


def test_plot_losses_with_risk_only():
    hist = [
        EpochStats(
            epoch=0,
            loss_d=0.0,
            loss_g=0.0,
            loss_y=0.0,
            loss_cons=0.0,
            loss_adv=0.0,
            val_loss_y=0.1,
            val_loss_cons=0.2,
            val_loss_adv=0.3,
        )
    ]
    fig = plot_losses(hist)
    assert fig is not None
    matplotlib.pyplot.close(fig)


def test_scatter_tau_returns_figure():
    model = ACX(p=3)
    X = torch.randn(4, 3)
    mu0 = torch.zeros(4, 1)
    mu1 = torch.ones(4, 1)
    fig = scatter_tau(model, X, mu0, mu1)
    assert fig is not None
    matplotlib.pyplot.close(fig)


def test_scatter_tau_device_mismatch():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    model = ACX(p=3).cuda()
    X = torch.randn(2, 3)
    mu0 = torch.zeros(2, 1)
    mu1 = torch.ones(2, 1)
    fig = scatter_tau(model, X, mu0, mu1)
    assert fig is not None
    matplotlib.pyplot.close(fig)


def test_plot_tau_distribution_returns_figure():
    tau = torch.randn(10, 1)
    fig = plot_tau_distribution(tau)
    assert fig is not None
    matplotlib.pyplot.close(fig)


def test_plot_covariate_balance_returns_figure():
    X = torch.randn(6, 4)
    T = torch.tensor([0, 1, 0, 1, 0, 1])
    fig = plot_covariate_balance(X, T)
    assert fig is not None
    matplotlib.pyplot.close(fig)


def test_plot_propensity_overlap_returns_figure():
    prop = torch.rand(8, 1)
    T = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1])
    fig = plot_propensity_overlap(prop, T)
    assert fig is not None
    matplotlib.pyplot.close(fig)


def test_plot_residuals_returns_figure():
    y_true = torch.randn(5, 1)
    y_pred = torch.randn(5, 1)
    fig = plot_residuals(y_true, y_pred)
    assert fig is not None
    matplotlib.pyplot.close(fig)


def test_plot_cate_calibration_returns_figure():
    tau_hat = torch.randn(20, 1)
    tau_true = torch.randn(20, 1)
    fig = plot_cate_calibration(tau_hat, tau_true)
    assert fig is not None
    matplotlib.pyplot.close(fig)


def test_plot_partial_dependence_returns_figure():
    model = ACX(p=2)
    X = torch.randn(10, 2)
    fig = plot_partial_dependence(model, X, feature=0)
    assert fig is not None
    matplotlib.pyplot.close(fig)


def test_plot_ice_returns_figure():
    model = ACX(p=2)
    X = torch.randn(5, 2)
    fig = plot_ice(model, X, feature=1)
    assert fig is not None
    matplotlib.pyplot.close(fig)


def test_plot_grad_norms_returns_figure():
    hist = [
        EpochStats(
            epoch=0,
            loss_d=0.0,
            loss_g=0.0,
            loss_y=0.0,
            loss_cons=0.0,
            loss_adv=0.0,
            grad_norm_g=1.0,
            grad_norm_d=2.0,
        )
    ]
    fig = plot_grad_norms(hist)
    assert fig is not None
    matplotlib.pyplot.close(fig)


def test_plot_grad_norms_with_weights_returns_figure():
    hist = [
        EpochStats(
            epoch=0,
            loss_d=0.0,
            loss_g=0.0,
            loss_y=0.0,
            loss_cons=0.0,
            loss_adv=0.0,
            grad_norm_g=1.0,
            grad_norm_d=2.0,
            w_y=0.5,
            w_cons=1.0,
            w_adv=1.5,
        )
    ]
    fig = plot_grad_norms(hist)
    assert fig is not None
    matplotlib.pyplot.close(fig)


def test_plot_learning_rates_returns_figure():
    hist = [
        EpochStats(
            epoch=0,
            loss_d=0.0,
            loss_g=0.0,
            loss_y=0.0,
            loss_cons=0.0,
            loss_adv=0.0,
            lr_g=1e-3,
            lr_d=1e-3,
        )
    ]
    fig = plot_learning_rates(hist)
    assert fig is not None
    matplotlib.pyplot.close(fig)
