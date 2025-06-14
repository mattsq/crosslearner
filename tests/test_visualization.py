import torch
import matplotlib

matplotlib.use("Agg")

from crosslearner.visualization import plot_losses, scatter_tau
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


def test_scatter_tau_returns_figure():
    model = ACX(p=3)
    X = torch.randn(4, 3)
    mu0 = torch.zeros(4, 1)
    mu1 = torch.ones(4, 1)
    fig = scatter_tau(model, X, mu0, mu1)
    assert fig is not None
    matplotlib.pyplot.close(fig)
