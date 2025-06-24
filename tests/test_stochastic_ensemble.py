import torch
from crosslearner.models.stochastic import (
    DropConnectLinear,
    BaseNet,
    StochasticEnsemble,
)


def test_dropconnect_linear_applies_mask():
    layer = DropConnectLinear(2, 1, bias=False, dropconnect_prob=0.5)
    layer.weight.data.fill_(1.0)
    x = torch.ones(1, 2)
    layer.train()
    samples = torch.stack([layer(x) for _ in range(5)])
    assert samples.std() > 0


def test_stochastic_ensemble_shapes():
    ensemble = StochasticEnsemble(
        BaseNet,
        ensemble_size=3,
        input_dim=2,
        hidden_dim=4,
        drop_prob=0.2,
        dropconnect_prob=0.1,
    )
    x = torch.randn(5, 2)
    ensemble.eval()
    y = ensemble(x)
    assert y.shape == (5, 1)


def test_mc_dropout_predict_shape():
    ensemble = StochasticEnsemble(
        BaseNet,
        ensemble_size=2,
        input_dim=3,
        hidden_dim=4,
        drop_prob=0.5,
        dropconnect_prob=0.5,
    )
    x = torch.randn(6, 3)
    preds = ensemble.mc_dropout_predict(x, mc_passes=4)
    assert preds.shape == (2 * 4, 6, 1)
    assert torch.any(preds.std(dim=0) > 0)
