import torch
from torch.utils.data import DataLoader, TensorDataset

from causal_consistency_nn.metrics import (
    average_treatment_effect,
    gaussian_log_likelihood,
)
from causal_consistency_nn.eval import evaluate
from causal_consistency_nn.serve import predict_z, counterfactual_z, impute_y


class DummyModel(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        z = x[:, :1]
        mu0 = z
        mu1 = mu0 + 1.0
        return z, mu0, mu1


def test_metrics_and_eval():
    torch.manual_seed(0)
    x = torch.randn(20, 2)
    t = torch.bernoulli(torch.full((20, 1), 0.5))
    mu0 = x[:, :1]
    mu1 = mu0 + 1.0
    y = torch.where(t.bool(), mu1, mu0)
    ds = TensorDataset(x, t, y)
    loader = DataLoader(ds, batch_size=5)

    ate_true = 1.0
    assert abs(average_treatment_effect(mu0, mu1) - ate_true) < 1e-6
    ll = gaussian_log_likelihood(y, y, torch.ones_like(y))
    expected_ll = float(-0.5 * torch.log(torch.tensor(2 * torch.pi)))
    assert abs(ll - expected_ll) < 1e-4

    model = DummyModel()
    metrics = evaluate(model, loader)
    assert abs(metrics["ate"] - ate_true) < 1e-6
    assert metrics["log_likelihood"] <= 0.0


def test_serving_helpers():
    model = DummyModel()
    x = torch.randn(4, 2)
    t = torch.tensor([[0], [1], [0], [1]], dtype=torch.float32)

    z = predict_z(model, x)
    assert z.shape == (4, 1)

    z2, ycf = counterfactual_z(model, x, t)
    assert z2.shape == (4, 1)
    assert ycf.shape == (4, 1)

    y_imp = impute_y(model, x, t)
    assert y_imp.shape == (4, 1)
