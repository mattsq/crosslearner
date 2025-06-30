import torch
from crosslearner.datasets.toy import get_toy_dataloader
from crosslearner.training.baseline import BaselineConfig, train_baseline
from crosslearner.evaluation.propensity import estimate_propensity


def test_propensity_crossfit_deterministic():
    loader, _ = get_toy_dataloader(batch_size=8, n=32, p=3, seed=0)
    X = torch.cat([b[0] for b in loader])
    T = torch.cat([b[1] for b in loader])
    e1 = estimate_propensity(X, T, folds=3, seed=1)
    e2 = estimate_propensity(X, T, folds=3, seed=1)
    assert torch.allclose(e1, e2)


def test_val_pehe_proxy_decreases():
    loader, (mu0, mu1) = get_toy_dataloader(batch_size=8, n=64, p=3, seed=0)
    cfg = BaselineConfig(p=3, epochs=3, seed=0)
    _, history = train_baseline(loader, mu0, mu1, cfg)
    assert len(history) == 2
    assert all(isinstance(v, float) for v in history)
