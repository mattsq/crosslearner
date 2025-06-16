import numpy as np
import torch

from crosslearner.evaluation.propensity import estimate_propensity


def test_estimate_propensity_returns_probs():
    rng = np.random.default_rng(0)
    X = torch.tensor(rng.normal(size=(50, 3)), dtype=torch.float32)
    T = torch.tensor(rng.integers(0, 2, size=(50, 1)), dtype=torch.float32)
    prop = estimate_propensity(X, T, folds=3, seed=0)
    assert prop.shape == (50, 1)
    assert torch.all(prop >= 0) and torch.all(prop <= 1)
    assert abs(prop.mean().item() - T.float().mean().item()) < 0.1
