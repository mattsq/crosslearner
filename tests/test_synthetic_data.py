import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import torch
from causal_consistency_nn.config import SyntheticDataConfig
from causal_consistency_nn.data.synthetic import (
    generate_synthetic,
    get_synthetic_dataloader,
)


def test_generate_synthetic_shapes():
    cfg = SyntheticDataConfig(n_samples=8, p=3, noise=0.1, seed=0)
    dataset, (mu0, mu1) = generate_synthetic(cfg)
    X, T, Y = dataset.tensors
    assert X.shape == (8, 3)
    assert T.shape == (8, 1)
    assert Y.shape == (8, 1)
    assert mu0.shape == (8, 1)
    assert mu1.shape == (8, 1)


def test_generate_synthetic_reproducible():
    cfg = SyntheticDataConfig(n_samples=8, p=3, noise=0.1, seed=0)
    ds_a, (mu0_a, mu1_a) = generate_synthetic(cfg)
    ds_b, (mu0_b, mu1_b) = generate_synthetic(cfg)
    assert torch.allclose(mu0_a, mu0_b)
    assert torch.allclose(mu1_a, mu1_b)
    assert torch.allclose(ds_a.tensors[0], ds_b.tensors[0])


def test_get_synthetic_dataloader_batching():
    cfg = SyntheticDataConfig(n_samples=8, p=3, noise=0.1, seed=0)
    loader, (mu0, mu1) = get_synthetic_dataloader(cfg, batch_size=4, shuffle=False)
    batch = next(iter(loader))
    assert batch[0].shape == (4, 3)
    assert len(loader.dataset) == 8
    assert mu0.shape == (8, 1)
    assert mu1.shape == (8, 1)


def test_missing_y_entries():
    cfg = SyntheticDataConfig(n_samples=10, p=3, noise=0.1, missing_y_prob=0.5, seed=0)
    dataset, _ = generate_synthetic(cfg)
    Y = dataset.tensors[2]
    assert torch.isnan(Y).sum() > 0
