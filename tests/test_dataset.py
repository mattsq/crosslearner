import torch
from crosslearner.datasets.toy import get_toy_dataloader


def test_get_toy_dataloader_shapes():
    loader, (mu0, mu1) = get_toy_dataloader(batch_size=4, n=8, p=3)
    assert len(loader.dataset) == 8
    X, T, Y = next(iter(loader))
    assert X.shape == (4, 3)
    assert T.shape == (4, 1)
    assert Y.shape == (4, 1)
    assert mu0.shape == (8, 1)
    assert mu1.shape == (8, 1)


def test_get_toy_dataloader_seed_reproducible():
    _, (mu0_a, mu1_a) = get_toy_dataloader(batch_size=4, n=8, p=3, seed=0)
    _, (mu0_b, mu1_b) = get_toy_dataloader(batch_size=4, n=8, p=3, seed=0)
    assert torch.allclose(mu0_a, mu0_b)
    assert torch.allclose(mu1_a, mu1_b)
