from crosslearner.datasets.complex import get_complex_dataloader


def test_get_complex_dataloader_shapes():
    loader, (mu0, mu1) = get_complex_dataloader(batch_size=4, n=8, p=6, seed=0)
    assert len(loader.dataset) == 8
    X, T, Y = next(iter(loader))
    assert X.shape == (4, 6)
    assert T.shape == (4, 1)
    assert Y.shape == (4, 1)
    assert mu0.shape == (8, 1)
    assert mu1.shape == (8, 1)
