import numpy as np
from crosslearner.datasets.toy import get_toy_dataloader
from crosslearner.datasets.complex import get_complex_dataloader
from crosslearner.datasets.jobs import get_jobs_dataloader
from crosslearner.datasets import ihdp


def test_get_toy_dataloader_shapes():
    loader, (mu0, mu1) = get_toy_dataloader(batch_size=4, n=8, p=3)
    assert len(loader.dataset) == 8
    X, T, Y = next(iter(loader))
    assert X.shape == (4, 3)
    assert T.shape == (4, 1)
    assert Y.shape == (4, 1)
    assert mu0.shape == (8, 1)
    assert mu1.shape == (8, 1)


def test_get_complex_dataloader_shapes():
    loader, (mu0, mu1) = get_complex_dataloader(batch_size=4, n=8, p=6, seed=0)
    assert len(loader.dataset) == 8
    X, T, Y = next(iter(loader))
    assert X.shape == (4, 6)
    assert T.shape == (4, 1)
    assert Y.shape == (4, 1)
    assert mu0.shape == (8, 1)
    assert mu1.shape == (8, 1)


def test_get_jobs_dataloader_shapes():
    loader, (mu0, mu1) = get_jobs_dataloader(batch_size=4)
    X, T, Y = next(iter(loader))
    assert X.shape[0] == 4
    assert T.shape == (4, 1)
    assert Y.shape == (4, 1)
    assert mu0 is None and mu1 is None


def test_get_ihdp_dataloader_shapes(monkeypatch, tmp_path):
    def fake_download(url: str, path: str) -> str:
        n = 2 if "train" in path else 3
        data = dict(
            x=np.zeros((n, 3, 1)),
            t=np.zeros((n, 1)),
            yf=np.zeros((n, 1)),
            mu0=np.zeros((n, 1)),
            mu1=np.ones((n, 1)),
        )
        np.savez(path, **data)
        return path

    monkeypatch.setattr(ihdp, "_download", fake_download)
    loader, (mu0, mu1) = ihdp.get_ihdp_dataloader(
        seed=0, batch_size=2, data_dir=tmp_path
    )
    X, T, Y = next(iter(loader))
    assert X.shape == (2, 3)
    assert T.shape == (2, 1)
    assert Y.shape == (2, 1)
    assert mu0.shape == (5, 1)
    assert mu1.shape == (5, 1)
