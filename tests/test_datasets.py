import numpy as np
from crosslearner.datasets.toy import get_toy_dataloader
from crosslearner.datasets.complex import get_complex_dataloader
from crosslearner.datasets.jobs import get_jobs_dataloader
from crosslearner.datasets.aircraft import get_aircraft_dataloader
from crosslearner.datasets.tricks import get_tricky_dataloader
from crosslearner.datasets.random_dag import get_random_dag_dataloader
from crosslearner.datasets import ihdp, acic2016, acic2018, twins, lalonde, synthetic


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

    monkeypatch.setattr(ihdp, "download_if_missing", fake_download)
    loader, (mu0, mu1) = ihdp.get_ihdp_dataloader(
        seed=0, batch_size=2, data_dir=tmp_path
    )
    X, T, Y = next(iter(loader))
    assert X.shape == (2, 3)
    assert T.shape == (2, 1)
    assert Y.shape == (2, 1)
    assert mu0.shape == (5, 1)
    assert mu1.shape == (5, 1)


def _fake_npz(path: str, n: int = 4, p: int = 3, replicate: bool = True) -> None:
    if replicate:
        x = np.zeros((n, p, 1))
    else:
        x = np.zeros((n, p))
    t = np.zeros(n) if not replicate else np.zeros((n, 1))
    if replicate:
        y = np.zeros((n, 1))
        mu0 = np.zeros((n, 1))
        mu1 = np.ones((n, 1))
    else:
        y = np.zeros(n)
        mu0 = np.zeros(n)
        mu1 = np.ones(n)
    data = dict(x=x, t=t, yf=y, mu0=mu0, mu1=mu1)
    np.savez(path, **data)


def test_get_acic2016_dataloader(monkeypatch, tmp_path):
    monkeypatch.setattr(
        acic2016,
        "download_if_missing",
        lambda url, path: _fake_npz(path, replicate=True) or path,
    )
    loader, (mu0, mu1) = acic2016.get_acic2016_dataloader(
        batch_size=2, data_dir=tmp_path
    )
    X, T, Y = next(iter(loader))
    assert X.shape == (2, 3)
    assert T.shape == (2, 1)
    assert Y.shape == (2, 1)
    assert mu0.shape == (4, 1)
    assert mu1.shape == (4, 1)


def test_get_acic2018_dataloader(monkeypatch, tmp_path):
    monkeypatch.setattr(
        acic2018,
        "download_if_missing",
        lambda url, path: _fake_npz(path, replicate=True) or path,
    )
    loader, (mu0, mu1) = acic2018.get_acic2018_dataloader(
        batch_size=2, data_dir=tmp_path
    )
    X, T, Y = next(iter(loader))
    assert X.shape == (2, 3)
    assert T.shape == (2, 1)
    assert Y.shape == (2, 1)
    assert mu0.shape == (4, 1)
    assert mu1.shape == (4, 1)


def test_get_twins_dataloader(monkeypatch, tmp_path):
    monkeypatch.setattr(
        twins,
        "download_if_missing",
        lambda url, path: _fake_npz(path, replicate=False) or path,
    )
    loader, (mu0, mu1) = twins.get_twins_dataloader(batch_size=2, data_dir=tmp_path)
    X, T, Y = next(iter(loader))
    assert X.shape == (2, 3)
    assert T.shape == (2, 1)
    assert Y.shape == (2, 1)
    assert mu0.shape == (4, 1)
    assert mu1.shape == (4, 1)


def test_get_lalonde_dataloader_shapes():
    loader, (mu0, mu1) = lalonde.get_lalonde_dataloader(batch_size=4)
    X, T, Y = next(iter(loader))
    assert X.shape[0] == 4
    assert T.shape == (4, 1)
    assert Y.shape == (4, 1)
    assert mu0 is None and mu1 is None


def test_get_confounding_dataloader():
    loader, (mu0, mu1) = synthetic.get_confounding_dataloader(
        batch_size=2, n=4, p=3, confounding=0.5, seed=0
    )
    X, T, Y = next(iter(loader))
    assert X.shape == (2, 3)
    assert T.shape == (2, 1)
    assert Y.shape == (2, 1)
    assert mu0.shape == (4, 1)
    assert mu1.shape == (4, 1)


def test_get_aircraft_dataloader():
    loader, (mu0, mu1) = get_aircraft_dataloader(batch_size=2, n=4, seed=0)
    X, T, Y = next(iter(loader))
    assert X.shape == (2, 5)
    assert T.shape == (2, 1)
    assert Y.shape == (2, 1)
    assert mu0.shape == (4, 1)
    assert mu1.shape == (4, 1)


def test_get_tricky_dataloader():
    loader, (mu0, mu1) = get_tricky_dataloader(batch_size=2, n=4, p=4, seed=0)
    X, T, Y = next(iter(loader))
    assert X.shape == (2, 4)
    assert T.shape == (2, 1)
    assert Y.shape == (2, 1)
    assert mu0.shape == (4, 1)
    assert mu1.shape == (4, 1)


def test_get_random_dag_dataloader():
    loader, (mu0, mu1) = get_random_dag_dataloader(batch_size=2, n=4, p=3, seed=0)
    X, T, Y = next(iter(loader))
    assert X.shape == (2, 3)
    assert T.shape == (2, 1)
    assert Y.shape == (2, 1)
    assert mu0.shape == (4, 1)
    assert mu1.shape == (4, 1)
