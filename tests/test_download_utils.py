import os
import urllib.request
import pytest

from crosslearner.datasets.utils import download_if_missing
from crosslearner.benchmarks import run_benchmarks


def test_download_if_missing_existing_file(tmp_path, monkeypatch):
    path = tmp_path / "data.txt"
    path.write_text("hi")
    called = {"count": 0}

    def fake_urlretrieve(url, p):
        called["count"] += 1

    monkeypatch.setattr(urllib.request, "urlretrieve", fake_urlretrieve)
    returned = download_if_missing("http://example.com", str(path))
    assert returned == str(path)
    assert called["count"] == 0


def test_download_if_missing_creates_file(tmp_path, monkeypatch):
    path = tmp_path / "data.txt"

    def fake_urlretrieve(url, p):
        with open(p, "w") as f:
            f.write("data")

    monkeypatch.setattr(urllib.request, "urlretrieve", fake_urlretrieve)
    returned = download_if_missing("http://example.com", str(path))
    assert path.read_text() == "data"
    assert returned == str(path)


def test_download_if_missing_failure(tmp_path, monkeypatch):
    path = tmp_path / "data.txt"

    def fake_urlretrieve(url, p):
        raise OSError("bad")

    monkeypatch.setattr(urllib.request, "urlretrieve", fake_urlretrieve)
    with pytest.raises(RuntimeError):
        download_if_missing("http://example.com", str(path))


def test_load_external_iris(monkeypatch):
    iris_path = os.path.join(os.path.dirname(run_benchmarks.__file__), "iris.csv")
    if os.path.exists(iris_path):
        os.remove(iris_path)

    def fake_urlretrieve(url, p):
        with open(p, "w") as f:
            f.write(
                "sepal_length,sepal_width,petal_length,petal_width,species\n"
                "1,2,3,4,0\n"
                "2,3,4,5,1\n"
            )

    monkeypatch.setattr(urllib.request, "urlretrieve", fake_urlretrieve)
    loader, (mu0, mu1) = run_benchmarks.load_external_iris(batch_size=1, seed=0)
    try:
        assert len(loader.dataset) == 2
        X, T, Y = next(iter(loader))
        assert X.shape == (1, 4)
        assert T.shape == (1, 1)
        assert Y.shape == (1, 1)
        assert mu0.shape == (2, 1)
        assert mu1.shape == (2, 1)
    finally:
        if os.path.exists(iris_path):
            os.remove(iris_path)
