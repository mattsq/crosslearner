import importlib
from crosslearner.datasets.toy import get_toy_dataloader
from crosslearner.models.acx import ACX

rb_module = importlib.import_module("crosslearner.benchmarks.run_benchmark")


def test_run_benchmark_with_baselines(monkeypatch):
    loader, data = get_toy_dataloader(batch_size=4, n=8, p=3)

    def fake_loader(seed=None):
        return loader, data

    rb_module.DATASET_LOADERS["toy"] = lambda seed=None: (loader, data)
    monkeypatch.setattr(rb_module, "train_acx", lambda *a, **k: ACX(p=3))
    results = rb_module.run("toy", replicates=1, epochs=1)
    assert len(results) == 1
    metrics = results[0]
    assert "acx" in metrics and "slearner" in metrics
