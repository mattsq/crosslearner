from crosslearner.datasets.toy import get_toy_dataloader
from crosslearner.experiments.sweep import run_sweep
import crosslearner.benchmarks.run_benchmarks as rb_module


def test_run_sweep(monkeypatch, tmp_path):
    loader, data = get_toy_dataloader(batch_size=4, n=16, p=3)
    rb_module.DATASET_LOADERS["toy"] = lambda seed=0: (loader, data)
    space = tmp_path / "space.yaml"
    space.write_text(
        "rep_dim:\n  type: int\n  low: 4\n  high: 8\nepochs:\n  type: int\n  low: 1\n  high: 1\n"
    )
    study = run_sweep("toy", n_trials=1, sampler="random", space_config=str(space))
    assert len(study.trials) == 1
