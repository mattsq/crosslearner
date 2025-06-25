import torch
from torch.utils.data import DataLoader, TensorDataset, SequentialSampler
import torch.nn as nn
import pytest

from crosslearner.utils import MutableBatchSampler, GNSBatchScheduler


def _simple_loss(model, batch):
    x, y = batch
    pred = model(x)
    return nn.functional.mse_loss(pred, y)


def test_mutable_batch_sampler_allows_resize():
    data = TensorDataset(torch.arange(10))
    sampler = MutableBatchSampler(
        SequentialSampler(data), batch_size=2, drop_last=False
    )
    loader = DataLoader(data, batch_sampler=sampler)

    batch = next(iter(loader))[0]
    assert batch.shape[0] == 2

    loader.batch_sampler.batch_size = 3
    batch = next(iter(loader))[0]
    assert batch.shape[0] == 3


def test_gns_batch_scheduler_grows(monkeypatch):
    x = torch.randn(8, 1)
    y = torch.randn(8, 1)
    data = TensorDataset(x, y)
    sampler = MutableBatchSampler(
        SequentialSampler(data), batch_size=2, drop_last=False
    )
    loader = DataLoader(data, batch_sampler=sampler)
    model = nn.Linear(1, 1)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    sched = GNSBatchScheduler(
        model,
        _simple_loss,
        loader,
        opt,
        target_gns=1.0,
        band=1.0,
        check_every=1,
    )
    monkeypatch.setattr(sched, "_grad_noise_scale", lambda a, b: 0.5)

    sched.after_train_step()

    assert loader.batch_sampler.batch_size == 4
    assert opt.param_groups[0]["lr"] == pytest.approx(0.2)


def test_scheduler_grows_on_plateau(monkeypatch):
    x = torch.randn(8, 1)
    y = torch.randn(8, 1)
    data = TensorDataset(x, y)
    sampler = MutableBatchSampler(
        SequentialSampler(data), batch_size=2, drop_last=False
    )
    loader = DataLoader(data, batch_sampler=sampler)
    model = nn.Linear(1, 1)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    sched = GNSBatchScheduler(
        model,
        _simple_loss,
        loader,
        opt,
        check_every=100,
        plateau_patience=2,
    )
    monkeypatch.setattr(sched, "_maybe_grow", lambda: None)

    for _ in range(3):
        sched.after_train_step(val_loss=1.0)

    assert loader.batch_sampler.batch_size == 4
    assert opt.param_groups[0]["lr"] == pytest.approx(0.2)
