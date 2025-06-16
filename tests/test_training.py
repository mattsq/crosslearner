import torch
from crosslearner import __main__ as cli
from crosslearner.benchmarks import run_benchmarks
from crosslearner.datasets.toy import get_toy_dataloader
from crosslearner.evaluation.evaluate import evaluate
from crosslearner.models.acx import ACX
from crosslearner.training.train_acx import train_acx
import torch.nn as nn
import pytest
from torch.utils.data import DataLoader, TensorDataset


def test_train_acx_short():
    torch.manual_seed(0)
    loader, (mu0, mu1) = get_toy_dataloader(batch_size=16, n=64, p=4)
    model = train_acx(loader, p=4, device="cpu", epochs=2)
    X = torch.cat([b[0] for b in loader])
    mu0_all = mu0
    mu1_all = mu1
    metric = evaluate(model, X, mu0_all, mu1_all)
    assert isinstance(metric, float)
    assert metric >= 0.0


def test_tensorboard_logging(tmp_path):
    loader, _ = get_toy_dataloader(batch_size=16, n=32, p=4)
    logdir = tmp_path / "tb"
    train_acx(loader, p=4, device="cpu", epochs=1, tensorboard_logdir=str(logdir))
    assert any(logdir.iterdir())


def test_weight_clipping():
    torch.manual_seed(0)
    loader, _ = get_toy_dataloader(batch_size=16, n=64, p=4)
    model = train_acx(loader, p=4, device="cpu", epochs=1, weight_clip=0.01)
    for p in model.disc.parameters():
        assert p.data.abs().max() <= 0.011


def test_early_stopping():
    torch.manual_seed(0)
    loader, (mu0, mu1) = get_toy_dataloader(batch_size=16, n=64, p=4)
    X = torch.cat([b[0] for b in loader])
    val_data = (X, mu0, mu1)
    _, history = train_acx(
        loader,
        p=4,
        device="cpu",
        epochs=5,
        val_data=val_data,
        patience=1,
        return_history=True,
    )
    assert len(history) <= 5


def test_cli_main(monkeypatch, capsys):
    loader, data = get_toy_dataloader(batch_size=4, n=8, p=3)
    monkeypatch.setattr(cli, "get_toy_dataloader", lambda: (loader, data))
    monkeypatch.setattr(cli, "train_acx", lambda *a, **k: ACX(p=3))
    monkeypatch.setattr(cli, "evaluate", lambda *a, **k: 0.0)
    cli.main()
    out = capsys.readouterr().out
    assert "sqrt(PEHE" in out


def test_run_benchmarks_all(monkeypatch):
    loader, data = get_toy_dataloader(batch_size=4, n=8, p=3)

    def fake_loader(*args, **kwargs):
        return loader, data

    monkeypatch.setattr(run_benchmarks, "get_toy_dataloader", fake_loader)
    monkeypatch.setattr(run_benchmarks, "get_complex_dataloader", fake_loader)
    monkeypatch.setattr(run_benchmarks, "load_external_iris", fake_loader)
    monkeypatch.setattr(run_benchmarks, "get_ihdp_dataloader", fake_loader)
    monkeypatch.setattr(run_benchmarks, "get_jobs_dataloader", fake_loader)
    monkeypatch.setattr(run_benchmarks, "get_acic2016_dataloader", fake_loader)
    monkeypatch.setattr(run_benchmarks, "get_acic2018_dataloader", fake_loader)
    monkeypatch.setattr(run_benchmarks, "get_twins_dataloader", fake_loader)
    monkeypatch.setattr(run_benchmarks, "get_lalonde_dataloader", fake_loader)
    monkeypatch.setattr(run_benchmarks, "get_confounding_dataloader", fake_loader)
    monkeypatch.setattr(run_benchmarks, "train_acx", lambda *a, **k: ACX(p=3))
    monkeypatch.setattr(run_benchmarks, "evaluate", lambda *a, **k: 0.0)
    results = run_benchmarks.run("all", replicates=1, epochs=1)
    assert len(results) == 5
    assert all(isinstance(r, float) for r in results)


def test_train_acx_options():
    torch.manual_seed(0)
    loader, (mu0, mu1) = get_toy_dataloader(batch_size=4, n=16, p=4)
    X = torch.cat([b[0] for b in loader])
    val_data = (X, mu0, mu1)
    train_acx(
        loader,
        p=4,
        device="cpu",
        epochs=2,
        warm_start=1,
        use_wgan_gp=True,
        spectral_norm=True,
        feature_matching=True,
        label_smoothing=True,
        instance_noise=True,
        gradient_reversal=True,
        ttur=True,
        lambda_gp=0.1,
        eta_fm=1.0,
        grl_weight=0.5,
        weight_clip=0.1,
        val_data=val_data,
        patience=1,
        verbose=False,
    )


def test_train_acx_custom_architecture():
    loader, _ = get_toy_dataloader(batch_size=8, n=32, p=3)
    model = train_acx(
        loader,
        p=3,
        device="cpu",
        epochs=1,
        rep_dim=16,
        phi_layers=[8],
        head_layers=[4],
        disc_layers=[4],
        activation="elu",
        verbose=False,
    )
    assert isinstance(model, ACX)


def test_train_acx_custom_optimizer():
    loader, _ = get_toy_dataloader(batch_size=8, n=32, p=3)
    model = train_acx(
        loader,
        p=3,
        device="cpu",
        epochs=1,
        optimizer="sgd",
        opt_g_kwargs={"momentum": 0.0},
        opt_d_kwargs={"momentum": 0.0},
        verbose=False,
    )
    assert isinstance(model, ACX)


def test_train_acx_custom_scheduler(monkeypatch):
    loader, _ = get_toy_dataloader(batch_size=8, n=32, p=3)

    steps = {"count": 0}

    class DummyScheduler:
        def __init__(self, *args, **kwargs):
            pass

        def step(self, *args, **kwargs):
            steps["count"] += 1

    monkeypatch.setattr(torch.optim.lr_scheduler, "StepLR", DummyScheduler)

    train_acx(
        loader,
        p=3,
        device="cpu",
        epochs=1,
        lr_scheduler="step",
        verbose=False,
    )

    assert steps["count"] == 2


def test_warm_start_logs_losses():
    loader, _ = get_toy_dataloader(batch_size=8, n=32, p=4)
    _, history = train_acx(
        loader,
        p=4,
        device="cpu",
        epochs=2,
        warm_start=1,
        return_history=True,
        verbose=False,
    )
    assert history[0].loss_y > 0
    assert history[0].loss_g > 0


def test_train_acx_1d_targets():
    X = torch.randn(16, 4)
    T = torch.randint(0, 2, (16,))
    mu0 = torch.randn(16)
    mu1 = mu0 + torch.randn(16)
    Y = torch.where(T.bool(), mu1, mu0) + 0.1 * torch.randn(16)
    loader = DataLoader(TensorDataset(X, T, Y), batch_size=8)
    model = train_acx(loader, p=4, device="cpu", epochs=1, verbose=False)
    assert isinstance(model, ACX)


def test_instance_noise_keeps_targets(monkeypatch):
    loader, _ = get_toy_dataloader(batch_size=4, n=8, p=4)
    loader = DataLoader(loader.dataset, batch_size=4, shuffle=False)

    targets: list[torch.Tensor] = []

    class RecMSE(torch.nn.Module):
        def forward(self, inp, tgt):
            targets.append(tgt.clone())
            return torch.nn.functional.mse_loss(inp, tgt)

    monkeypatch.setattr(torch.nn, "MSELoss", lambda: RecMSE())
    monkeypatch.setattr(torch, "randn_like", lambda t: torch.ones_like(t))

    train_acx(loader, p=4, device="cpu", epochs=1, instance_noise=True, verbose=False)

    assert torch.allclose(targets[0], loader.dataset.tensors[2][:4])


def test_train_acx_feature_mismatch():
    loader, _ = get_toy_dataloader(batch_size=4, n=8, p=4)
    with pytest.raises(ValueError):
        train_acx(loader, p=3, device="cpu", epochs=1, verbose=False)


def test_train_acx_invalid_activation():
    loader, _ = get_toy_dataloader(batch_size=4, n=8, p=4)
    with pytest.raises(ValueError):
        train_acx(loader, p=4, device="cpu", epochs=1, activation="bad", verbose=False)


def test_train_acx_invalid_optimizer():
    loader, _ = get_toy_dataloader(batch_size=4, n=8, p=4)
    with pytest.raises(ValueError):
        train_acx(loader, p=4, device="cpu", epochs=1, optimizer="bad", verbose=False)


def test_train_acx_invalid_scheduler():
    loader, _ = get_toy_dataloader(batch_size=4, n=8, p=4)
    with pytest.raises(ValueError):
        train_acx(
            loader, p=4, device="cpu", epochs=1, lr_scheduler="bad", verbose=False
        )


def test_train_acx_negative_grad_clip():
    loader, _ = get_toy_dataloader(batch_size=4, n=8, p=4)
    with pytest.raises(ValueError):
        train_acx(loader, p=4, device="cpu", epochs=1, grad_clip=-1, verbose=False)


def test_train_acx_negative_weight_clip():
    loader, _ = get_toy_dataloader(batch_size=4, n=8, p=4)
    with pytest.raises(ValueError):
        train_acx(loader, p=4, device="cpu", epochs=1, weight_clip=-0.1, verbose=False)


def test_train_acx_dropout_options():
    loader, _ = get_toy_dataloader(batch_size=4, n=8, p=4)
    model = train_acx(
        loader,
        p=4,
        device="cpu",
        epochs=1,
        phi_dropout=0.1,
        head_dropout=0.1,
        disc_dropout=0.1,
        verbose=False,
    )
    assert any(isinstance(m, nn.Dropout) for m in model.phi.net.modules())
    assert any(isinstance(m, nn.Dropout) for m in model.mu0.net.modules())
    assert any(isinstance(m, nn.Dropout) for m in model.disc.net.modules())
