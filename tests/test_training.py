import torch
from crosslearner.utils import set_seed
from crosslearner import __main__ as cli
from crosslearner.benchmarks import run_benchmarks
from crosslearner.datasets.toy import get_toy_dataloader
from crosslearner.evaluation.evaluate import evaluate
from crosslearner.models.acx import ACX
from crosslearner.training.train_acx import train_acx
from crosslearner.training import ModelConfig, TrainingConfig
import torch.nn as nn
import pytest
from torch.utils.data import DataLoader, TensorDataset


def test_train_acx_short():
    set_seed(0)
    loader, (mu0, mu1) = get_toy_dataloader(batch_size=16, n=64, p=4)
    model_cfg = ModelConfig(p=4)
    train_cfg = TrainingConfig(epochs=2)
    model = train_acx(loader, model_cfg, train_cfg, device="cpu")
    X = torch.cat([b[0] for b in loader])
    mu0_all = mu0
    mu1_all = mu1
    metric = evaluate(model, X, mu0_all, mu1_all)
    assert isinstance(metric, float)
    assert metric >= 0.0


def test_tensorboard_logging(tmp_path):
    loader, _ = get_toy_dataloader(batch_size=16, n=32, p=4)
    logdir = tmp_path / "tb"
    model_cfg = ModelConfig(p=4)
    train_cfg = TrainingConfig(epochs=1, tensorboard_logdir=str(logdir))
    train_acx(loader, model_cfg, train_cfg, device="cpu")
    assert any(logdir.iterdir())


def test_weight_clipping():
    set_seed(0)
    loader, _ = get_toy_dataloader(batch_size=16, n=64, p=4)
    model_cfg = ModelConfig(p=4)
    train_cfg = TrainingConfig(epochs=1, weight_clip=0.01)
    model = train_acx(loader, model_cfg, train_cfg, device="cpu")
    for p in model.disc.parameters():
        assert p.data.abs().max() <= 0.011


def test_early_stopping():
    set_seed(0)
    loader, (mu0, mu1) = get_toy_dataloader(batch_size=16, n=64, p=4)
    X = torch.cat([b[0] for b in loader])
    val_data = (X, mu0, mu1)
    model_cfg = ModelConfig(p=4)
    train_cfg = TrainingConfig(
        epochs=5,
        val_data=val_data,
        patience=1,
        return_history=True,
    )
    _, history = train_acx(loader, model_cfg, train_cfg, device="cpu")
    assert len(history) <= 5


def test_risk_early_stopping():
    set_seed(0)
    loader, _ = get_toy_dataloader(batch_size=16, n=64, p=4)
    X = torch.cat([b[0] for b in loader])
    T_all = torch.cat([b[1] for b in loader])
    Y_all = torch.cat([b[2] for b in loader])
    model_cfg = ModelConfig(p=4)
    train_cfg = TrainingConfig(
        epochs=5,
        risk_data=(X, T_all, Y_all),
        risk_folds=2,
        patience=1,
        return_history=True,
        verbose=False,
    )
    _, history = train_acx(loader, model_cfg, train_cfg, device="cpu")
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
    set_seed(0)
    loader, (mu0, mu1) = get_toy_dataloader(batch_size=4, n=16, p=4)
    X = torch.cat([b[0] for b in loader])
    val_data = (X, mu0, mu1)
    model_cfg = ModelConfig(p=4)
    train_cfg = TrainingConfig(
        epochs=2,
        warm_start=1,
        adv_loss="wgan-gp",
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
        ema_decay=0.5,
        verbose=False,
    )
    train_acx(loader, model_cfg, train_cfg, device="cpu")


def test_train_acx_custom_architecture():
    loader, _ = get_toy_dataloader(batch_size=8, n=32, p=3)
    model_cfg = ModelConfig(
        p=3,
        rep_dim=16,
        phi_layers=[8],
        head_layers=[4],
        disc_layers=[4],
        activation="elu",
        disc_pack=2,
    )
    train_cfg = TrainingConfig(epochs=1, verbose=False)
    model = train_acx(loader, model_cfg, train_cfg, device="cpu")
    assert isinstance(model, ACX)


def test_train_acx_custom_optimizer():
    loader, _ = get_toy_dataloader(batch_size=8, n=32, p=3)
    model_cfg = ModelConfig(p=3)
    train_cfg = TrainingConfig(
        epochs=1,
        optimizer="sgd",
        opt_g_kwargs={"momentum": 0.0},
        opt_d_kwargs={"momentum": 0.0},
        verbose=False,
    )
    model = train_acx(loader, model_cfg, train_cfg, device="cpu")
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

    model_cfg = ModelConfig(p=3)
    train_cfg = TrainingConfig(epochs=1, lr_scheduler="step", verbose=False)
    train_acx(loader, model_cfg, train_cfg, device="cpu")

    assert steps["count"] == 2


def test_warm_start_logs_losses():
    loader, _ = get_toy_dataloader(batch_size=8, n=32, p=4)
    model_cfg = ModelConfig(p=4)
    train_cfg = TrainingConfig(
        epochs=2,
        warm_start=1,
        return_history=True,
        verbose=False,
    )
    _, history = train_acx(loader, model_cfg, train_cfg, device="cpu")
    assert history[0].loss_y > 0
    assert history[0].loss_g > 0


def test_warm_start_grad_clip():
    loader, _ = get_toy_dataloader(batch_size=8, n=32, p=4)
    model_cfg = ModelConfig(p=4)
    train_cfg = TrainingConfig(
        epochs=2,
        warm_start=1,
        grad_clip=1.0,
        verbose=False,
    )
    model = train_acx(loader, model_cfg, train_cfg, device="cpu")
    assert isinstance(model, ACX)


def test_train_acx_1d_targets():
    X = torch.randn(16, 4)
    T = torch.randint(0, 2, (16,))
    mu0 = torch.randn(16)
    mu1 = mu0 + torch.randn(16)
    Y = torch.where(T.bool(), mu1, mu0) + 0.1 * torch.randn(16)
    loader = DataLoader(TensorDataset(X, T, Y), batch_size=8)
    model_cfg = ModelConfig(p=4)
    train_cfg = TrainingConfig(epochs=1, verbose=False)
    model = train_acx(loader, model_cfg, train_cfg, device="cpu")
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

    model_cfg = ModelConfig(p=4)
    train_cfg = TrainingConfig(epochs=1, instance_noise=True, verbose=False)
    train_acx(loader, model_cfg, train_cfg, device="cpu")

    assert torch.allclose(targets[0], loader.dataset.tensors[2][:4])


def test_train_acx_feature_mismatch():
    loader, _ = get_toy_dataloader(batch_size=4, n=8, p=4)
    with pytest.raises(ValueError):
        model_cfg = ModelConfig(p=3)
        train_cfg = TrainingConfig(epochs=1, verbose=False)
        train_acx(loader, model_cfg, train_cfg, device="cpu")


def test_train_acx_invalid_activation():
    loader, _ = get_toy_dataloader(batch_size=4, n=8, p=4)
    with pytest.raises(ValueError):
        model_cfg = ModelConfig(p=4, activation="bad")
        train_cfg = TrainingConfig(epochs=1, verbose=False)
        train_acx(loader, model_cfg, train_cfg, device="cpu")


def test_train_acx_activation_instance():
    loader, _ = get_toy_dataloader(batch_size=4, n=8, p=4)
    with pytest.raises(TypeError):
        model_cfg = ModelConfig(p=4, activation=nn.ReLU())
        train_cfg = TrainingConfig(epochs=1, verbose=False)
        train_acx(loader, model_cfg, train_cfg, device="cpu")


def test_train_acx_invalid_optimizer():
    loader, _ = get_toy_dataloader(batch_size=4, n=8, p=4)
    with pytest.raises(ValueError):
        model_cfg = ModelConfig(p=4)
        train_cfg = TrainingConfig(epochs=1, optimizer="bad", verbose=False)
        train_acx(loader, model_cfg, train_cfg, device="cpu")


def test_train_acx_invalid_scheduler():
    loader, _ = get_toy_dataloader(batch_size=4, n=8, p=4)
    with pytest.raises(ValueError):
        model_cfg = ModelConfig(p=4)
        train_cfg = TrainingConfig(epochs=1, lr_scheduler="bad", verbose=False)
        train_acx(loader, model_cfg, train_cfg, device="cpu")


def test_train_acx_negative_grad_clip():
    loader, _ = get_toy_dataloader(batch_size=4, n=8, p=4)
    with pytest.raises(ValueError):
        model_cfg = ModelConfig(p=4)
        train_cfg = TrainingConfig(epochs=1, grad_clip=-1, verbose=False)
        train_acx(loader, model_cfg, train_cfg, device="cpu")


def test_train_acx_negative_weight_clip():
    loader, _ = get_toy_dataloader(batch_size=4, n=8, p=4)
    with pytest.raises(ValueError):
        model_cfg = ModelConfig(p=4)
        train_cfg = TrainingConfig(epochs=1, weight_clip=-0.1, verbose=False)
        train_acx(loader, model_cfg, train_cfg, device="cpu")


def test_train_acx_dropout_options():
    loader, _ = get_toy_dataloader(batch_size=4, n=8, p=4)
    model_cfg = ModelConfig(p=4, phi_dropout=0.1, head_dropout=0.1, disc_dropout=0.1)
    train_cfg = TrainingConfig(epochs=1, verbose=False)
    model = train_acx(loader, model_cfg, train_cfg, device="cpu")
    assert any(isinstance(m, nn.Dropout) for m in model.phi.net.modules())
    assert any(isinstance(m, nn.Dropout) for m in model.mu0.net.modules())
    assert any(isinstance(m, nn.Dropout) for m in model.disc.net.modules())


def test_train_acx_batch_norm_option():
    loader, _ = get_toy_dataloader(batch_size=4, n=8, p=4)
    model_cfg = ModelConfig(p=4, batch_norm=True)
    train_cfg = TrainingConfig(epochs=1, verbose=False)
    model = train_acx(loader, model_cfg, train_cfg, device="cpu")
    assert any(isinstance(m, nn.BatchNorm1d) for m in model.phi.net.modules())


def test_alt_adv_losses():
    loader, _ = get_toy_dataloader(batch_size=4, n=8, p=4)
    model_cfg = ModelConfig(p=4)
    for loss in ("hinge", "lsgan"):
        train_cfg = TrainingConfig(epochs=1, adv_loss=loss, verbose=False)
        model = train_acx(loader, model_cfg, train_cfg, device="cpu")
        assert isinstance(model, ACX)


def test_train_acx_ema():
    loader, _ = get_toy_dataloader(batch_size=4, n=8, p=4)
    model_cfg = ModelConfig(p=4)
    train_cfg = TrainingConfig(epochs=1, ema_decay=0.5, verbose=False)
    model = train_acx(loader, model_cfg, train_cfg, device="cpu")
    assert isinstance(model, ACX)


def test_train_acx_r1_r2_unrolled():
    loader, _ = get_toy_dataloader(batch_size=4, n=8, p=4)
    model_cfg = ModelConfig(p=4)
    cfg = TrainingConfig(epochs=1, r1_gamma=0.1, verbose=False)
    model = train_acx(loader, model_cfg, cfg, device="cpu")
    assert isinstance(model, ACX)
    cfg = TrainingConfig(epochs=1, r2_gamma=0.1, verbose=False)
    model = train_acx(loader, model_cfg, cfg, device="cpu")
    assert isinstance(model, ACX)
    cfg = TrainingConfig(epochs=1, unrolled_steps=1, verbose=False)
    model = train_acx(loader, model_cfg, cfg, device="cpu")
    assert isinstance(model, ACX)
