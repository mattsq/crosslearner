import torch
from crosslearner.utils import set_seed
from crosslearner import __main__ as cli
from crosslearner.benchmarks import run_benchmarks
from crosslearner.datasets.toy import get_toy_dataloader
from crosslearner.evaluation.evaluate import evaluate
from crosslearner.models.acx import ACX
from crosslearner.training.train_acx import train_acx
from crosslearner.training import ModelConfig, TrainingConfig
from crosslearner.training.trainer import ACXTrainer
from crosslearner.utils import GNSBatchScheduler
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
        early_stop_metric="pehe",
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
        early_stop_metric="risk",
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
        early_stop_metric="pehe",
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


def test_head_specific_optimizer_kwargs():
    loader, _ = get_toy_dataloader(batch_size=8, n=32, p=3)
    model_cfg = ModelConfig(p=3)
    train_cfg = TrainingConfig(
        epochs=1,
        optimizer="sgd",
        opt_g_kwargs={"momentum": 0.0},
        opt_phi_kwargs={"momentum": 0.1},
        opt_head_kwargs={"momentum": 0.2},
        opt_disc_kwargs={"momentum": 0.3},
        verbose=False,
    )
    trainer = ACXTrainer(model_cfg, train_cfg, device="cpu")
    opt_g, opt_d = trainer._make_optimizers()
    assert opt_g.param_groups[0]["momentum"] == 0.1
    assert opt_g.param_groups[1]["momentum"] == 0.2
    assert opt_d.param_groups[0]["momentum"] == 0.3


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


@pytest.mark.parametrize(
    "norm, cls",
    [
        ("batch", nn.BatchNorm1d),
        ("layer", nn.LayerNorm),
        ("group", nn.GroupNorm),
    ],
)
def test_train_acx_norm_option(norm, cls):
    loader, _ = get_toy_dataloader(batch_size=4, n=8, p=4)
    model_cfg = ModelConfig(p=4, normalization=norm)
    train_cfg = TrainingConfig(epochs=1, verbose=False)
    model = train_acx(loader, model_cfg, train_cfg, device="cpu")
    assert any(isinstance(m, cls) for m in model.phi.net.modules())


def test_alt_adv_losses():
    loader, _ = get_toy_dataloader(batch_size=4, n=8, p=4)
    model_cfg = ModelConfig(p=4)
    for loss in ("hinge", "lsgan", "rgan"):
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


def test_train_acx_moe_heads():
    loader, _ = get_toy_dataloader(batch_size=4, n=8, p=4)
    model_cfg = ModelConfig(p=4, moe_experts=2)
    cfg = TrainingConfig(epochs=1, moe_entropy_weight=0.1, verbose=False)
    model = train_acx(loader, model_cfg, cfg, device="cpu")
    assert model.use_moe
    cfg = TrainingConfig(epochs=1, r2_gamma=0.1, verbose=False)
    model = train_acx(loader, model_cfg, cfg, device="cpu")
    assert isinstance(model, ACX)
    cfg = TrainingConfig(epochs=1, unrolled_steps=1, verbose=False)
    model = train_acx(loader, model_cfg, cfg, device="cpu")
    assert isinstance(model, ACX)


def test_unrolled_steps_epochs():
    loader, _ = get_toy_dataloader(batch_size=4, n=8, p=4)
    model_cfg = ModelConfig(p=4)
    cfg = TrainingConfig(
        epochs=2,
        unrolled_steps=1,
        unrolled_steps_epochs=1,
        verbose=False,
    )
    model = train_acx(loader, model_cfg, cfg, device="cpu")
    assert isinstance(model, ACX)


def test_train_acx_contrastive_loss():
    loader, _ = get_toy_dataloader(batch_size=4, n=8, p=4)
    model_cfg = ModelConfig(p=4)
    cfg = TrainingConfig(
        epochs=1,
        contrastive_weight=1.0,
        contrastive_margin=0.5,
        contrastive_noise=0.01,
        verbose=False,
    )
    model = train_acx(loader, model_cfg, cfg, device="cpu")
    assert isinstance(model, ACX)


def test_train_acx_doubly_robust():
    loader, _ = get_toy_dataloader(batch_size=4, n=8, p=4)
    model_cfg = ModelConfig(p=4)
    cfg = TrainingConfig(
        epochs=1,
        delta_prop=1.0,
        lambda_dr=0.1,
        verbose=False,
    )
    model = train_acx(loader, model_cfg, cfg, device="cpu")
    assert hasattr(model, "prop")
    assert hasattr(model, "epsilon")


def test_train_acx_noise_consistency():
    loader, _ = get_toy_dataloader(batch_size=4, n=8, p=4)
    model_cfg = ModelConfig(p=4)
    cfg = TrainingConfig(
        epochs=1,
        noise_std=0.1,
        noise_consistency_weight=0.5,
        verbose=False,
    )
    model = train_acx(loader, model_cfg, cfg, device="cpu")
    assert isinstance(model, ACX)


def test_train_acx_rep_consistency():
    loader, _ = get_toy_dataloader(batch_size=4, n=8, p=4)
    model_cfg = ModelConfig(p=4)
    cfg = TrainingConfig(
        epochs=2,
        rep_consistency_weight=0.5,
        verbose=False,
    )
    model = train_acx(loader, model_cfg, cfg, device="cpu")
    assert isinstance(model, ACX)


def test_active_counterfactual_augmentation():
    loader, _ = get_toy_dataloader(batch_size=4, n=16, p=4, seed=0)
    model_cfg = ModelConfig(p=4)
    cfg = TrainingConfig(
        epochs=2,
        active_aug_freq=1,
        active_aug_samples=4,
        active_aug_steps=1,
        verbose=False,
    )
    model = train_acx(loader, model_cfg, cfg, device="cpu")
    assert isinstance(model, ACX)


def test_augment_loader_preserves_settings():
    loader, _ = get_toy_dataloader(batch_size=4, n=8, p=4)
    model_cfg = ModelConfig(p=4)
    cfg = TrainingConfig(
        active_aug_freq=1, active_aug_samples=2, active_aug_steps=1, verbose=False
    )
    trainer = ACXTrainer(model_cfg, cfg, device="cpu")
    trainer._pseudo_data = trainer._search_disagreement(2, 1, 0.1)
    aug = trainer._augment_loader(loader)
    assert aug.batch_size == loader.batch_size
    assert aug.num_workers == loader.num_workers
    assert aug.pin_memory == loader.pin_memory
    assert aug.drop_last == loader.drop_last


def test_train_acx_epistemic_consistency():
    loader, _ = get_toy_dataloader(batch_size=4, n=8, p=4)
    model_cfg = ModelConfig(p=4, tau_heads=3)
    cfg = TrainingConfig(
        epochs=1,
        epistemic_consistency=True,
        verbose=False,
    )
    model = train_acx(loader, model_cfg, cfg, device="cpu")
    assert isinstance(model, ACX)


def test_epistemic_consistency_requires_multiple_heads():
    loader, _ = get_toy_dataloader(batch_size=4, n=8, p=4)
    model_cfg = ModelConfig(p=4, tau_heads=1)
    cfg = TrainingConfig(epochs=1, epistemic_consistency=True, verbose=False)
    with pytest.raises(ValueError):
        train_acx(loader, model_cfg, cfg, device="cpu")


def test_adaptive_regularization_updates_lambda():
    loader, _ = get_toy_dataloader(batch_size=4, n=16, p=4)
    model_cfg = ModelConfig(p=4)
    cfg = TrainingConfig(
        epochs=2,
        lambda_gp=0.2,
        adaptive_reg=True,
        d_reg_upper=0.6,
        reg_factor=2.0,
        lambda_gp_min=0.05,
        verbose=False,
    )
    train_acx(loader, model_cfg, cfg, device="cpu")
    assert cfg.lambda_gp <= 0.1


def test_search_disagreement_no_param_gradients():
    loader, _ = get_toy_dataloader(batch_size=4, n=8, p=4)
    model_cfg = ModelConfig(p=4)
    cfg = TrainingConfig(
        active_aug_freq=1, active_aug_samples=2, active_aug_steps=1, verbose=False
    )
    trainer = ACXTrainer(model_cfg, cfg, device="cpu")
    trainer._search_disagreement(2, 1, 0.1)
    assert all(p.grad is None for p in trainer.model.parameters())


def test_pretrain_representation():
    loader, _ = get_toy_dataloader(batch_size=4, n=8, p=4)
    model_cfg = ModelConfig(p=4)
    cfg = TrainingConfig(pretrain_epochs=1, epochs=1, verbose=False)
    train_acx(loader, model_cfg, cfg, device="cpu")
    assert cfg.lr_g < 1e-3


def test_pretrain_with_embeddings():
    X = torch.randn(8, 2)
    X_cat = torch.randint(0, 3, (8, 1))
    T = torch.randint(0, 2, (8, 1)).float()
    mu0 = X[:, :1]
    mu1 = mu0 + 1.0
    Y = torch.where(T.bool(), mu1, mu0) + 0.1 * torch.randn(8, 1)
    loader = DataLoader(TensorDataset(X, X_cat, T, Y), batch_size=4)
    model_cfg = ModelConfig(p=2, cat_dims=(3,), embed_dim=2)
    cfg = TrainingConfig(pretrain_epochs=1, epochs=1, verbose=False)
    train_acx(loader, model_cfg, cfg, device="cpu")
    assert cfg.lr_g < 1e-3


def test_adaptive_batch_parameters(monkeypatch):
    loader, _ = get_toy_dataloader(batch_size=2, n=8, p=4)
    model_cfg = ModelConfig(p=4)
    cfg = TrainingConfig(
        epochs=1,
        adaptive_batch=True,
        gns_target=1.0,
        gns_band=1.0,
        gns_growth_factor=2,
        gns_check_every=1,
        gns_plateau_patience=1,
        gns_ema=0.0,
        gns_max_batch=None,
        verbose=False,
    )
    trainer = ACXTrainer(model_cfg, cfg, device="cpu")
    monkeypatch.setattr(GNSBatchScheduler, "_grad_noise_scale", lambda self, a, b: 0.0)
    trainer.train(loader)
    assert trainer.scheduler is not None
    assert trainer.scheduler.max_B == len(loader.dataset)


def test_train_acx_gradnorm():
    loader, _ = get_toy_dataloader(batch_size=4, n=16, p=4)
    model_cfg = ModelConfig(p=4)
    cfg = TrainingConfig(epochs=1, use_gradnorm=True, verbose=False)
    trainer = ACXTrainer(model_cfg, cfg, device="cpu")
    trainer.train(loader)
    assert trainer.loss_weights.numel() == 3
