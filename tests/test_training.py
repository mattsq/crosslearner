import torch
from crosslearner.datasets.toy import get_toy_dataloader
from crosslearner.training.train_acx import train_acx
from crosslearner.evaluation.evaluate import evaluate


def test_train_acx_short():
    torch.manual_seed(0)
    loader, (mu0, mu1) = get_toy_dataloader(batch_size=16, n=64, p=4)
    model = train_acx(loader, p=4, device='cpu', epochs=2)
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
