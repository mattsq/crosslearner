import torch
from crosslearner.datasets.toy import get_toy_dataloader
from crosslearner.training import ModelConfig, TrainingConfig
from crosslearner.training.train_acx import train_acx_ensemble
from crosslearner.evaluation.uncertainty import predict_tau_ensemble
from crosslearner.models.acx import ACX
from crosslearner import set_seed


def test_train_acx_ensemble_simple():
    set_seed(0)
    loader, _ = get_toy_dataloader(batch_size=4, n=16, p=3)
    model_cfg = ModelConfig(p=3)
    train_cfg = TrainingConfig(epochs=1)
    models = train_acx_ensemble(
        loader, model_cfg, train_cfg, n_models=2, device="cpu", seed=0
    )
    assert len(models) == 2
    assert all(isinstance(m, ACX) for m in models)


def test_predict_tau_ensemble_mean_std():
    model1 = ACX(p=2)
    model2 = ACX(p=2)
    with torch.no_grad():
        for p in model1.parameters():
            p.zero_()
        for p in model2.parameters():
            p.zero_()
        model1.tau.out.bias.fill_(1.0)
        model2.tau.out.bias.fill_(2.0)
    X = torch.zeros(3, 2)
    mean, std = predict_tau_ensemble([model1, model2], X)
    assert torch.allclose(mean, torch.full((3, 1), 1.5))
    assert torch.allclose(std, torch.full((3, 1), 0.70710677))


def test_train_acx_ensemble_seed_none_reproducible():
    set_seed(0)
    loader, _ = get_toy_dataloader(batch_size=4, n=16, p=3)
    model_cfg = ModelConfig(p=3)
    train_cfg = TrainingConfig(epochs=1)
    models1 = train_acx_ensemble(loader, model_cfg, train_cfg, n_models=2, device="cpu")

    set_seed(0)
    loader, _ = get_toy_dataloader(batch_size=4, n=16, p=3)
    models2 = train_acx_ensemble(loader, model_cfg, train_cfg, n_models=2, device="cpu")

    for m1, m2 in zip(models1, models2):
        for p1, p2 in zip(m1.parameters(), m2.parameters()):
            assert torch.allclose(p1, p2)
