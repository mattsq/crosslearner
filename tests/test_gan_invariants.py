import torch
from crosslearner.models.acx import ACX
from crosslearner.datasets.toy import get_toy_dataloader
import torch.nn as nn


def test_generator_preserves_batch_size():
    model = ACX(p=3)
    X = torch.randn(7, 3)
    _, m0, m1, tau = model(X)
    bs = X.size(0)
    assert m0.size(0) == bs
    assert m1.size(0) == bs
    assert tau.size(0) == bs


def test_discriminator_sigmoid_range():
    model = ACX(p=3)
    X = torch.randn(5, 3)
    h, _, _, _ = model(X)
    y = torch.randn(5, 1)
    t = torch.randint(0, 2, (5, 1)).float()
    logits = model.discriminator(h, y, t)
    probs = torch.sigmoid(logits)
    assert torch.all(probs >= 0)
    assert torch.all(probs <= 1)
    assert probs.shape == (5, 1)


def test_losses_non_negative():
    loader, _ = get_toy_dataloader(batch_size=4, n=4, p=3)
    X, T, Y = next(iter(loader))
    model = ACX(p=3)
    h, m0, m1, _ = model(X)
    mse = nn.MSELoss()
    bce = nn.BCEWithLogitsLoss()
    m_obs = torch.where(T.bool(), m1, m0)
    loss_y = mse(m_obs, Y)
    Ycf = torch.where(T.bool(), m0, m1)
    fake_logits = model.discriminator(h, Ycf, T)
    loss_adv = bce(fake_logits, torch.ones_like(fake_logits))
    assert loss_y.item() >= 0
    assert loss_adv.item() >= 0
