import torch
from crosslearner.datasets.toy import get_toy_dataloader
from crosslearner.training.nuisance import estimate_nuisances


def test_estimate_nuisances_output_shapes():
    loader, _ = get_toy_dataloader(batch_size=8, n=16, p=3, seed=0)
    X = torch.cat([b[0] for b in loader])
    T = torch.cat([b[1] for b in loader])
    Y = torch.cat([b[2] for b in loader])
    e_hat, mu0_hat, mu1_hat = estimate_nuisances(
        X,
        T,
        Y,
        folds=2,
        lr=1e-2,
        batch=8,
        propensity_epochs=1,
        outcome_epochs=1,
        early_stop=1,
        device="cpu",
        seed=0,
    )
    assert e_hat.shape == T.shape
    assert mu0_hat.shape == Y.shape
    assert mu1_hat.shape == Y.shape
    # predictions should not all be identical
    assert e_hat.std() > 0
    assert mu0_hat.std() > 0
    assert mu1_hat.std() > 0
