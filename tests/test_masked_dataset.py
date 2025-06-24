import torch
from torch.utils.data import TensorDataset
from crosslearner.datasets.masked import MaskedFeatureDataset


def test_masked_feature_dataset_returns_pair():
    X = torch.randn(8, 4)
    dset = MaskedFeatureDataset(TensorDataset(X), mask_prob=0.5)
    x_m, x = dset[0]
    assert x.shape == (4,)
    assert x_m.shape == (4,)
    assert torch.all((x_m == 0) | torch.isclose(x_m, x))


def test_masked_feature_dataset_with_categories():
    X = torch.randn(4, 2)
    X_cat = torch.randint(0, 3, (4, 1))
    dset = MaskedFeatureDataset(
        TensorDataset(X, X_cat, torch.zeros(4, 1), torch.zeros(4, 1))
    )
    x_m, x_cat, x = dset[0]
    assert x_cat.shape == (1,)
    assert x.shape == (2,)
    assert x_m.shape == (2,)
