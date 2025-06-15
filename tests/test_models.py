import numpy as np
import torch

from crosslearner.models.acx import ACX
from crosslearner.models.baselines import DRLearner, SLearner, TLearner, XLearner


def _make_data(n=20, p=3):
    rng = np.random.default_rng(0)
    X = rng.standard_normal((n, p))
    T = rng.integers(0, 2, size=(n, 1))
    Y = X[:, :1] + T + rng.normal(scale=0.1, size=(n, 1))
    return X, T, Y


def test_acx_forward_shapes():
    model = ACX(p=5)
    X = torch.randn(2, 5)
    h, m0, m1, tau = model(X)
    assert h.shape == (2, 64)
    assert m0.shape == (2, 1)
    assert m1.shape == (2, 1)
    assert tau.shape == (2, 1)


def test_slearner_fit_predict():
    X, T, Y = _make_data()
    model = SLearner(p=X.shape[1])
    model.fit(X, T, Y)
    tau = model.predict_tau(X)
    assert tau.shape == (X.shape[0],)


def test_tlearner_fit_predict():
    X, T, Y = _make_data()
    model = TLearner(p=X.shape[1])
    model.fit(X, T, Y)
    tau = model.predict_tau(X)
    assert tau.shape == (X.shape[0],)


def test_xlearner_fit_predict():
    X, T, Y = _make_data()
    model = XLearner(p=X.shape[1])
    model.fit(X, T, Y)
    tau = model.predict_tau(X)
    assert tau.shape == (X.shape[0],)


def test_drlearner_fit_predict():
    X, T, Y = _make_data()
    model = DRLearner(p=X.shape[1])
    model.fit(X, T, Y)
    tau = model.predict_tau(X)
    assert tau.shape == (X.shape[0],)
