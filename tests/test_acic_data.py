import os
import numpy as np
import torch
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from data.acic import load_acic


def _fake_npz(path: str) -> None:
    n, p = 4, 3
    np.savez(
        path,
        x=np.zeros((n, p, 1)),
        t=np.zeros((n, 1)),
        yf=np.zeros((n, 1)),
        mu0=np.zeros((n, 1)),
        mu1=np.ones((n, 1)),
    )


def test_load_acic_reproducible(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "data.acic.download_if_missing",
        lambda url, p: _fake_npz(p) or p,
    )
    (train1, val1, test1), (mu0_a, mu1_a) = load_acic(
        year=2018, seed=0, data_dir=tmp_path
    )
    (train2, val2, test2), (mu0_b, mu1_b) = load_acic(
        year=2018, seed=0, data_dir=tmp_path
    )

    assert torch.allclose(mu0_a, mu0_b)
    assert torch.allclose(mu1_a, mu1_b)
    for d1, d2 in zip((train1, val1, test1), (train2, val2, test2)):
        for t1, t2 in zip(d1.tensors, d2.tensors):
            assert torch.allclose(t1, t2)
