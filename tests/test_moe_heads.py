import torch

from crosslearner.models.acx import MOEHeads


def test_moe_heads_gating_weights_sum_to_one():
    moe = MOEHeads(in_dim=4, num_experts=3, hidden=(8,))
    x = torch.randn(5, 4)
    m0, m1 = moe(x)
    w = moe.gates
    assert w.shape == (5, 3)
    assert torch.allclose(w.sum(dim=1), torch.ones(5))
    assert not w.requires_grad
    assert m0.shape == (5, 1)
    assert m1.shape == (5, 1)


def test_moe_entropy_matches_manual_computation():
    moe = MOEHeads(in_dim=6, num_experts=4, hidden=(10,))
    x = torch.randn(3, 6)
    moe(x)
    w = moe.gates
    expected = -(w.clamp_min(1e-12) * w.clamp_min(1e-12).log()).sum(dim=1).mean()
    assert torch.allclose(moe.entropy(), expected)
