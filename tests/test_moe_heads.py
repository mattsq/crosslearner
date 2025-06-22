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
