import torch
from crosslearner.training.grl import grad_reverse


def test_grad_reverse_backward():
    x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    y = grad_reverse(x, lambd=1.0)
    y.sum().backward()
    assert torch.allclose(x.grad, torch.tensor([-1.0, -1.0, -1.0]))
