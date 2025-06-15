"""Gradient reversal layer used for adversarial training."""

import torch


class GradReverse(torch.autograd.Function):
    """Autograd function that scales the gradient by ``-lambd``."""

    @staticmethod
    def forward(ctx, x, lambd):
        """Forward pass that stores ``lambd`` for the backward step."""
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        """Reverse the gradient multiplied by ``lambd``."""
        return grad_output.neg() * ctx.lambd, None


def grad_reverse(x, lambd=1.0):
    """Convenience wrapper for the gradient reversal function."""

    return GradReverse.apply(x, lambd)
