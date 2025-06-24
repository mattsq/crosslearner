"""Gradient reversal layer used for adversarial training."""

import torch


class GradReverse(torch.autograd.Function):
    """Autograd function that scales the gradient by ``-lambd``."""

    @staticmethod
    def forward(ctx, x, lambd):
        """Forward pass that stores ``lambd`` for the backward step.

        Args:
            ctx: Autograd context provided by PyTorch.
            x: Input tensor.
            lambd: Scaling factor for the reversed gradient.

        Returns:
            The input ``x`` unchanged.
        """
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        """Reverse the gradient multiplied by ``lambd``.

        Args:
            ctx: Autograd context with ``lambd`` saved from ``forward``.
            grad_output: Upstream gradient tensor.

        Returns:
            Tuple containing the modified gradient and ``None`` for ``lambd``.
        """
        return grad_output.neg() * ctx.lambd, None


def grad_reverse(x, lambd=1.0):
    """Convenience wrapper for the gradient reversal function.

    Args:
        x: Input tensor.
        lambd: Scaling factor for the reversed gradient.

    Returns:
        ``x`` as a gradient-reversing autograd variable.
    """

    return GradReverse.apply(x, lambd)
