"""Core AC-X model definition."""

import torch
import torch.nn as nn


class MLP(nn.Module):
    """Simple multi-layer perceptron used throughout the framework.

    Args:
        in_dim: Size of the input features.
        out_dim: Output dimension.
        hidden: Tuple with hidden layer sizes.
    """

    def __init__(self, in_dim: int, out_dim: int, hidden=(128, 64)) -> None:
        super().__init__()
        layers = []
        d = in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU()]
            d = h
        layers += [nn.Linear(d, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the network.

        Args:
            x: Input tensor of shape ``(batch, in_dim)``.

        Returns:
            Tensor with shape ``(batch, out_dim)``.
        """

        return self.net(x)


class ACX(nn.Module):
    """Adversarial-Consistency X-learner model."""

    def __init__(self, p: int, rep_dim: int = 64) -> None:
        """Instantiate the model.

        Args:
            p: Number of covariates.
            rep_dim: Dimensionality of the shared representation ``phi``.
        """

        super().__init__()
        self.phi = MLP(p, rep_dim, hidden=(128,))
        self.mu0 = MLP(rep_dim, 1, hidden=(64,))
        self.mu1 = MLP(rep_dim, 1, hidden=(64,))
        self.tau = MLP(rep_dim, 1, hidden=(64,))
        self.disc = MLP(rep_dim + 2, 1, hidden=(64,))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Forward pass.

        Args:
            x: Input covariates of shape ``(batch, p)``.

        Returns:
            Tuple ``(h, mu0, mu1, tau)`` containing the shared
            representation and head outputs.
        """

        h = self.phi(x)
        m0 = self.mu0(h)
        m1 = self.mu1(h)
        tau = self.tau(h)
        return h, m0, m1, tau

    def discriminator(
        self, h: torch.Tensor, y: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """Evaluate the discriminator.

        Args:
            h: Representation from ``phi``.
            y: Outcome tensor.
            t: Treatment indicator.

        Returns:
            Discriminator logits.
        """

        return self.disc(torch.cat([h, y, t], dim=1))
