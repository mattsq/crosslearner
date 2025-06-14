"""Core AC-X model definition."""

import torch
import torch.nn as nn
from typing import Callable, Iterable


def _get_activation(act: str | Callable[[], nn.Module]) -> Callable[[], nn.Module]:
    """Return an activation constructor from string or callable."""
    if isinstance(act, str):
        name = act.lower()
        mapping = {
            "relu": nn.ReLU,
            "tanh": nn.Tanh,
            "elu": nn.ELU,
            "gelu": nn.GELU,
            "leakyrelu": nn.LeakyReLU,
        }
        if name not in mapping:
            raise ValueError(f"Unknown activation '{act}'")
        return mapping[name]
    return act


class MLP(nn.Module):
    """Simple multi-layer perceptron used throughout the framework.

    Args:
        in_dim: Size of the input features.
        out_dim: Output dimension.
        hidden: Iterable with hidden layer sizes.
        activation: Activation function to apply between layers.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden: Iterable[int] | None = None,
        *,
        activation: str | Callable[[], nn.Module] = nn.ReLU,
    ) -> None:
        super().__init__()
        layers = []
        d = in_dim
        hidden = tuple(hidden or [])
        act_fn = _get_activation(activation)
        for h in hidden:
            layers += [nn.Linear(d, h), act_fn()]
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

    def __init__(
        self,
        p: int,
        *,
        rep_dim: int = 64,
        phi_layers: Iterable[int] | None = (128,),
        head_layers: Iterable[int] | None = (64,),
        disc_layers: Iterable[int] | None = (64,),
        activation: str | Callable[[], nn.Module] = nn.ReLU,
    ) -> None:
        """Instantiate the model.

        Args:
            p: Number of covariates.
            rep_dim: Dimensionality of the shared representation ``phi``.
            phi_layers: Sizes of hidden layers for the representation MLP.
            head_layers: Hidden layers for the outcome and effect heads.
            disc_layers: Hidden layers for the discriminator.
            activation: Activation function used in all networks.
        """

        super().__init__()
        act_fn = _get_activation(activation)
        self.phi = MLP(p, rep_dim, hidden=phi_layers, activation=act_fn)
        self.mu0 = MLP(rep_dim, 1, hidden=head_layers, activation=act_fn)
        self.mu1 = MLP(rep_dim, 1, hidden=head_layers, activation=act_fn)
        self.tau = MLP(rep_dim, 1, hidden=head_layers, activation=act_fn)
        self.disc = MLP(rep_dim + 2, 1, hidden=disc_layers, activation=act_fn)

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
