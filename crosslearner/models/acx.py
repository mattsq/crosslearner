"""Core AC-X model definition."""

import torch
import torch.nn as nn
from typing import Callable, Iterable


def _get_activation(act: str | Callable[[], nn.Module]) -> Callable[[], nn.Module]:
    """Return an activation constructor from string or callable.

    The function accepts either a string identifier or a callable returning a
    fresh ``nn.Module`` instance.  Passing an ``nn.Module`` instance is a common
    mistake that leads to cryptic runtime errors, therefore it is explicitly
    disallowed.
    """

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

    if isinstance(act, nn.Module):
        raise TypeError(
            "activation must be a string or a callable returning a new module;"
            f" got instance of {act.__class__.__name__}"
        )

    if not callable(act):
        raise TypeError("activation must be a string or callable")

    return act


class MLP(nn.Module):
    """Simple multi-layer perceptron used throughout the framework.

    Args:
        in_dim: Size of the input features.
        out_dim: Output dimension.
        hidden: Iterable with hidden layer sizes.
        activation: Activation function to apply between layers.
        dropout: Dropout probability applied after each hidden layer.
        residual: If ``True`` add skip connections between blocks when the
            dimensions match.
        batch_norm: If ``True`` add ``BatchNorm1d`` after each hidden linear
            layer.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden: Iterable[int] | None = None,
        *,
        activation: str | Callable[[], nn.Module] = nn.ReLU,
        dropout: float = 0.0,
        residual: bool = False,
        batch_norm: bool = False,
    ) -> None:
        super().__init__()
        self.residual = residual
        self.blocks = nn.ModuleList()
        d = in_dim
        hidden = tuple(hidden or [])
        act_fn = _get_activation(activation)
        dropout = float(dropout)
        if not (0 <= dropout < 1):
            raise ValueError(f"Dropout must be in the range [0, 1), but got {dropout}.")
        for h in hidden:
            block = [nn.Linear(d, h)]
            if batch_norm:
                block.append(nn.BatchNorm1d(h))
            block.append(act_fn())
            if dropout > 0:
                block.append(nn.Dropout(dropout))
            self.blocks.append(nn.Sequential(*block))
            d = h
        self.out = nn.Linear(d, out_dim)
        self.net = nn.Sequential(*self.blocks, self.out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the network.

        Args:
            x: Input tensor of shape ``(batch, in_dim)``.

        Returns:
            Tensor with shape ``(batch, out_dim)``.
        """

        if self.residual:
            h = x
            for block in self.blocks:
                z = block(h)
                if z.shape == h.shape:
                    h = z + h
                else:
                    h = z
            return self.out(h)
        else:
            return self.net(x)

    @torch.jit.export
    def features(self, x: torch.Tensor) -> torch.Tensor:
        """Return hidden representations before the final linear layer."""
        if self.residual:
            h = x
            for block in self.blocks:
                z = block(h)
                if z.shape == h.shape:
                    h = z + h
                else:
                    h = z
            return h
        else:
            for block in self.blocks:
                x = block(x)
            return x


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
        phi_dropout: float = 0.0,
        head_dropout: float = 0.0,
        disc_dropout: float = 0.0,
        residual: bool = False,
        phi_residual: bool | None = None,
        head_residual: bool | None = None,
        disc_residual: bool | None = None,
        disc_pack: int = 1,
        batch_norm: bool = False,
    ) -> None:
        """Instantiate the model.

        Args:
            p: Number of covariates.
            rep_dim: Dimensionality of the shared representation ``phi``.
            phi_layers: Sizes of hidden layers for the representation MLP.
            head_layers: Hidden layers for the outcome and effect heads.
            disc_layers: Hidden layers for the discriminator.
            activation: Activation function used in all networks.
            phi_dropout: Dropout probability for the representation MLP.
            head_dropout: Dropout probability for the outcome and effect heads.
            disc_dropout: Dropout probability for the discriminator.
            residual: Enable residual connections in all MLPs.
            phi_residual: Override residual connections just for ``phi``.
            head_residual: Override residual connections for outcome and effect
                heads.
            disc_residual: Override residual connections for the discriminator.
            disc_pack: Number of samples concatenated for the discriminator.
            batch_norm: Insert ``BatchNorm1d`` layers in all MLPs.
        """

        super().__init__()
        act_fn = _get_activation(activation)

        phi_residual = residual if phi_residual is None else phi_residual
        head_residual = residual if head_residual is None else head_residual
        disc_residual = residual if disc_residual is None else disc_residual

        self.phi = MLP(
            p,
            rep_dim,
            hidden=phi_layers,
            activation=act_fn,
            dropout=phi_dropout,
            residual=phi_residual,
            batch_norm=batch_norm,
        )
        self.mu0 = MLP(
            rep_dim,
            1,
            hidden=head_layers,
            activation=act_fn,
            dropout=head_dropout,
            residual=head_residual,
            batch_norm=batch_norm,
        )
        self.mu1 = MLP(
            rep_dim,
            1,
            hidden=head_layers,
            activation=act_fn,
            dropout=head_dropout,
            residual=head_residual,
            batch_norm=batch_norm,
        )
        self.prop = MLP(
            rep_dim,
            1,
            hidden=head_layers,
            activation=act_fn,
            dropout=head_dropout,
            residual=head_residual,
            batch_norm=batch_norm,
        )
        self.prop.net.add_module("sigmoid", nn.Sigmoid())
        self.epsilon = nn.Parameter(torch.tensor(0.0))
        self.tau = MLP(
            rep_dim,
            1,
            hidden=head_layers,
            activation=act_fn,
            dropout=head_dropout,
            residual=head_residual,
            batch_norm=batch_norm,
        )
        self.disc_pack = max(1, int(disc_pack))
        self.disc = MLP(
            self.disc_pack * (rep_dim + 2),
            1,
            hidden=disc_layers,
            activation=act_fn,
            dropout=disc_dropout,
            residual=disc_residual,
            batch_norm=batch_norm,
        )

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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

    @torch.jit.export
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

    @torch.jit.export
    def propensity(self, h: torch.Tensor) -> torch.Tensor:
        """Return propensity score predictions from representation ``h``."""

        return self.prop(h)

    @torch.jit.export
    def disc_features(
        self, h: torch.Tensor, y: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """Return discriminator features before the final linear layer."""

        x = torch.cat([h, y, t], dim=1)
        return self.disc.features(x)
