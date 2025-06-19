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
    """Adversarial-Consistency X-learner model.

    Optionally decomposes the representation into confounder-, outcome- and
    instrument-specific parts. When ``disentangle`` is ``True`` the encoder
    outputs ``rep_dim_c + rep_dim_a + rep_dim_i`` features which are routed to
    different heads and adversaries.
    """

    def __init__(
        self,
        p: int,
        *,
        rep_dim: int = 64,
        disentangle: bool = False,
        rep_dim_c: int | None = None,
        rep_dim_a: int | None = None,
        rep_dim_i: int | None = None,
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
            rep_dim: Dimensionality of the shared representation ``phi`` when
                ``disentangle`` is ``False``. Ignored otherwise.
            disentangle: If ``True`` split the representation into three parts
                with sizes ``rep_dim_c``, ``rep_dim_a`` and ``rep_dim_i``.
            rep_dim_c: Size of the confounder representation.
            rep_dim_a: Size of the outcome-specific representation.
            rep_dim_i: Size of the instrument representation.
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

        self.disentangle = disentangle
        if disentangle:
            if None in (rep_dim_c, rep_dim_a, rep_dim_i):
                raise ValueError(
                    "rep_dim_c, rep_dim_a and rep_dim_i must be specified when disentangle=True"
                )
            self.rep_dim_c = int(rep_dim_c)
            self.rep_dim_a = int(rep_dim_a)
            self.rep_dim_i = int(rep_dim_i)
            rep_dim_total = self.rep_dim_c + self.rep_dim_a + self.rep_dim_i
        else:
            rep_dim_total = rep_dim
            self.rep_dim_c = rep_dim_total
            self.rep_dim_a = 0
            self.rep_dim_i = 0

        self.rep_dim = rep_dim_total

        self.phi = MLP(
            p,
            rep_dim_total,
            hidden=phi_layers,
            activation=act_fn,
            dropout=phi_dropout,
            residual=phi_residual,
            batch_norm=batch_norm,
        )
        head_in = self.rep_dim_c + self.rep_dim_a if disentangle else rep_dim_total

        self.mu0 = MLP(
            head_in,
            1,
            hidden=head_layers,
            activation=act_fn,
            dropout=head_dropout,
            residual=head_residual,
            batch_norm=batch_norm,
        )
        self.mu1 = MLP(
            head_in,
            1,
            hidden=head_layers,
            activation=act_fn,
            dropout=head_dropout,
            residual=head_residual,
            batch_norm=batch_norm,
        )
        prop_in = self.rep_dim_c + self.rep_dim_i if disentangle else rep_dim_total
        self.prop = MLP(
            prop_in,
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
            head_in,
            1,
            hidden=head_layers,
            activation=act_fn,
            dropout=head_dropout,
            residual=head_residual,
            batch_norm=batch_norm,
        )
        self.disc_pack = max(1, int(disc_pack))
        self.disc = MLP(
            self.disc_pack * (rep_dim_total + 2),
            1,
            hidden=disc_layers,
            activation=act_fn,
            dropout=disc_dropout,
            residual=disc_residual,
            batch_norm=batch_norm,
        )
        if disentangle:
            self.adv_t = MLP(
                self.rep_dim_c + self.rep_dim_a,
                1,
                hidden=disc_layers,
                activation=act_fn,
                dropout=disc_dropout,
                residual=disc_residual,
                batch_norm=batch_norm,
            )
            self.adv_y = MLP(
                self.rep_dim_c + self.rep_dim_i,
                1,
                hidden=disc_layers,
                activation=act_fn,
                dropout=disc_dropout,
                residual=disc_residual,
                batch_norm=batch_norm,
            )
        else:
            self.adv_t = None
            self.adv_y = None

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
        if self.disentangle:
            zc, za, zi = torch.split(
                h, [self.rep_dim_c, self.rep_dim_a, self.rep_dim_i], dim=1
            )
            ha = torch.cat([zc, za], dim=1)
            m0 = self.mu0(ha)
            m1 = self.mu1(ha)
            tau = self.tau(ha)
        else:
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

        if self.disentangle:
            zc, _, zi = self.split(h)
            h_prop = torch.cat([zc, zi], dim=1)
            return self.prop(h_prop)
        return self.prop(h)

    @torch.jit.export
    def disc_features(
        self, h: torch.Tensor, y: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """Return discriminator features before the final linear layer."""

        x = torch.cat([h, y, t], dim=1)
        return self.disc.features(x)

    @torch.jit.export
    def split(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Split representation into ``(z_c, z_a, z_i)``."""

        if not self.disentangle:
            raise RuntimeError("Model was not created with disentangle=True")
        start = 0
        zc = h[:, start : start + self.rep_dim_c]
        start += self.rep_dim_c
        za = h[:, start : start + self.rep_dim_a]
        start += self.rep_dim_a
        zi = h[:, start : start + self.rep_dim_i]
        return zc, za, zi

    @torch.jit.export
    def adv_t_pred(self, zc: torch.Tensor, za: torch.Tensor) -> torch.Tensor:
        """Predict treatment assignment from ``(z_c, z_a)``."""

        if not self.disentangle or self.adv_t is None:
            raise RuntimeError("Model was not created with disentangle=True")
        return self.adv_t(torch.cat([zc, za], dim=1))

    @torch.jit.export
    def adv_y_pred(self, zc: torch.Tensor, zi: torch.Tensor) -> torch.Tensor:
        """Predict outcome from ``(z_c, z_i)``."""

        if not self.disentangle or self.adv_y is None:
            raise RuntimeError("Model was not created with disentangle=True")
        return self.adv_y(torch.cat([zc, zi], dim=1))
