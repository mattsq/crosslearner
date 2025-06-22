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


class MOEHeads(nn.Module):
    """Mixture-of-experts potential-outcome heads."""

    def __init__(
        self,
        in_dim: int,
        num_experts: int,
        *,
        hidden: Iterable[int] | None,
        activation: str | Callable[[], nn.Module] = nn.ReLU,
        dropout: float = 0.0,
        residual: bool = False,
        batch_norm: bool = False,
    ) -> None:
        super().__init__()
        self.num_experts = int(num_experts)
        act_fn = _get_activation(activation)
        self.mu0 = nn.ModuleList(
            [
                MLP(
                    in_dim,
                    1,
                    hidden=hidden,
                    activation=act_fn,
                    dropout=dropout,
                    residual=residual,
                    batch_norm=batch_norm,
                )
                for _ in range(self.num_experts)
            ]
        )
        self.mu1 = nn.ModuleList(
            [
                MLP(
                    in_dim,
                    1,
                    hidden=hidden,
                    activation=act_fn,
                    dropout=dropout,
                    residual=residual,
                    batch_norm=batch_norm,
                )
                for _ in range(self.num_experts)
            ]
        )
        self.gate = MLP(
            in_dim,
            self.num_experts,
            hidden=hidden,
            activation=act_fn,
            dropout=dropout,
            residual=residual,
            batch_norm=batch_norm,
        )
        self.softmax = nn.Softmax(dim=1)
        self._gates: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        w = self.softmax(self.gate(x))
        self._gates = w.detach()
        m0 = torch.stack([head(x) for head in self.mu0], dim=1).squeeze(-1)
        m1 = torch.stack([head(x) for head in self.mu1], dim=1).squeeze(-1)
        m0 = (w * m0).sum(dim=1, keepdim=True)
        m1 = (w * m1).sum(dim=1, keepdim=True)
        return m0, m1

    @property
    def gates(self) -> torch.Tensor:
        if self._gates is None:
            raise RuntimeError("Gating weights have not been computed yet")
        return self._gates

    def entropy(self) -> torch.Tensor:
        w = self.gates.clamp_min(1e-12)
        return -(w * w.log()).sum(dim=1).mean()


class NullMOE(nn.Module):
    """Fallback when ``moe_experts`` is ``1``."""

    def __init__(self, mu0: nn.Module, mu1: nn.Module) -> None:
        super().__init__()
        self.mu0 = mu0
        self.mu1 = mu1

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.mu0(x), self.mu1(x)

    def entropy(self) -> torch.Tensor:
        return torch.tensor(0.0, device=next(self.mu0.parameters()).device)


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
        moe_experts: int = 1,
        tau_heads: int = 1,
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
            moe_experts: Number of expert pairs for the potential-outcome heads.
            tau_heads: Number of parallel effect heads for ensembling.
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

        self.use_moe = int(moe_experts) > 1
        self.moe_experts = int(moe_experts)

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
        if self.use_moe:
            self.moe = MOEHeads(
                head_in,
                self.moe_experts,
                hidden=head_layers,
                activation=act_fn,
                dropout=head_dropout,
                residual=head_residual,
                batch_norm=batch_norm,
            )
        else:
            self.moe = NullMOE(self.mu0, self.mu1)
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
        self.num_tau_heads = int(tau_heads)
        self.tau_heads = nn.ModuleList(
            [
                MLP(
                    head_in,
                    1,
                    hidden=head_layers,
                    activation=act_fn,
                    dropout=head_dropout,
                    residual=head_residual,
                    batch_norm=batch_norm,
                )
                for _ in range(self.num_tau_heads)
            ]
        )
        self.tau = self.tau_heads[0]
        self.register_buffer("_tau_var", torch.tensor(0.0), persistent=False)
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
            m0, m1 = self.moe(ha)
            tau_samples = torch.stack([head(ha) for head in self.tau_heads], dim=2)
        else:
            m0, m1 = self.moe(h)
            tau_samples = torch.stack([head(h) for head in self.tau_heads], dim=2)
        tau = tau_samples.mean(dim=2)
        self._tau_var = tau_samples.var(dim=2, unbiased=False)
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

    def head_parameters(self) -> Iterable[nn.Parameter]:
        """Return parameters of the potential-outcome heads."""

        if self.use_moe and self.moe is not None:
            return self.moe.parameters()
        return list(self.mu0.parameters()) + list(self.mu1.parameters())

    def tau_parameters(self) -> Iterable[nn.Parameter]:
        """Return parameters of all effect heads."""

        return [p for head in self.tau_heads for p in head.parameters()]

    @property
    def tau_variance(self) -> torch.Tensor:
        """Variance of ensemble treatment effect predictions from last forward."""
        return self._tau_var

    @property
    def effect_consistency_weight(self) -> torch.Tensor:
        """Inverse variance weight used for epistemic consistency loss."""
        return 1.0 / (1.0 + self._tau_var.detach())

    def moe_entropy(self) -> torch.Tensor:
        """Entropy of the gating distribution."""

        if not self.use_moe or self.moe is None:
            return torch.tensor(0.0, device=self.phi.out.weight.device)
        return self.moe.entropy()
