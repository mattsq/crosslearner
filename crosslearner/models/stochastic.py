import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Type


class DropConnectLinear(nn.Linear):
    """Linear layer with DropConnect regularization."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        dropconnect_prob: float = 0.0,
    ) -> None:
        super().__init__(in_features, out_features, bias=bias)
        self.dropconnect_prob = float(dropconnect_prob)

    def forward(self, input: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        weight = self.weight
        if self.training and self.dropconnect_prob > 0.0:
            mask = (torch.rand_like(weight) > self.dropconnect_prob).float()
            mask = mask / (1 - self.dropconnect_prob)
            weight = weight * mask
        return F.linear(input, weight, self.bias)


class StochasticEnsemble(nn.Module):
    """Ensemble of stochastic networks supporting MC-dropout predictions."""

    def __init__(
        self, base_model_cls: Type[nn.Module], ensemble_size: int = 5, **base_kwargs
    ) -> None:
        super().__init__()
        if ensemble_size <= 0:
            raise ValueError("ensemble_size must be positive")
        self.members = nn.ModuleList(
            [base_model_cls(**base_kwargs) for _ in range(ensemble_size)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        preds = [m(x) for m in self.members]
        return torch.stack(preds, dim=0).mean(dim=0)

    @torch.no_grad()
    def mc_dropout_predict(self, x: torch.Tensor, mc_passes: int = 10) -> torch.Tensor:
        """Return predictions from all ensemble members with active dropout."""

        preds = []
        for member in self.members:
            member.train()
            for _ in range(mc_passes):
                preds.append(member(x))
            member.eval()
        return torch.stack(preds, dim=0)


class BaseNet(nn.Module):
    """Simple MLP using dropout and DropConnectLinear."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        *,
        drop_prob: float = 0.0,
        dropconnect_prob: float = 0.0,
    ) -> None:
        super().__init__()
        self.fc1 = DropConnectLinear(
            input_dim, hidden_dim, dropconnect_prob=dropconnect_prob
        )
        self.dropout = nn.Dropout(drop_prob)
        self.fc2 = DropConnectLinear(hidden_dim, 1, dropconnect_prob=dropconnect_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)
