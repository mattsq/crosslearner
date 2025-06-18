# Training script for causal_consistency_nn
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, asdict, field
from typing import Tuple, Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml

from .config import SyntheticDataConfig
from .data.synthetic import get_synthetic_dataloader
from crosslearner.utils import set_seed, default_device


class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(
        self, x: torch.Tensor
    ) -> torch.Tensor:  # pragma: no cover - thin wrapper
        return self.net(x)


class Backbone(MLP):
    pass


class ZgivenXY(MLP):
    pass


class YgivenXZ(MLP):
    pass


class XgivenYZ(MLP):
    pass


@dataclass
class Settings:
    """Training configuration."""

    data: SyntheticDataConfig = field(default_factory=SyntheticDataConfig)
    batch_size: int = 64
    epochs: int = 5
    lr: float = 1e-3
    backbone_dim: int = 16
    z_dim: int = 8
    hidden_dim: int = 32
    out_dir: str = "outputs"
    seed: int | None = None
    device: str | None = None


@torch.no_grad()
def _eval_loss(
    loader: DataLoader,
    backbone: Backbone,
    zgivenxy: ZgivenXY,
    ygivenxz: YgivenXZ,
    xgivenyz: XgivenYZ,
    device: torch.device,
) -> float:
    mse = nn.MSELoss()
    total = 0.0
    count = 0
    for Xb, _, Yb in loader:
        Xb = Xb.to(device)
        Yb = Yb.to(device)
        h = backbone(Xb)
        z = zgivenxy(torch.cat([h, Yb], dim=1))
        pred_y = ygivenxz(torch.cat([h, z], dim=1))
        pred_x = xgivenyz(torch.cat([Yb, z], dim=1))
        loss = mse(pred_y, Yb) + mse(pred_x, Xb)
        total += float(loss) * Xb.size(0)
        count += Xb.size(0)
    return total / count if count else 0.0


def train_em(settings: Settings) -> Tuple[list[float], Dict[str, nn.Module]]:
    """Train simple networks on synthetic data."""

    if settings.seed is not None:
        set_seed(settings.seed)
    device = torch.device(settings.device or default_device())

    loader, _ = get_synthetic_dataloader(
        settings.data, batch_size=settings.batch_size, shuffle=True
    )

    p = settings.data.p
    backbone = Backbone(p, settings.backbone_dim, settings.hidden_dim).to(device)
    zgivenxy = ZgivenXY(
        settings.backbone_dim + 1, settings.z_dim, settings.hidden_dim
    ).to(device)
    ygivenxz = YgivenXZ(
        settings.backbone_dim + settings.z_dim, 1, settings.hidden_dim
    ).to(device)
    xgivenyz = XgivenYZ(settings.z_dim + 1, p, settings.hidden_dim).to(device)

    optim = torch.optim.Adam(
        list(backbone.parameters())
        + list(zgivenxy.parameters())
        + list(ygivenxz.parameters())
        + list(xgivenyz.parameters()),
        lr=settings.lr,
    )
    mse = nn.MSELoss()

    losses = [_eval_loss(loader, backbone, zgivenxy, ygivenxz, xgivenyz, device)]
    for _ in range(settings.epochs):
        for Xb, _, Yb in loader:
            Xb = Xb.to(device)
            Yb = Yb.to(device)
            h = backbone(Xb)
            z = zgivenxy(torch.cat([h, Yb], dim=1))
            pred_y = ygivenxz(torch.cat([h, z], dim=1))
            pred_x = xgivenyz(torch.cat([Yb, z], dim=1))
            loss = mse(pred_y, Yb) + mse(pred_x, Xb)
            optim.zero_grad()
            loss.backward()
            optim.step()
        losses.append(
            _eval_loss(loader, backbone, zgivenxy, ygivenxz, xgivenyz, device)
        )

    models = {
        "backbone": backbone,
        "zgivenxy": zgivenxy,
        "ygivenxz": ygivenxz,
        "xgivenyz": xgivenyz,
    }
    return losses, models


def save_checkpoint(models: Dict[str, nn.Module], settings: Settings) -> None:
    os.makedirs(settings.out_dir, exist_ok=True)
    torch.save(
        {k: m.state_dict() for k, m in models.items()},
        os.path.join(settings.out_dir, "checkpoint.pt"),
    )
    with open(os.path.join(settings.out_dir, "config.yaml"), "w") as f:
        yaml.safe_dump(asdict(settings), f)


def main(args: list[str] | None = None) -> None:  # pragma: no cover - CLI
    parser = argparse.ArgumentParser(description="Train simple EM model")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--out-dir", type=str, default=None)
    parsed = parser.parse_args(args)

    if parsed.config:
        with open(parsed.config) as f:
            cfg_dict = yaml.safe_load(f)
        settings = Settings(**cfg_dict)
    else:
        settings = Settings()

    if parsed.out_dir:
        settings.out_dir = parsed.out_dir

    losses, models = train_em(settings)
    save_checkpoint(models, settings)
    print("Final loss:", losses[-1])


if __name__ == "__main__":  # pragma: no cover - CLI
    main()
