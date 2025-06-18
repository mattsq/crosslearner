from __future__ import annotations

from typing import Optional

from torch.utils.data import DataLoader

from .config import ModelConfig, TrainingConfig
from .trainer import ACXTrainer
from .history import History
from ..models.acx import ACX


def train_acx(
    loader: DataLoader,
    model_config: ModelConfig,
    training_config: TrainingConfig,
    *,
    device: Optional[str] = None,
    seed: int | None = None,
) -> ACX | tuple[ACX, History]:
    """Train an ACX model using configuration objects."""

    trainer = ACXTrainer(model_config, training_config, device=device, seed=seed)
    return trainer.train(loader)
