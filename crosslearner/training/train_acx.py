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


def train_acx_ensemble(
    loader: DataLoader,
    model_config: ModelConfig,
    training_config: TrainingConfig,
    *,
    n_models: int,
    device: Optional[str] = None,
    seed: int | None = None,
) -> list[ACX] | tuple[list[ACX], list[History]]:
    """Train ``n_models`` independent ACX models.

    Each model is trained with the same configuration but with incremented
    random seeds for reproducibility.
    """

    models: list[ACX] = []
    histories: list[History] = []
    for i in range(n_models):
        model_seed = None if seed is None else seed + i
        trainer = ACXTrainer(
            model_config,
            training_config,
            device=device,
            seed=model_seed,
        )
        result = trainer.train(loader)
        if training_config.return_history:
            model, history = result  # type: ignore[misc]
            models.append(model)
            histories.append(history)
        else:
            models.append(result)  # type: ignore[arg-type]
    return (models, histories) if training_config.return_history else models
