from __future__ import annotations

from typing import Optional
import random

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
    """Train an ACX model using configuration objects.

    Args:
        loader: Training data loader.
        model_config: Configuration for the ``ACX`` architecture.
        training_config: Hyperparameters controlling training.
        device: Optional device string.
        seed: Optional random seed.

    Returns:
        The trained model or ``(model, history)`` when ``return_history`` is
        enabled.
    """

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

    Args:
        loader: Training data loader.
        model_config: Configuration for the ``ACX`` architecture.
        training_config: Hyperparameters controlling training.
        n_models: Number of models to train.
        device: Optional device string.
        seed: Optional base random seed. Different seeds are derived from it.

    Returns:
        List of trained models or additionally their histories when requested.

    Each model is trained with the same configuration. When ``seed`` is
    provided, subsequent models are initialised with ``seed + i``. If ``seed``
    is ``None``, a random base seed is drawn and incremented so that each model
    receives a distinct seed.
    """

    base_seed = seed if seed is not None else random.randrange(2**32)
    models: list[ACX] = []
    histories: list[History] = []
    for i in range(n_models):
        model_seed = base_seed + i
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
