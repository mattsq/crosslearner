from __future__ import annotations

from typing import Optional, Tuple
from torch.utils.data import DataLoader

from .config import ModelConfig, TrainingConfig
from .history import History
from ..models.acx import ACX, _get_activation
from ..utils import set_seed, default_device, apply_spectral_norm


class ACXTrainer:
    """Trainer class encapsulating the ACX training loop."""

    def __init__(
        self,
        model_cfg: ModelConfig,
        train_cfg: TrainingConfig,
        *,
        device: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> None:
        self.model_cfg = model_cfg
        self.train_cfg = train_cfg
        self.device = device or default_device()
        if seed is not None:
            set_seed(seed)
        act_fn = _get_activation(model_cfg.activation)
        self.model = ACX(
            model_cfg.p,
            rep_dim=model_cfg.rep_dim,
            phi_layers=model_cfg.phi_layers,
            head_layers=model_cfg.head_layers,
            disc_layers=model_cfg.disc_layers,
            activation=act_fn,
            phi_dropout=model_cfg.phi_dropout,
            head_dropout=model_cfg.head_dropout,
            disc_dropout=model_cfg.disc_dropout,
            residual=model_cfg.residual,
        ).to(self.device)
        if train_cfg.spectral_norm:
            apply_spectral_norm(self.model)

    def train(self, loader: DataLoader) -> ACX | Tuple[ACX, History]:
        from .train_acx import train_acx

        args = dict(vars(self.model_cfg))
        args.update(vars(self.train_cfg))
        args.update({"device": self.device})
        return train_acx(loader, **args)
