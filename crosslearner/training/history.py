"""Utilities for tracking training statistics."""

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class EpochStats:
    """Metrics tracked for a single training epoch."""

    epoch: int
    loss_d: float
    loss_g: float
    loss_y: float
    loss_cons: float
    loss_adv: float
    val_pehe: Optional[float] = None


History = List[EpochStats]
