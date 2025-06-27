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
    val_loss_y: Optional[float] = None
    val_loss_cons: Optional[float] = None
    val_loss_adv: Optional[float] = None
    grad_norm_g: Optional[float] = None
    grad_norm_d: Optional[float] = None
    w_y: Optional[float] = None
    w_cons: Optional[float] = None
    w_adv: Optional[float] = None
    lr_g: Optional[float] = None
    lr_d: Optional[float] = None
    gns: Optional[float] = None
    batch_size: Optional[int] = None


History = List[EpochStats]
