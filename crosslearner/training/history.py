from dataclasses import dataclass
from typing import List

@dataclass
class EpochStats:
    """Metrics tracked for a single training epoch."""
    epoch: int
    loss_d: float
    loss_g: float
    loss_y: float
    loss_cons: float
    loss_adv: float

History = List[EpochStats]
