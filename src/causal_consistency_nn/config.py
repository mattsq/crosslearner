from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class SyntheticDataConfig:
    """Configuration for the synthetic SCM dataset."""

    n_samples: int = 8000
    p: int = 10
    noise: float = 0.5
    missing_y_prob: float = 0.0
    seed: Optional[int] = None
