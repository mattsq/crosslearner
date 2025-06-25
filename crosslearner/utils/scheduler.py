"""Adaptive batch-size scheduler using gradient noise scale."""

from __future__ import annotations

import itertools
import math
from typing import Iterable, Optional

import torch
from torch.utils.data import BatchSampler, DataLoader, RandomSampler, SequentialSampler
from torch.cuda.amp import autocast


class MutableBatchSampler(BatchSampler):
    """Batch sampler whose batch size can be modified at runtime."""

    def __init__(
        self, sampler: Iterable[int], batch_size: int, drop_last: bool
    ) -> None:
        super().__init__(sampler, batch_size, drop_last)
        self.batch_size = batch_size


class GNSBatchScheduler:
    """Grow the batch size when the gradient noise scale is low."""

    def __init__(
        self,
        model: torch.nn.Module,
        loss_fn,
        dataloader: DataLoader,
        optimizer: torch.optim.Optimizer,
        target_gns: float = 1.0,
        band: float = 0.7,
        growth_factor: int = 2,
        check_every: int = 200,
        plateau_patience: int = 3,
        ema: float = 0.9,
        max_global_batch: Optional[int] = None,
    ) -> None:
        self.model = model
        self.loss_fn = loss_fn
        self.loader = dataloader
        self.opt = optimizer
        self.tgt = target_gns
        self.band_lo = band
        self.band_hi = 1.0 / band
        self.growth = growth_factor
        self.check_every = check_every
        self.patience = plateau_patience
        self.ema = ema
        self.bad_evals = 0
        self.best_val = math.inf
        self.step = 0
        self.max_B = max_global_batch
        self.smoothed_gns = 0.0
        self.base_lr = [
            g["lr"] / self.loader.batch_sampler.batch_size
            for g in self.opt.param_groups
        ]

    @torch.no_grad()
    def _grad_noise_scale(self, batch1, batch2) -> float:
        grads = []
        for b in (batch1, batch2):
            self.opt.zero_grad(set_to_none=True)
            with autocast():
                loss = self.loss_fn(self.model, b)
            loss.backward()
            g = torch.cat(
                [
                    p.grad.reshape(-1)
                    for p in self.model.parameters()
                    if p.grad is not None
                ]
            )
            grads.append(g.clone())
        diff = grads[0] - grads[1]
        gns = diff.pow(2).sum() / (2 * grads[0].pow(2).sum().clamp(min=1e-12))
        return gns.item()

    def _grow(self) -> None:
        old_B = self.loader.batch_sampler.batch_size
        new_B = (
            min(old_B * self.growth, self.max_B) if self.max_B else old_B * self.growth
        )
        self.loader.batch_sampler.batch_size = new_B
        for pg, base in zip(self.opt.param_groups, self.base_lr):
            pg["lr"] = base * new_B
        print(
            f"[BatchScheduler] \u2191 batch {old_B}->{new_B}, LR scaled to {self.opt.param_groups[0]['lr']:.3e}"
        )

    def _maybe_grow(self) -> None:
        if self.max_B and self.loader.batch_sampler.batch_size >= self.max_B:
            return
        batches = list(itertools.islice(self.loader, 2))
        if len(batches) < 2:
            return
        gns = self._grad_noise_scale(*batches)
        if self.step == 0:
            self.smoothed_gns = gns
        else:
            self.smoothed_gns = self.ema * self.smoothed_gns + (1 - self.ema) * gns
        if self.smoothed_gns < self.band_lo * self.tgt:
            self._grow()

    def after_train_step(self, val_loss: Optional[float] = None) -> None:
        self.step += 1
        if self.step % self.check_every == 0:
            self._maybe_grow()
        if val_loss is not None:
            if val_loss + 1e-6 < self.best_val:
                self.best_val = val_loss
                self.bad_evals = 0
            else:
                self.bad_evals += 1
                if self.bad_evals >= self.patience:
                    self.bad_evals = 0
                    self._grow()
