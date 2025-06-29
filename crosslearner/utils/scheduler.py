"""Adaptive batch-size scheduler using the gradient noise scale.

The utilities in this module allow adjusting the ``DataLoader`` batch size
during training.  ``GNSBatchScheduler`` monitors the gradient noise scale (GNS)
and grows the batch once the optimisation noise drops below a threshold.
``MutableBatchSampler`` is a ``BatchSampler`` whose ``batch_size`` attribute can
be changed on-the-fly by the scheduler.
"""

from __future__ import annotations

import itertools
import math
from typing import Iterable, Optional

import torch
from torch.utils.data import BatchSampler, DataLoader

try:  # torch>=1.12 provides a unified autocast
    from torch.amp import autocast
except Exception:  # pragma: no cover - older PyTorch
    from torch.cuda.amp import autocast


class MutableBatchSampler(BatchSampler):
    """Batch sampler whose ``batch_size`` can be changed on the fly.

    ``DataLoader`` objects expose their sampler as read-only, so the usual
    ``batch_size`` argument cannot be updated after construction.  This
    subclass stores the size in an attribute that may be reassigned by a
    scheduler such as :class:`GNSBatchScheduler` to grow the batch during
    training.
    """

    def __init__(
        self, sampler: Iterable[int], batch_size: int, drop_last: bool
    ) -> None:
        super().__init__(sampler, batch_size, drop_last)
        self.batch_size = batch_size


class GNSBatchScheduler:
    """Dynamically increase batch size based on the gradient noise scale.

    The scheduler periodically measures the noise in the optimisation process
    using two neighbouring mini-batches.  When this **gradient noise scale**
    (GNS) falls below a target value, the ``DataLoader``'s batch is grown by a
    ``growth_factor``.  Learning rates are scaled proportionally so that the
    effective step size per sample stays constant.  This allows starting with
    small batches for fast convergence and automatically scaling up when the
    gradients become less noisy.
    """

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
        """Construct a new scheduler.

        Args:
            model: Model whose gradients are inspected.
            loss_fn: Function returning the loss given ``(model, batch)``.
            dataloader: Training loader with a :class:`MutableBatchSampler`.
            optimizer: Optimiser for ``model`` whose learning rate is scaled.
            target_gns: Desired gradient noise scale before increasing the
                batch size.
            band: Multiplicative tolerance around ``target_gns`` for deciding
                when to grow.
            growth_factor: Factor by which to multiply the batch size.
            check_every: Number of steps between GNS evaluations.
            plateau_patience: Number of evaluations without improvement before
                forcing a growth step when ``val_loss`` plateaus.
            ema: Exponential moving average factor for smoothing GNS.
            max_global_batch: Optional cap on the absolute batch size. ``None``
                uses the full dataset size.
        """
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
        self.max_B = (
            max_global_batch
            if max_global_batch is not None
            else len(dataloader.dataset)
        )
        self.smoothed_gns = 0.0
        self.base_lr = [
            g["lr"] / self.loader.batch_sampler.batch_size
            for g in self.opt.param_groups
        ]

    @torch.no_grad()
    def _grad_noise_scale(self, batch1, batch2) -> float:
        """Estimate the gradient noise scale from two mini-batches."""

        grads = []
        device_type = next(self.model.parameters()).device.type
        for b in (batch1, batch2):
            self.opt.zero_grad(set_to_none=True)
            with autocast(device_type=device_type):
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
        """Increase the loader's batch size and scale the optimiser ``lr``."""

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
        """Check the GNS and grow the batch when below the target."""

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
        """Update the scheduler state after each optimisation step."""

        self.step += 1
        if self.max_B and self.loader.batch_sampler.batch_size >= self.max_B:
            return
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
