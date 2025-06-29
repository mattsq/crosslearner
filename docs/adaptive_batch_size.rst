Adaptive Batch Scheduling
=========================

``adaptive_batch`` turns on an experimental scheduler that increases the
``DataLoader`` batch size during training.  The scheduler estimates the
*gradient noise scale* (GNS) every few steps using two mini-batches. When the
noise falls below ``gns_target`` the batch size is multiplied by
``gns_growth_factor`` and the optimiser learning rates are scaled accordingly.

Motivation
----------

Early epochs often benefit from small batches that yield noisy but informative
gradients. As optimisation progresses, however, the same noise can slow
convergence. Automatically growing the batch size allows rapid initial learning
while still reaching a stable large-batch regime.

Usage
-----

Enable the scheduler in :class:`~crosslearner.training.TrainingConfig`::

   cfg = TrainingConfig(
       epochs=30,
       adaptive_batch=True,
       gns_target=1.0,
       gns_growth_factor=2,
   )
   model = train_acx(loader, ModelConfig(p=10), cfg)

Additional knobs control how aggressively the batch size grows. ``gns_band``
sets the tolerance around ``gns_target``, ``gns_check_every`` determines how
often the gradient noise scale is measured and ``gns_plateau_patience`` triggers
growth when validation loss stops improving. ``gns_ema`` smooths the noise
estimates and ``gns_max_batch`` caps the final batch size (defaults to the
dataset size). Once this limit is reached, the scheduler no longer measures the
gradient noise.

When to use it
--------------

Adaptive batching is helpful when manually tuning the batch size is difficult or
when training begins smoothly but later stagnates. The scheduler requires a
``DataLoader`` with a :class:`~crosslearner.utils.MutableBatchSampler`, which is
automatically created by the trainer when ``adaptive_batch`` is set.
