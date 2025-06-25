Adaptive Batch Scheduling
=========================

``adaptive_batch`` turns on an experimental scheduler that increases the
``DataLoader`` batch size during training.  The scheduler estimates the
*gradient noise scale* (GNS) every few steps using two mini-batches. When the
noise falls below ``gns_target`` the batch size is multiplied by
``growth_factor`` and the optimiser learning rates are scaled accordingly.

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
   )
   model = train_acx(loader, ModelConfig(p=10), cfg)

When to use it
--------------

Adaptive batching is helpful when manually tuning the batch size is difficult or
when training begins smoothly but later stagnates. The scheduler requires a
``DataLoader`` with a :class:`~crosslearner.utils.MutableBatchSampler`, which is
automatically created by the trainer when ``adaptive_batch`` is set.
