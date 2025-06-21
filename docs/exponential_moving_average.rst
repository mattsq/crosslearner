Exponential Moving Average of Generator Parameters
=================================================

The ``ema_decay`` option creates a slowly-updated shadow copy of the
generator network. After every optimisation step each non-discriminator
parameter is blended with its previous value using
``ema_param = ema_decay * ema_param + (1 - ema_decay) * param``.
The resulting exponential moving average (EMA) often yields smoother
predictions than the raw, rapidly changing weights.

Motivation
----------

GAN training can be noisy: generator weights fluctuate as they react to
the discriminator. Averaging the parameters over time reduces this
variance and can improve generalisation. It also provides a stable model
for computing validation metrics. The technique is lightweight and
requires only a single extra model copy.

Usage
-----

Specify ``ema_decay`` in :class:`~crosslearner.training.TrainingConfig`
with a value between ``0`` and ``1`` (e.g. ``0.999``)::

   cfg = TrainingConfig(
       epochs=30,
       ema_decay=0.99,
   )
   model = train_acx(loader, ModelConfig(p=10), cfg)

During training the trainer maintains an EMA model whose parameters are
updated after each generator step. This copy is used for validation and
is returned when training finishes.

When to use it
--------------

Enable the EMA when you observe unstable validation performance or want
more reliable treatment effect estimates. Smaller decay values (around
``0.9``) adapt faster but track the current weights closely, while larger
values (``0.99`` or ``0.999``) provide stronger smoothing. If memory is
limited or you do not need extra stability set ``ema_decay`` to ``None``
(the default) to disable the feature.

References
----------

.. [Polyak1992] Polyak, B., & Juditsky, A. *Acceleration of stochastic
   approximation by averaging.* SIAM Journal on Control and Optimization,
   1992. Introduces the idea of averaging iterates for variance reduction.

