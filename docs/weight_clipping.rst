Weight Clipping for Wasserstein Training
=======================================

``weight_clip`` sets a hard bound on the discriminator's parameters after each
update.  When a positive value is supplied, every weight in the discriminator
is clamped to the interval ``[-weight_clip, weight_clip]`` immediately after an
optimiser step.  This mirrors the original *Wasserstein GAN* training scheme
and keeps the discriminator within a bounded Lipschitz constant.

Motivation
----------

Wasserstein GANs require the discriminator ("critic") to be 1-Lipschitz.  The
simplest way to enforce this constraint is by clipping its weights to a small
range.  Although more advanced techniques like gradient penalties and spectral
normalisation exist, weight clipping remains a lightweight option that can
stabilise adversarial training when other methods are unsuitable or
computationally expensive.

Usage
-----

Specify a positive ``weight_clip`` value in
:class:`~crosslearner.training.TrainingConfig`::

   cfg = TrainingConfig(
       epochs=30,
       weight_clip=0.01,
   )
   model = train_acx(loader, ModelConfig(p=10), cfg)

After each discriminator update the trainer applies
``p.data.clamp_(-weight_clip, weight_clip)`` to every parameter ``p`` in the
adversary.  Setting ``weight_clip`` to ``None`` disables this behaviour.

When to use it
--------------

Weight clipping is most appropriate when experimenting with Wasserstein-style
losses or when gradient penalties slow down training.  It can help avoid
exploding discriminator gradients on small datasets but may limit capacity if
the clip value is too low.  Start with a value around ``0.01`` and adjust
based on training stability.  Avoid combining it with spectral normalisation,
as both aim to bound the discriminator and may over-constrain it.

References
----------

.. [Arjovsky2017] Arjovsky, M., Chintala, S., & Bottou, L. *Wasserstein GAN.*
   ICML 2017. Introduces weight clipping as a method to enforce the
   Lipschitz constraint in Wasserstein GANs.
