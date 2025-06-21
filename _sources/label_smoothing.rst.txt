Label Smoothing for the Discriminator
====================================

The ``label_smoothing`` option replaces the discriminator's hard targets
``0`` and ``1`` with softened labels ``0.1`` and ``0.9`` when using the
binary cross-entropy adversarial loss. This one-sided smoothing keeps the
adversary from becoming overconfident early in training.

Motivation
----------

Overly confident discriminators can drive the generator's gradients towards
zero. Softening the targets reduces this effect and encourages more stable
learning. Label smoothing was introduced for classification tasks in
:footcite:`Szegedy2016` and has since proven effective in GAN setups.

Usage
-----

Enable label smoothing via :class:`~crosslearner.training.TrainingConfig`::

   cfg = TrainingConfig(
       epochs=30,
       label_smoothing=True,
   )
   model = train_acx(loader, ModelConfig(p=10), cfg)

When ``adv_loss`` is ``"bce"`` (the default), the discriminator learns from
``0.9`` and ``0.1`` targets instead of perfect ``1`` or ``0``. Other loss
choices ignore this option.

When to use it
--------------

Activate ``label_smoothing`` if the discriminator quickly dominates and the
generator struggles to improve. On larger datasets or with hinge or
Wasserstein losses the effect is smaller and the option can remain ``False``.

References
----------

.. [Szegedy2016] Szegedy, C., Vanhoucke, V., Ioffe, S., Shlens, J., &
   Wojna, Z. *Rethinking the Inception Architecture for Computer Vision.*
   CVPR 2016. Proposes label smoothing as a regularisation technique.
