Unrolled Discriminator Updates
==============================

``unrolled_steps`` controls how many virtual optimisation steps
are performed on the discriminator before computing the adversarial
loss. Instead of updating the discriminator parameters in a separate
optimizer step, the trainer backpropagates through the discriminator
updates themselves. This makes the generator aware of how the
discriminator will react, leading to more stable adversarial training
when the discriminator learns much faster.

Motivation
----------

Adversarial objectives can oscillate if the discriminator quickly
overfits the current generator output. Unrolling the discriminator
for a few steps computes gradients that anticipate the discriminator's
learning trajectory, effectively performing a short look-ahead.
This can mitigate sudden jumps in the loss landscape and prevent the
discriminator from collapsing to trivial solutions.

Usage
-----

Enable unrolled updates by setting ``unrolled_steps`` in
:class:`~crosslearner.training.TrainingConfig`::

   cfg = TrainingConfig(
       epochs=30,
       unrolled_steps=1,
   )
   model = train_acx(loader, ModelConfig(p=10), cfg)

Setting ``unrolled_steps`` to ``1`` performs one virtual discriminator
update using the current batch, differentiating through that step when
calculating the generator gradients. Values greater than ``3`` rarely
help and slow down training considerably.

When to use it
--------------

Use unrolled updates when the discriminator dominates the generator or
training becomes unstable early on. They are most beneficial on small
or imbalanced datasets where the discriminator can easily separate real
and fake samples. For large datasets or well-balanced problems the
additional computation may not be worth the minor stability gains.
Start with ``unrolled_steps=1`` and disable the feature if training
slows down without improvement.
