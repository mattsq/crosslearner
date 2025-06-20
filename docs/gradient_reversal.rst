Gradient Reversal for Adversarial Training
=========================================

The ``gradient_reversal`` option offers an alternative way to train the discriminator.
Instead of alternating optimisation steps for the generator and discriminator,
the model can use a gradient reversal layer (GRL) so that the discriminator's loss
flows through a fixed feature extractor while its gradients are multiplied by
``-grl_weight`` before reaching the encoder.
This yields an adversarial signal without a separate discriminator update step.

Motivation
----------

Alternating generator and discriminator updates can be unstable, especially
on small datasets where the discriminator quickly overfits. By attaching a GRL
between the encoder and discriminator we force the representation network to
produce features that confuse the discriminator while still training the
classifier in a single backward pass.
This typically leads to simpler code, fewer optimisation hyperparameters and more
stable training in the early stages.

Usage
-----

Enable gradient reversal by passing ``gradient_reversal=True`` to
:class:`~crosslearner.training.TrainingConfig` and optionally adjust the
strength of the adversarial signal with ``grl_weight``::

   cfg = TrainingConfig(
       epochs=30,
       gradient_reversal=True,
       grl_weight=0.5,
   )

   model = train_acx(loader, ModelConfig(p=10), cfg)

This will bypass explicit discriminator updates and instead update the
discriminator together with the generator. During backpropagation the GRL
multiplies the gradients from the discriminator by ``-grl_weight``.

When to use it
--------------

Gradient reversal works best when the discriminator tends to overpower the
generator or when computational simplicity is preferred. It is particularly
useful for quick experiments or low-resource settings where performing
multiple discriminator steps per batch might be costly.

However, for larger datasets or when fine-grained control over adversarial
optimisation is required, disabling ``gradient_reversal`` and using separate
``disc_steps`` may yield better performance.

