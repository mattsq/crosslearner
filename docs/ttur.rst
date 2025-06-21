Two Time-Scale Update Rule (TTUR)
================================

The ``ttur`` flag of :class:`~crosslearner.training.TrainingConfig` enables a
simple variant of the *two time-scale update rule*. After each epoch the
training loop checks the discriminator loss. When the loss drops below ``0.3``
the discriminator is temporarily frozen for the next epoch, allowing the
generator to catch up without competing against an overly strong adversary.

Motivation
----------

GAN optimisation can become unstable when the discriminator learns much faster
than the generator. A dominating discriminator drives its loss close to zero,
which leaves the generator with vanishing gradients. The two time-scale update
rule counteracts this by updating the two networks at different speeds. This
implementation freezes discriminator updates whenever it is already performing
well, effectively reducing its learning rate compared to the generator.

Usage
-----

Activate the scheme by passing ``ttur=True`` when constructing the training
configuration::

    train_cfg = TrainingConfig(
        epochs=50,
        ttur=True,
    )

When to use it
--------------

Enable ``ttur`` if you observe that the discriminator quickly achieves near-zero
loss and the generator fails to improve. Freezing the discriminator for short
periods gives the generator breathing room and can stabilise training on small
or noisy datasets. Leave ``ttur`` disabled when both networks converge smoothly
or when using alternative techniques such as gradient penalties to balance their
learning speeds.
