Relativistic GAN Loss
=====================

The ``adv_loss='rgan'`` option replaces the standard discriminator objective with
*Relativistic Average GAN* losses. Instead of judging real and fake samples in
isolation, the discriminator compares them directly. This often yields more
stable gradients and encourages diverse generations.

Usage
-----

Enable the loss via :class:`~crosslearner.training.TrainingConfig`::

   cfg = TrainingConfig(
       epochs=30,
       adv_loss='rgan',
       use_wgan_gp=True,       # optional gradient penalty
   )
   model = train_acx(loader, ModelConfig(p=10), cfg)

When ``use_wgan_gp`` is ``True`` a gradient penalty of strength
``lambda_gp`` is applied to the relativistic discriminator. The generator
then maximises the probability that fake outcomes appear more realistic
than real ones.

When to use it
--------------

Relativistic adversaries can stabilise training when ordinary
cross-entropy saturates. Try this option if the discriminator quickly
overpowers the generator or if you observe mode collapse with the
standard loss.
