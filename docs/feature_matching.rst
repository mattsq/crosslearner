Feature Matching for Stable GAN Training
=======================================

The ``feature_matching`` option adds an auxiliary loss that encourages the generator
and discriminator to match internal feature representations. Instead of solely
optimising the adversarial objective, the generator tries to reproduce the mean
feature activations of the discriminator for real data. This can stabilise
training when the discriminator quickly overpowers the generator.

Motivation
----------

GAN objectives tend to be unstable when the discriminator learns faster than the
generator. By penalising the distance between discriminator features of real and
fake samples we provide additional gradients that guide the generator even when
the adversarial loss saturates. This technique has been shown to reduce mode
collapse and is especially useful on small datasets.

Usage
-----

Enable feature matching by setting ``feature_matching=True`` in
:class:`~crosslearner.training.TrainingConfig`. The strength of the penalty is
controlled by ``eta_fm`` which defaults to ``5.0``::

   cfg = TrainingConfig(
       epochs=30,
       feature_matching=True,
       eta_fm=1.0,
   )
   model = train_acx(loader, ModelConfig(p=10), cfg)

During training the mean discriminator features on the current batch of real
samples are compared to those of the generator's fake samples. Their squared
Euclidean difference is multiplied by ``eta_fm`` and added to the generator
loss.

When to use it
--------------

Use feature matching when the discriminator becomes too confident early on or
when adversarial training oscillates. It works well in combination with
spectral normalisation and gradient reversal. Set ``eta_fm`` between ``0.5`` and
``5`` to gently regularise the generator without overpowering the main losses.
If training slows down or the generator fails to learn, decrease ``eta_fm`` or
turn off ``feature_matching``.

References
----------

.. [Salimans2016] Salimans, T., Goodfellow, I., Zaremba, W., Cheung, V., Radford,
   A., & Chen, X. *Improved Techniques for Training GANs.* NIPS 2016. Describes
   feature matching for stabilising adversarial learning.
