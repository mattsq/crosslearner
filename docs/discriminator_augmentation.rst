Discriminator Feature Augmentation
=================================

The ``disc_aug_prob`` and ``disc_aug_noise`` options provide simple data
augmentation for the discriminator. During each discriminator update the
encoded representation ``h_b`` is optionally corrupted before being fed to the
adversary. ``disc_aug_prob`` applies dropout with the given probability and
``disc_aug_noise`` adds Gaussian noise with the specified standard deviation.
This augmentation is applied equally to real and fake samples so the
discriminator must remain robust to these perturbations.

Motivation
----------

GAN discriminators often learn sharp decision boundaries that quickly
separate the generator's outputs from real data. When this happens the
generator receives vanishing gradients and training stalls. By randomly
dropping features or adding jitter we force the discriminator to rely on
broader patterns rather than exact activations, preventing early
overfitting. These simple perturbations can stabilise adversarial training,
especially on small datasets.

Usage
-----

Enable the augmentation through :class:`~crosslearner.training.TrainingConfig`
by setting either ``disc_aug_prob`` or ``disc_aug_noise`` (or both)::

   cfg = TrainingConfig(
       epochs=30,
       disc_aug_prob=0.2,
       disc_aug_noise=0.05,
   )
   model = train_acx(loader, ModelConfig(p=10), cfg)

With the above configuration each discriminator step will use dropout with
probability ``0.2`` on the representation passed to the discriminator and
add Gaussian noise with standard deviation ``0.05``.

When to use it
--------------

``disc_aug_prob`` and ``disc_aug_noise`` are most helpful when the
discriminator easily distinguishes real from fake samples, leading to
training instability. They act as lightweight regularisers and pair well with
other stabilisation techniques such as feature matching or gradient
reversal. On very large datasets or when the discriminator already trains
slowly, these options can usually be left at ``0``.
