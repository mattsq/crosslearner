Warm Start for Stable Optimisation
=================================

The ``warm_start`` option of :class:`~crosslearner.training.TrainingConfig`
pretrains the generator before adversarial updates begin.  When set to a
positive integer only the outcome heads are trained for that many epochs
using the mean squared error on the observed outcomes.  No adversarial or
consistency terms are applied during this phase.

Motivation
----------

GAN objectives can be brittle at the start of training when both the
generator and discriminator are poorly calibrated.  Jumping straight into
the adversarial game might lead to exploding or vanishing gradients.  By
warming up with a few epochs of simple outcome prediction the generator can
learn a reasonable initialisation which stabilises the subsequent
adversarial training.  This is particularly helpful on small or noisy
datasets where the discriminator tends to overpower the generator early on.

Usage
-----

Pass the desired number of warm-up epochs to ``warm_start``::

   cfg = TrainingConfig(
       epochs=30,
       warm_start=5,
   )
   model = train_acx(loader, ModelConfig(p=10), cfg)

During the first five epochs only the outcome reconstruction loss is
optimised.  Starting from epoch six the full objective---including
consistency and adversarial losses---is used.

When to use it
--------------

Enable a warm start when training becomes unstable right from the start or
when the discriminator loss decreases too quickly.  A short warm-up of one
or two epochs often suffices.  Set ``warm_start`` to ``0`` (the default) to
skip this phase if training is already stable.
