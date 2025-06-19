.. _gan_tricks:

Feature Matching
================

``crosslearner`` provides several ``GAN``-inspired stability tricks.  Feature
matching reduces variance in the adversarial game by making the generator match
the average statistics of an intermediate discriminator layer instead of trying
to fool the discriminator on every sample.  This section describes how feature
matching works, when it is useful and how to use it.

How it works
------------

During each generator update the discriminator computes feature vectors
:math:`f_{\mathrm{real}}` and :math:`f_{\mathrm{fake}}` for real and
counterfactual outcomes.  Instead of maximising the usual adversarial loss, the
model minimises the distance between their mean activations:

.. math::

    \ell_{\mathrm{fm}} = \Bigl\| \operatorname{mean}(f_{\mathrm{real}})
    - \operatorname{mean}(f_{\mathrm{fake}}) \Bigr\|_2^2.

The generator loss becomes
:code:`loss_g += eta_fm * loss_fm`.  This encourages generated counterfactuals to
match the global statistics of real observations, providing smoother gradients
especially on small datasets.

When to use it
--------------

Feature matching helps when the discriminator easily separates real and
counterfactual examples, leading to high variance in the adversarial loss.
Datasets with limited sample sizes or severe covariate imbalance often benefit
from this regulariser.  It is also useful in early training when the outcome
models have not yet converged and the discriminator dominates the game.

Usage
-----

Enable feature matching by setting ``feature_matching`` in
:class:`~crosslearner.training.TrainingConfig`.  The weight of the term is
controlled by ``eta_fm`` which defaults to ``5.0``::

   from crosslearner.training import TrainingConfig

   train_cfg = TrainingConfig(
       feature_matching=True,
       eta_fm=1.0,
   )

Evaluation
----------

The effect of feature matching can be assessed by tracking generator and
discriminator losses as well as validation :math:`\sqrt{\mathrm{PEHE}}`.
Improved stability typically manifests as smoother loss curves and lower
variance across runs.  The :func:`crosslearner.visualization.plot_losses`
utility plots both losses for easy comparison.

