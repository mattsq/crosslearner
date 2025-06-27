Adaptive Loss Balancing with GradNorm
====================================

``use_gradnorm`` turns on dynamic weighting of the outcome, consistency and
adversarial losses during training.  The algorithm adjusts three learnable
weights so that the gradient norms of each objective with respect to the shared
encoder match.  This prevents any single loss from dominating optimisation and
removes the need to manually tune ``alpha_out``, ``beta_cons`` and
``gamma_adv``.

Motivation
----------

When multiple objectives compete for the same parameters, one loss can quickly
overpower the others.  GradNorm balances the optimisation by scaling each loss
according to its relative training speed, encouraging all tasks to learn at a
similar rate.

Usage
-----

Enable the feature via :class:`~crosslearner.training.TrainingConfig`::

   cfg = TrainingConfig(
       epochs=30,
       use_gradnorm=True,
   )
   model = train_acx(loader, ModelConfig(p=10), cfg)

``gradnorm_alpha`` controls how aggressively slower objectives are up-weighted
(default ``1.0``) and ``gradnorm_lr`` sets the learning rate for the internal
weight optimiser.

When to use it
--------------

GradNorm is helpful whenever the outcome, consistency and adversarial losses
have very different magnitudes or convergence speeds.  It automatically
rebalances their contributions and often yields more stable training.
