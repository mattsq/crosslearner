Doubly Robust Objective and Propensity Head
==========================================

``crosslearner`` supports an optional propensity head and a doubly robust
objective. Both are controlled through
:class:`~crosslearner.training.TrainingConfig`.

Motivation
----------

The propensity head predicts treatment assignment from the learned
representation. Including a binary cross-entropy term encourages the
network to model treatment probabilities alongside the potential outcomes,
which can stabilise adversarial training. The doubly robust loss further
reduces bias by combining regression and inverse propensity weighting. A
learned ``epsilon`` parameter adjusts the contribution of the weighted
pseudo-outcome.

Usage
-----

Enable these components by setting ``delta_prop`` and ``lambda_dr`` when
constructing ``TrainingConfig``::

   cfg = TrainingConfig(
       epochs=30,
       delta_prop=1.0,
       lambda_dr=0.1,
   )
   model = train_acx(loader, ModelConfig(p=10), cfg)

During training the model minimises ``delta_prop`` times the binary
cross-entropy between predicted propensity scores and observed treatment.
The doubly robust penalty weighted by ``lambda_dr`` compares the outcome to a
bias-corrected estimate using the propensity head and ``epsilon``.

When to use it
--------------

Use ``delta_prop`` when you want the network to explicitly model the
treatment assignment mechanism or when the discriminator struggles due to
covariate imbalance. ``lambda_dr`` is most helpful on smaller datasets or
in the presence of confounding, as it blends outcome regression with
importance weighting. Typical values are between ``0.5`` and ``2`` for
``delta_prop`` and around ``0.1`` for ``lambda_dr``.
