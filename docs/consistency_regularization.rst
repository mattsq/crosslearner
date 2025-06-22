Consistency Regularization
==========================

Several optional hyperparameters encourage stable representations and robust predictions
by penalising inconsistency in the encoder and outputs.
These settings are all part of :class:`~crosslearner.training.TrainingConfig`.

Noise Consistency
-----------------

``noise_std`` controls the standard deviation of Gaussian noise added to the
input features during training.  When ``noise_consistency_weight`` is positive,
the model is evaluated on both the clean and noisy inputs and their predictions
are compared with a mean squared error penalty.  The additional loss
``noise_consistency_weight`` \* ``MSE(f(X), f(X+\epsilon))`` encourages the
potential outcome and treatment effect heads to be invariant to small
perturbations.  This acts as a lightweight data augmentation that can improve
robustness.

Example usage::

   cfg = TrainingConfig(
       epochs=30,
       noise_std=0.1,
       noise_consistency_weight=1.0,
   )
   model = train_acx(loader, ModelConfig(p=10), cfg)

Representation Drift
--------------------

``rep_consistency_weight`` introduces a penalty on changes in the encoder's
representation statistics across epochs.  For each treatment group the trainer
tracks an exponential moving average of the mean and variance of the encoded
features.  ``rep_momentum`` sets the decay rate for this moving average
(default ``0.99``).  During training the current batch statistics are compared
to the stored averages and their squared difference is scaled by
``rep_consistency_weight``.

This regularisation discourages large shifts in the latent space, which can
otherwise destabilise adversarial training.  It is especially helpful when the
encoder oscillates or collapses as the discriminator improves.

Example usage::

   cfg = TrainingConfig(
       epochs=30,
       rep_consistency_weight=0.5,
       rep_momentum=0.95,
   )
   model = train_acx(loader, ModelConfig(p=10), cfg)

Epistemic-Aware Consistency
--------------------------

When ``epistemic_consistency`` is enabled the consistency weight is scaled
by the ensemble variance of the treatment effect head. This down-weights the
penalty in regions where the model is uncertain, preventing over-regularisation
on scarcely observed samples. Configure the number of ensemble heads via
``ModelConfig.tau_heads`` (must be at least ``1``).

Example usage::

   model_cfg = ModelConfig(p=10, tau_heads=3)
   cfg = TrainingConfig(
       epochs=30,
       epistemic_consistency=True,
   )
   model = train_acx(loader, model_cfg, cfg)

When to use it
--------------

Enable these penalties when training becomes unstable or when small input
perturbations lead to noticeably different predictions.  Start with small
weights (around ``0.5`` for ``rep_consistency_weight`` and ``0.1`` to ``1.0``
for ``noise_consistency_weight``) and only increase them if the model still
exhibits large variance between epochs.  On very large datasets or when
adversarial training is already stable, these options can be left at their
default value of ``0``.

References
----------

.. [Tarvainen2017] Tarvainen, A., & Valpola, H. *Mean Teachers Are Better Role
   Models: Weight-Averaged Consistency Targets Improve Semi-Supervised Deep
   Learning Results.* NIPS 2017. Explores consistency regularisation via noisy
   inputs.

