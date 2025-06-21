Representation Disentanglement and Auxiliary Adversaries
=======================================================

The ``disentangle`` flag of :class:`crosslearner.models.ACX` and
:class:`~crosslearner.training.ModelConfig` splits the encoder output into
three parts:

``rep_dim_c``
    Dimensionality of the confounder representation ``z_c``.
``rep_dim_a``
    Dimensionality of the outcome-specific representation ``z_a``.
``rep_dim_i``
    Dimensionality of the instrument representation ``z_i``.

When ``disentangle=True`` the model routes these feature blocks to different
heads. Two optional adversaries try to predict treatment and outcome from
combinations of the blocks:

``adv_t_weight``
    Weight for a logistic adversary predicting treatment from
    ``(z_c, z_a)``.
``adv_y_weight``
    Weight for a regression adversary predicting the observed outcome from
    ``(z_c, z_i)``.

The adversaries are trained alongside the discriminator. Their gradients are
reversed when backpropagated through ``phi`` so the encoder learns features that
do not reveal treatment or outcome information beyond the designated blocks.

Motivation
----------

Real-world datasets often contain strong confounders or instruments that can
leak into every part of the representation. By disentangling the encoder and
penalising treatment or outcome predictability, the model is encouraged to store
relevant information only where appropriate. This can reduce bias and improve
counterfactual estimates when the independence assumptions of causal inference
are violated.

Usage
-----

Enable disentanglement by setting ``disentangle=True`` in ``ModelConfig`` (or the
``ACX`` constructor) and provide sizes for all three representation parts. Set
``adv_t_weight`` and ``adv_y_weight`` to positive values to activate the
adversaries::

   from crosslearner.training import ModelConfig, TrainingConfig, train_acx

   model_cfg = ModelConfig(
       p=10,
       disentangle=True,
       rep_dim_c=32,
       rep_dim_a=16,
       rep_dim_i=16,
   )

   train_cfg = TrainingConfig(
       epochs=40,
       adv_t_weight=1.0,
       adv_y_weight=0.5,
   )

   model = train_acx(loader, model_cfg, train_cfg)

When to use it
--------------

Representation disentanglement is most beneficial when the treatment assignment
or outcomes are strongly correlated with nuisance factors that the model should
ignore. Start with small adversary weights (e.g. ``0.1`` to ``1.0``) and
increase them if the encoder continues to leak information. On very small
datasets or when causal assumptions hold perfectly, disabling this feature keeps
the architecture simpler.

References
----------

.. [Locatello2019] Locatello, F., Bauer, S., Lucic, M., Gelly, S., Bachem, O.,
   & Schölkopf, B. *Challenging Common Assumptions in the Unsupervised Learning
   of Disentangled Representations.* ICML 2019. Discusses limitations and
   benefits of disentangled representations.
