Contrastive Representation Loss
==============================

The ``contrastive_weight`` parameter introduces a triplet margin loss on the
encoder representations. By drawing a positive example from the current batch and
a negative example from the opposite treatment group, the model learns to keep
representations of different treatments apart while clustering similar
observations.

Motivation
----------

Adversarial objectives alone may fail to perfectly align the covariate
distributions of treated and control groups. A small contrastive penalty can
encourage the encoder to produce features that separate dissimilar units and
reinforce treatment balancing. This is especially useful when the discriminator
quickly overfits or when mini-batches contain few samples from one treatment
class.

Usage
-----

Enable the loss by setting ``contrastive_weight`` to a positive value in
:class:`~crosslearner.training.TrainingConfig`. ``contrastive_margin`` controls
how far apart negatives must be, while ``contrastive_noise`` adds optional
Gaussian noise to the positive samples::

   cfg = TrainingConfig(
       epochs=30,
       contrastive_weight=1.0,
       contrastive_margin=0.5,
       contrastive_noise=0.01,
   )
   model = train_acx(loader, ModelConfig(p=10), cfg)

During training the encoder output ``h`` for each batch is treated as the anchor
vector. A second pass with optional noise yields the positive example ``h_pos``.
The negative example ``h_neg`` is sampled from the opposite treatment group. The
triplet margin loss ``triplet_margin_loss(h, h_pos, h_neg, margin)`` is then
scaled by ``contrastive_weight`` and added to the generator objective.

When to use it
--------------

Use contrastive regularisation when covariate imbalance remains after
adversarial training or when you observe unstable discriminator behaviour. A
small ``contrastive_weight`` (between ``0.1`` and ``1.0``) often suffices. If
the loss dominates training or decreases convergence speed, reduce the weight or
set it to zero. Adding a little ``contrastive_noise`` (e.g. ``0.01``) can
prevent collapse by ensuring positives are not identical to the anchor.

References
----------

.. [Hadsell2006] Hadsell, R., Chopra, S., & LeCun, Y. *Dimensionality Reduction
   by Learning an Invariant Mapping.* CVPR 2006. Early work on contrastive and
   triplet losses for representation learning.
