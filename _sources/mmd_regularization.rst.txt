MMD Regularization for Balanced Representations
===============================================

The ``mmd_weight`` and ``mmd_sigma`` options introduce a kernel-based
regularisation term. During each generator update the encoder outputs for
treated and control units are compared using the **maximum mean discrepancy**
(MMD) with a radial basis function (RBF) kernel. The squared MMD is multiplied
by ``mmd_weight`` and added to the generator loss.

Motivation
----------

Adversarial consistency alone may not fully align the representation
across treatment groups. If the discriminator easily separates real and
counterfactual outcomes, the generator can still learn distinct feature
distributions for treated and control samples. The MMD penalty explicitly
drives the mean embeddings of both groups together, encouraging an
invariant representation that improves effect estimation on imbalanced
datasets.

Usage
-----

Activate the penalty by setting ``mmd_weight`` to a positive value in
:class:`~crosslearner.training.TrainingConfig`::

   cfg = TrainingConfig(
       epochs=30,
       mmd_weight=0.5,
       mmd_sigma=2.0,
   )
   model = train_acx(loader, ModelConfig(p=10), cfg)

``mmd_sigma`` controls the bandwidth of the RBF kernel. Start with a
value similar to the standard deviation of the encoder output (often
between ``1`` and ``3``).

When to use it
--------------

Use MMD regularisation when treatment groups have markedly different
covariate distributions or when the discriminator quickly overpowers the
generator. It pairs well with feature matching and discriminator
augmentation. Large ``mmd_weight`` values can slow convergence, so begin
with a small weight such as ``0.1`` and increase only if the
representations remain divergent.

References
----------

.. [Gretton2012] Gretton, A., Borgwardt, K., Rasch, M., Schölkopf, B., & Smola,
   A. *A Kernel Two-Sample Test.* JMLR 2012. Provides the maximum mean
   discrepancy statistic used for this regulariser.
