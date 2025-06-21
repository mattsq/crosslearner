Instance Noise for Robust Discriminator
======================================

The ``instance_noise`` option injects Gaussian noise into the discriminator's
input targets early in training. Both the real outcomes ``Y`` and counterfactual
estimates ``Ycf`` receive noise whose standard deviation decays linearly from
``0.2`` to ``0`` over the course of ``epochs``. This prevents the
discriminator from overfitting to exact outcome values and gives the generator
a smoother optimisation landscape.

Motivation
----------

When the discriminator can perfectly separate real from fake samples it may
drastically reduce the generator's gradient signal. Adding small noise to the
outcomes effectively blurs the decision boundary and regularises the
discriminator. This is particularly helpful when datasets are small or noisy,
where the discriminator might otherwise memorise the training data.

Usage
-----

Enable instance noise by setting ``instance_noise=True`` in
:class:`~crosslearner.training.TrainingConfig`::

   cfg = TrainingConfig(
       epochs=30,
       instance_noise=True,
   )
   model = train_acx(loader, ModelConfig(p=10), cfg)

During each discriminator update a noise term is added to the observed outcome
and the counterfactual prediction. The noise magnitude starts at ``0.2`` and
decays linearly to ``0`` by the final epoch.

When to use it
--------------

Use ``instance_noise`` when training is unstable or the discriminator quickly
achieves near-perfect accuracy. It works well together with other
regularisation techniques such as feature matching or gradient reversal. On
very large datasets or when discriminator overfitting is not an issue, the
option can be left disabled.

References
----------

.. [Arjovsky2017] Arjovsky, M., & Bottou, L. *Towards Principled Methods for
   Training Generative Adversarial Networks.* 2017. Introduced instance noise
   as a stabilisation strategy.
