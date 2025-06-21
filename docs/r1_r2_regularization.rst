R1 and R2 Gradient Penalties
===========================

The ``r1_gamma`` and ``r2_gamma`` options apply gradient penalties on the
discriminator at real and fake samples respectively. They are inspired by
the R1 and R2 regularisation techniques from GAN literature which stabilise
training by discouraging sudden changes in the discriminator.

Motivation
----------

When the discriminator learns too quickly it may perfectly separate the
observed outcomes from the generator's counterfactual predictions. This
leads to vanishing gradients for the generator and unstable adversarial
updates. R1 and R2 penalties regularise the discriminator by directly
penalising the norm of its gradients. ``r1_gamma`` measures gradients at
real data points while ``r2_gamma`` measures them at generated data. Both
penalties keep the discriminator smooth and help the generator catch up.

Usage
-----

Enable the penalties by setting ``r1_gamma`` and/or ``r2_gamma`` in
:class:`~crosslearner.training.TrainingConfig`::

   cfg = TrainingConfig(
       epochs=30,
       r1_gamma=0.1,
       r2_gamma=0.1,
   )
   model = train_acx(loader, ModelConfig(p=10), cfg)

During each discriminator step the trainer computes the gradients of the
logits with respect to its inputs. The squared gradient norms are scaled by
``0.5 * r1_gamma`` or ``0.5 * r2_gamma`` and added to the discriminator
loss.

When to use it
--------------

Use R1 and R2 penalties when adversarial training becomes unstable or the
loss oscillates because the discriminator dominates the generator. They can
replace or complement the standard WGAN-GP penalty. Start with small values
around ``0.1`` and adjust based on discriminator smoothness. If training
slows down excessively, reduce the weights or disable the penalties.

References
----------

.. [Mescheder2018] Mescheder, L., Geiger, A., & Nowozin, S. *Which Training
   Methods for GANs Do Actually Converge?* ICML 2018. Analyses gradient
   penalties such as R1 and R2 for stabilising GANs.
