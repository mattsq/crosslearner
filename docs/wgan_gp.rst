Wasserstein Gradient Penalty
===========================

The ``use_wgan_gp`` flag of :class:`~crosslearner.training.TrainingConfig`
replaces the default binary cross-entropy objective with the
*Wasserstein GAN with Gradient Penalty* (WGAN-GP) loss.  When enabled the
trainer minimises the Wasserstein distance between real and generated
outcome pairs while adding a gradient penalty of strength ``lambda_gp``.
This promotes a 1-Lipschitz discriminator and often yields more stable
adversarial training.

Motivation
----------

Standard GAN losses can suffer from vanishing gradients when the
discriminator learns much faster than the generator.  The WGAN-GP loss
addresses this by directly estimating the Wasserstein distance and
regularising the discriminator's gradients on random interpolations
between real and fake samples.  Enforcing a Lipschitz constraint in this
way leads to smoother optimisation and better behaved adversaries
:footcite:`Gulrajani2017`.

Usage
-----

Activate the loss by either passing ``use_wgan_gp=True`` or setting
``adv_loss='wgan-gp'`` when constructing the training configuration::

   cfg = TrainingConfig(
       epochs=50,
       use_wgan_gp=True,
       lambda_gp=10.0,
   )
   model = train_acx(loader, ModelConfig(p=10), cfg)

The gradient penalty weight ``lambda_gp`` determines how strongly the
norm of the discriminator gradients is pushed towards one.  Lower values
behave similarly to ordinary Wasserstein training, while higher values
can prevent the discriminator from overfitting.

Adaptive regularisation
-----------------------

Setting ``adaptive_reg=True`` automatically tunes ``lambda_gp`` based on
the discriminator loss.  After each epoch ``lambda_gp`` is increased by
``reg_factor`` when the loss falls below ``d_reg_lower`` and decreased
when it exceeds ``d_reg_upper``.  The value is clipped to the range
``[lambda_gp_min, lambda_gp_max]`` to avoid excessive penalties.  This
feedback mechanism keeps the discriminator in a healthy loss range
without manual tweaking.

When to use it
--------------

Enable WGAN-GP when binary cross-entropy or hinge losses cause unstable
training or the generator receives weak gradients.  The method is
particularly helpful on small or noisy datasets where the discriminator
quickly dominates.  Combine ``use_wgan_gp`` with ``adaptive_reg`` to
maintain smooth learning without constant monitoring.

References
----------

.. [Gulrajani2017] Gulrajani, I., Ahmed, F., Arjovsky, M., Dumoulin, V., &
   Courville, A. *Improved Training of Wasserstein GANs.* NIPS 2017. Introduces
   the gradient penalty for Wasserstein GANs.
