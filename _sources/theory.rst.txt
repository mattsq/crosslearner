Theoretical Background
======================

AC-X builds on two key ideas in modern causal inference: the X-learner
meta-approach and the DragonNet architecture.  The X-learner imputes
counterfactual outcomes from separate models for the treated and
control groups, producing pseudo-outcomes that can be used to learn a
final treatment effect estimator.  DragonNet in contrast trains a
single neural network with shared representations and an auxiliary
propensity head.  Its objective ties the potential outcome heads
together through regularisation based on treatment probability.

AC-X combines these perspectives.  The network resembles a simplified
DragonNet with explicit potential outcome heads and a separate
treatment-effect head.  It still follows the X-learner philosophy of
borrowing the opposite arm’s prediction as a pseudo-outcome, but all
components are trained jointly so that gradients from the
consistency and adversarial terms update both outcome heads and the
shared representation.

Benefits of Adversarial Training
--------------------------------

Enforcing an adversarial game between a generator (the potential
outcome models) and a discriminator helps the model learn
counterfactuals that are indistinguishable from real observations.
This adversarial signal provides useful gradients even when overlap is
limited, acting as an implicit regulariser and reducing bias from
covariate imbalance.  Empirically, adversarial training tends to lower
root PEHE and encourages smoother treatment-effect surfaces.

AC-X Training Procedure
-----------------------

The following pseudocode sketches the training loop implemented in
:mod:`crosslearner.training.train_acx`.

.. code-block:: text

   initialise AC-X model and optimisers
   for each epoch:
       for each batch (X, T, Y):
           # update discriminator
           h, m0, m1, _ = model(X)
           Y_cf = where(T == 1, m0, m1)
           real_logits = discriminator(h, Y, T)
           fake_logits = discriminator(h, Y_cf, T)
           loss_d = adv_loss(real_logits, fake_logits)
           backprop and update discriminator

           # update generator
           h, m0, m1, tau = model(X)
           m_obs = where(T == 1, m1, m0)
           loss_y = mse(m_obs, Y)
           loss_cons = mse(tau, m1 - m0)
           Y_cf = where(T == 1, m0, m1)
           fake_logits = discriminator(h, Y_cf, T)
           loss_adv = generator_adv_loss(fake_logits)
           loss_g = alpha_out * loss_y + beta_cons * loss_cons + gamma_adv * loss_adv
           backprop and update generator

   return trained model

