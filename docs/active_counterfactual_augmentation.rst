Active Counterfactual Data Augmentation
======================================

``active_aug_freq`` enables periodic "dreaming" of synthetic samples. During
training the optimiser searches the covariate space for inputs that maximise the
disagreement between the potential outcome heads and the explicit treatment
-effect prediction. These pseudo inputs are labelled using the generator's own
outputs and added to the training loader.

Motivation
----------

Actively generating counterfactual pairs targets regions where the model is most
unsure about the treatment effect. By exploring covariates that lead to large
head disagreement the learner focuses on areas that matter for ranking effects.

Usage
-----

Enable the augmentation by setting ``active_aug_freq`` to a positive epoch
interval along with the search parameters ``active_aug_samples``,
``active_aug_steps`` and ``active_aug_lr``::

   cfg = TrainingConfig(
       epochs=30,
       active_aug_freq=5,
       active_aug_samples=64,
       active_aug_steps=20,
       active_aug_lr=0.05,
   )
   model = train_acx(loader, ModelConfig(p=10), cfg)

Every ``active_aug_freq`` epochs the trainer optimises randomly initialised
covariates for ``active_aug_steps`` steps using gradient ascent on
``|mu1(x) - mu0(x) - tau(x)|``. The resulting samples are appended to the loader
with generator predictions as outcomes. All ``DataLoader`` options such as
``num_workers`` and ``pin_memory`` are preserved when the augmented loader is
constructed.

Model parameters are frozen during this search so no gradients are stored for
the network.

When to use it
--------------

Use active augmentation when treatment-effect ordering on rare covariate
regions is critical. Keep ``active_aug_samples`` small (tens of samples) so the
synthetic data only gently influences training.
