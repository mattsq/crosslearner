Orthogonal Risk for Early Stopping
=================================

Training AC-X models often relies on validation PEHE computed from known potential outcomes.
However, real datasets rarely provide counterfactual outcomes.  The ``risk_data`` option of
:class:`~crosslearner.training.TrainingConfig` allows early stopping without ground truth by
minimising an **orthogonal risk** estimated on a held-out triplet ``(X, T, Y)``.

When ``risk_data`` is supplied the trainer first fits cross-fitted propensity and outcome
models on the provided data using :func:`~crosslearner.training.nuisance.estimate_nuisances`.
These nuisance estimates yield an orthogonal loss

.. math::
   R = \mathbb{E}\bigl[(Y - \mu_{T}(X) - (T - e(X))\,\tau(X))^2\bigr],

where :math:`\tau(X)` is the predicted treatment effect and :math:`e(X)` and
:math:`\mu_{T}(X)` are the learned nuisances.  Lower values indicate better causal
estimates and the training loop stops if the risk fails to improve for
``patience`` epochs.

Example usage
-------------

.. code-block:: python

   from crosslearner.datasets.toy import get_toy_dataloader
   from crosslearner.training import ModelConfig, TrainingConfig
   from crosslearner.training.train_acx import train_acx
   import torch

   loader, _ = get_toy_dataloader()
   X = torch.cat([b[0] for b in loader])
   T = torch.cat([b[1] for b in loader])
   Y = torch.cat([b[2] for b in loader])

   model_cfg = ModelConfig(p=10)
   train_cfg = TrainingConfig(
       epochs=50,
       risk_data=(X, T, Y),
       risk_folds=3,
       patience=5,
       early_stop_metric="risk",
   )
   model = train_acx(loader, model_cfg, train_cfg)

Tips
----

* Increase ``risk_folds`` for more accurate cross-fitting on larger datasets.
* ``nuisance_propensity_epochs`` and ``nuisance_outcome_epochs`` control the
  training length of the nuisance models.
* Combine ``risk_data`` with ``tensorboard_logdir`` to plot the risk over time.
* Select ``early_stop_metric="risk"`` to explicitly monitor the orthogonal risk
  instead of validation PEHE when both metrics are available.
* If you have ground-truth potential outcomes prefer ``val_data`` to monitor
  PEHE directly.

References
----------

.. [Foster2019] Foster, D., & Syrgkanis, V. *Orthogonal Statistical Learning.*
   2019. Provides the orthogonal risk used for early stopping.
