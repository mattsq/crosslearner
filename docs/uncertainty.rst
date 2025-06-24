Uncertainty Estimation
======================

``crosslearner`` provides a simple Monte Carlo dropout utility to quantify
prediction uncertainty. The function
:func:`crosslearner.evaluation.predict_tau_mc_dropout` performs multiple forward
passes with dropout enabled and returns the mean and standard deviation of the
estimated treatment effects.

Example usage::

    from crosslearner.evaluation import predict_tau_mc_dropout

   mean, std = predict_tau_mc_dropout(model, X, passes=50)
   lower = mean - 1.96 * std
   upper = mean + 1.96 * std

Ensembles of independently trained models provide an alternative estimate of
epistemic uncertainty::

    from crosslearner.training import train_acx_ensemble
    from crosslearner.evaluation import predict_tau_ensemble, predict_tau_mc_ensemble

    models = train_acx_ensemble(loader, model_cfg, train_cfg, n_models=5)
    mean, std = predict_tau_ensemble(models, X)

You can also combine the two approaches::

    mean, std = predict_tau_mc_ensemble(models, X, passes=20)

This approximates a Bayesian posterior over the model weights and yields
pointwise credible intervals for the CATE.

References
----------

.. [Gal2016] Gal, Y., & Ghahramani, Z. *Dropout as a Bayesian Approximation:"
   Representing Model Uncertainty in Deep Learning.* ICML 2016. Shows how
   Monte Carlo dropout provides uncertainty estimates.
