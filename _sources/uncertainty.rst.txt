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

This approximates a Bayesian posterior over the model weights and yields
pointwise credible intervals for the CATE.
