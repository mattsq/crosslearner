Effect Head Bias
================

``tau_bias`` controls whether the effect heads learn a separate bias term.
When set to ``False`` all biases are fixed at ``0`` so the model's treatment

effect predictions rely purely on the learned features. This can reduce
run-to-run variance in small data regimes where a constantly shifting offset
can destabilise training.

Usage
-----

Set ``tau_bias=False`` in :class:`~crosslearner.training.ModelConfig`::

   model_cfg = ModelConfig(p=10, tau_bias=False)
   model = ACX(
       model_cfg.p,
       tau_bias=model_cfg.tau_bias,
   )

Alternatively, create a :class:`~crosslearner.models.ACX` instance directly::

   model = ACX(p=10, tau_bias=False)

When to use it
--------------

Disable the bias when the effect head tends to learn a large constant offset
across different runs. Fixing the bias to ``0`` forces the model to learn the
scaling from the covariates instead of relying on a free parameter.
