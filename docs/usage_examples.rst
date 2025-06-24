Usage Examples
==============

This page demonstrates common ways to train and evaluate an AC-X model.

Training from the command line
------------------------------

A quick start for experimenting is provided by the ``crosslearner-train``
entry point.  It trains a small model on a synthetic dataset and prints the
final :math:`\sqrt{\mathrm{PEHE}}` metric::

   crosslearner-train

Running benchmarks
------------------

Use ``crosslearner-benchmarks`` to evaluate the implementation on several
built-in datasets.  The following command trains for a single epoch per
dataset and reports the mean score::

   crosslearner-benchmarks all --replicates 1 --epochs 1

Programmatic training
---------------------

The training loop is also accessible from Python.  The snippet below fits a
model on the toy dataset and computes the final PEHE:

.. code-block:: python

   from crosslearner.datasets.toy import get_toy_dataloader
   from crosslearner.training.train_acx import train_acx
   from crosslearner.training import ModelConfig, TrainingConfig
   from crosslearner.evaluation import evaluate
   from crosslearner import set_seed
   import torch

   set_seed(0)
   loader, (mu0, mu1) = get_toy_dataloader()
   X = torch.cat([b[0] for b in loader])

   model_cfg = ModelConfig(p=10)
   train_cfg = TrainingConfig(epochs=30)
   model = train_acx(loader, model_cfg, train_cfg)
   pehe = evaluate(model, X, mu0, mu1)
   print("sqrt(PEHE)", pehe)

Step-by-step notebook
---------------------

For a more interactive introduction open the ``examples/notebook.ipynb``
Jupyter notebook which walks through data loading, model creation and
evaluation step by step.  The notebook links back to :doc:`theory` for
explanations of the underlying objective.

Experiment manager
------------------

For repeated training and hyperparameter optimisation the
``ExperimentManager`` combines cross-validation with Optuna searches and
TensorBoard logging.  See :doc:`hyperparameter_sweeps` for tuning strategies.

Exporting models
----------------

Trained models can be exported to TorchScript or ONNX for deployment.
Use :func:`crosslearner.export.export_model`:

.. code-block:: python

   from crosslearner.models.acx import ACX
   from crosslearner.export import export_model
   import torch

   model = ACX(p=10)
   x = torch.randn(1, 10)
   export_model(model, x, "acx.pt")           # TorchScript
   export_model(model, x, "acx.onnx", onnx=True)

Additional metrics
------------------

``crosslearner`` ships several helpers to analyse the quality of treatment
effect predictions.  After computing ``tau_hat`` you can compute the PEHE
directly, evaluate policy risk and aggregate estimation errors::

   from crosslearner.evaluation import (
       pehe,
       policy_risk,
       ate_error,
       att_error,
       bootstrap_ci,
   )

   sqrt_pehe = pehe(tau_hat, mu1 - mu0)
   risk = policy_risk(tau_hat, mu0, mu1)
   ate_err = ate_error(tau_hat, mu0, mu1)
   att_err = att_error(tau_hat, mu0, mu1, T)
   lower, upper = bootstrap_ci(tau_hat)

Propensity estimation
---------------------

For observational datasets use :func:`crosslearner.evaluation.estimate_propensity`
to obtain cross-fitted treatment probabilities::

   from crosslearner.evaluation import estimate_propensity

   propensity = estimate_propensity(X, T)

