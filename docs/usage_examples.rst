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
   from crosslearner.evaluation import evaluate
   from crosslearner import set_seed
   import torch

   set_seed(0)
   loader, (mu0, mu1) = get_toy_dataloader()
   X = torch.cat([b[0] for b in loader])

   model = train_acx(loader, p=10, epochs=30)
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

