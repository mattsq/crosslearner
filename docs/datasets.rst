Datasets
========

``crosslearner`` ships with several datasets used for benchmarking and
experimentation. Each dataset is available through a loader function in
:mod:`crosslearner.datasets` that returns a ``DataLoader`` and ground
truth outcomes where applicable.

Available loaders
-----------------

.. autosummary::
   :toctree: generated/datasets

   crosslearner.datasets.get_toy_dataloader
   crosslearner.datasets.get_complex_dataloader
   crosslearner.datasets.get_confounding_dataloader
   crosslearner.datasets.get_ihdp_dataloader
   crosslearner.datasets.get_jobs_dataloader
   crosslearner.datasets.get_acic2016_dataloader
   crosslearner.datasets.get_acic2018_dataloader
   crosslearner.datasets.get_twins_dataloader
   crosslearner.datasets.get_lalonde_dataloader
   crosslearner.datasets.get_aircraft_dataloader

Dataset descriptions
--------------------

The loaders cover both synthetic benchmarks and popular real-world datasets.

``get_toy_dataloader``
  Generates a small synthetic task with 10 covariates and known potential
  outcomes.
``get_complex_dataloader``
  Harder synthetic generator with 20 features and nonlinear treatment effects.
``get_confounding_dataloader``
  Synthetic data where the amount of unobserved confounding can be adjusted.
``get_ihdp_dataloader``
  Semi-synthetic Infant Health and Development Program benchmark with 100
  replications.
``get_jobs_dataloader``
  Jobs training dataset (NSW study) used for off-policy evaluation.
``get_acic2016_dataloader`` / ``get_acic2018_dataloader``
  Load the ACIC challenge benchmarks.  The loaders automatically download the
  ``.npz`` files if they are not present locally.
``get_twins_dataloader``
  Twins dataset of US twin births with fully observed counterfactuals.
``get_lalonde_dataloader``
  Original LaLonde dataset with only the ATE available.
``get_aircraft_dataloader``
  Simulated aircraft performance data based on the Breguet range equation.
