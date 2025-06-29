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
   crosslearner.datasets.get_tricky_dataloader

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
  Jobs training dataset (NSW study) used for off-policy evaluation. The
  covariates and outcome are standardised.
``get_acic2016_dataloader`` / ``get_acic2018_dataloader``
  Load the ACIC challenge benchmarks.  The 2016 dataset is provided by
  ``causallib`` while the 2018 loader downloads the ``.npz`` file if needed.
``get_twins_dataloader``
  Twins dataset of US twin births packaged in ``causaldata`` with fully
  observed counterfactuals.
``get_lalonde_dataloader``
  Original LaLonde dataset with only the ATE available.
``get_aircraft_dataloader``
  Simulated aircraft performance data based on the Breguet range equation.
``get_tricky_dataloader``
  Small imbalanced synthetic dataset designed to highlight discriminator tricks.
