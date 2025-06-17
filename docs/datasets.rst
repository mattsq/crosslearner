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
