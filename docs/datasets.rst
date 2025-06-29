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
   crosslearner.datasets.get_cps_mixtape_dataloader
   crosslearner.datasets.get_thornton_hiv_dataloader
   crosslearner.datasets.get_nhefs_dataloader
   crosslearner.datasets.get_social_insure_dataloader
   crosslearner.datasets.get_credit_cards_dataloader
   crosslearner.datasets.get_close_elections_dataloader
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
``get_cps_mixtape_dataloader``
  Observational CPS data mirroring the NSW job-training study.
``get_thornton_hiv_dataloader``
  HIV cash-incentive RCT with binary test collection outcome.
``get_nhefs_dataloader``
  Health follow-up study evaluating the effect of quitting smoking.
``get_social_insure_dataloader``
  Network experiment dataset examining insurance uptake.
``get_credit_cards_dataloader``
  Large-scale credit card delinquency records for temporal causal tests.
``get_close_elections_dataloader``
  Panel of close US House elections for regression-discontinuity analyses.
``get_aircraft_dataloader``
  Simulated aircraft performance data based on the Breguet range equation.
``get_tricky_dataloader``
  Small imbalanced synthetic dataset designed to highlight discriminator tricks.

Real datasets from ``causaldata``
-------------------------------

The `causaldata` package exposes several tabular datasets that pair a clear
treatment indicator with a suitable outcome. They make convenient benchmarks for
X-learners.  The table below summarises the most relevant options.

.. list-table:: Causaldata quick reference
   :header-rows: 1
   :widths: 15 6 28 22 14 40

   * - Dataset
     - N
     - Treatment variable(s)
     - Outcome of interest
     - Balance (T:C)
     - Notes on suitability
   * - ``nsw_mixtape``
     - 445
     - ``treat`` (job-training)
     - ``re78`` earnings (continuous)
     - 185 : 260 (≈42 % treated)
     - Classic LaLonde RCT; moderate covariate set; clean ground-truth ATE lets you check if X-Learner can recover experimental TE
   * - ``cps_mixtape``
     - 15 992
     - ``treat`` (self-selected into NSW programme)
     - ``re78`` earnings
     - 297 : 15 695 (≈2 % treated)
     - Real-world selection bias and extreme class-imbalance—exactly the regime where X-Learner is theoretically strongest
   * - ``thornton_hiv``
     - 4 820
     - ``any`` (received any cash incentive) or multi-level ``tinc``
     - ``got`` (collected test result, binary) or ``hiv2004`` (status)
     - 1 940 : 2 880 (≈40 % treated)
     - Large RCT with simple structure; easy to binarise incentive; nice for heterogeneous-effect diagnostics
   * - ``nhefs`` / ``nhefs_complete``
     - 1 629 (complete)
     - ``qsmk`` (quit smoking 1971-82)
     - ``wt82_71`` weight change (continuous)
     - 248 : 1 381 (≈15 % treated)
     - Observational health data with many confounders; good stress-test for propensity modelling in Stage 1
   * - ``social_insure``
     - 1 410
     - ``any`` (received any info/peer intervention)
     - ``takeup_survey`` etc. (binary)
     - ~50 % treated
     - Network-experiment; covariates include village-level and demographics—useful for testing effect heterogeneity in sparse, mid-sized data
   * - ``credit_cards``
     - 30 000
     - Use ``LateApril`` (late payment in Apr 2005) as “treatment” for predicting ``LateSept``
     - ``LateSept`` (binary)
     - 5 756 : 24 244 (≈19 % treated)
     - A temporal-ordering causal proxy; huge, purely tabular; handy for scalability & calibration checks
   * - ``close_elections_lmb``
     - 13 588
     - ``dwin`` (Democrat wins)
     - policy outcomes
     - ≈50 %
     - Regression-discontinuity panel—valuable if you want to see how X-Learner behaves when RD assumptions are ignored, but requires careful feature engineering

Recommended test set
~~~~~~~~~~~~~~~~~~~~

#. Fast smoke test: ``nsw_mixtape`` – small, clean binary treatment, continuous outcome.
#. Imbalance stress test: ``cps_mixtape`` – same variables as NSW but extreme 2 % treated share.
#. Larger RCT: ``thornton_hiv`` – lets you check variance-reduction and subgroup CATE accuracy.
#. Observational with rich confounding: ``nhefs_complete`` – evaluates full causal pipeline.
#. Add ``social_insure`` for mid-sized network context, and ``credit_cards`` for high-volume tabular evaluation.
