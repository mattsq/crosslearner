Fitting AC-X Models
===================

This page provides a high level guide for training AC-X models. It covers
selecting sensible hyperparameters, diagnosing common training issues and
interpreting model outputs.

Selecting hyperparameters
-------------------------

``crosslearner`` exposes two configuration dataclasses:
:class:`~crosslearner.training.ModelConfig` controls the architecture while
:class:`~crosslearner.training.TrainingConfig` handles optimisation. The
following guidelines help choose good defaults.

* **Representation size** ``rep_dim`` – start small (``32`` or ``64``) for toy
  datasets and increase to ``128`` or larger as the number of covariates grows.
* **Hidden layers** ``phi_layers`` and ``head_layers`` – one or two layers with
  ``relu`` activations often suffice. Add depth only when the dataset is
  complex or highly nonlinear.
* **Dropout and residual connections** – use dropout on small datasets to
  prevent overfitting. Residual connections can stabilise deep networks.
* **Learning rates** ``lr_g`` and ``lr_d`` – the generator and discriminator
  often use similar rates around ``1e-3``. If the discriminator overwhelms the
  generator, lower ``lr_d`` or enable :doc:`ttur`.
* **Regularisation weights** – penalties such as ``lambda_gp`` and
  ``r1_gamma`` help when the discriminator loss diverges. Increase them
  gradually until training stabilises.

For a systematic search refer to :doc:`hyperparameter_sweeps` which integrates
``optuna`` for automated tuning.

Diagnosing training pathologies
-------------------------------

Monitoring losses and metrics during training is essential. Enable
``return_history=True`` in the training configuration to obtain a list of loss
values per epoch. :doc:`training_history` details how to visualise this data.

Typical symptoms and fixes include:

* **Discriminator loss near zero** – the discriminator dominates. Reduce its
  learning rate, add dropout or increase the gradient penalty.
* **Generator loss stagnates** – try higher ``lr_g`` or apply
  :doc:`warm_start` so adversarial objectives kick in later.
* **Noisy metrics** – log to TensorBoard via ``tensorboard_logdir`` and smooth
  the curves. Increase batch size if possible.
* **Divergence** – check that inputs are normalised and consider enabling
  :doc:`spectral_norm` or :doc:`instance_noise`.

Generating and interpreting results
-----------------------------------

After training, pass a tensor of covariates ``X`` through the model to obtain
three outputs ``(mu0_hat, mu1_hat, tau_hat)``. The treatment effect
``tau_hat`` can be evaluated with :func:`crosslearner.evaluation.evaluate`
when ground truth counterfactuals are known. For observational data use
:func:`crosslearner.evaluation.evaluate_ipw` or
:func:`crosslearner.evaluation.evaluate_dr`.

The outcome heads ``mu0`` and ``mu1`` predict potential outcomes under control
and treatment. Subtracting them also yields the CATE. The estimates can be
exported to TorchScript or ONNX using :func:`crosslearner.export.export_model`
for deployment or further analysis.

Interpreting the effect estimates in context requires domain knowledge.
Inspect distributions of ``tau_hat`` across subgroups and compare against known
benchmarks or randomised data if available.

