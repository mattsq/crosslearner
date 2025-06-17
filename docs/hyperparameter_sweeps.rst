Hyperparameter Sweeps with Optuna
================================

`optuna <https://optuna.org/>`_ is a lightweight library for automated
hyperparameter search. It repeatedly samples configurations, trains an AC-X
model and keeps the one with the best validation metric.

The snippet below sweeps a few hyperparameters and minimises the validation
:math:`\sqrt{\mathrm{PEHE}}`.

.. code-block:: python

   import optuna
   import torch
   from crosslearner.datasets.toy import get_toy_dataloader
   from crosslearner.training.train_acx import train_acx
   from crosslearner.training.config import ModelConfig, TrainingConfig
   from crosslearner.evaluation import evaluate

   loader, (mu0, mu1) = get_toy_dataloader()
   X = torch.cat([b[0] for b in loader])
   mu0_all = mu0
   mu1_all = mu1

   def objective(trial):
       model_cfg = ModelConfig(
           p=10,
           rep_dim=trial.suggest_int("rep_dim", 32, 128),
       )
       train_cfg = TrainingConfig(
           lr_g=trial.suggest_loguniform("lr_g", 1e-4, 1e-2),
           lr_d=trial.suggest_loguniform("lr_d", 1e-4, 1e-2),
           beta_cons=trial.suggest_float("beta_cons", 1.0, 20.0),
           epochs=30,
       )
       model = train_acx(loader, model_cfg, train_cfg)
       return evaluate(model, X, mu0_all, mu1_all)

   study = optuna.create_study(direction="minimize")
   study.optimize(objective, n_trials=50)
   print("Best value", study.best_value)
   print("Best params", study.best_params)
