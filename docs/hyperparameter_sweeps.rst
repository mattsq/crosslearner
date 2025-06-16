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
   from crosslearner.evaluation import evaluate

   loader, (mu0, mu1) = get_toy_dataloader()
   X = torch.cat([b[0] for b in loader])
   mu0_all = mu0
   mu1_all = mu1

   def objective(trial):
       rep_dim = trial.suggest_int("rep_dim", 32, 128)
       lr_g = trial.suggest_loguniform("lr_g", 1e-4, 1e-2)
       lr_d = trial.suggest_loguniform("lr_d", 1e-4, 1e-2)
       beta_cons = trial.suggest_float("beta_cons", 1.0, 20.0)

       model = train_acx(
           loader,
           p=10,
           rep_dim=rep_dim,
           lr_g=lr_g,
           lr_d=lr_d,
           beta_cons=beta_cons,
           epochs=30,
           device="cpu",
       )
       return evaluate(model, X, mu0_all, mu1_all)

   study = optuna.create_study(direction="minimize")
   study.optimize(objective, n_trials=50)
   print("Best value", study.best_value)
   print("Best params", study.best_params)
