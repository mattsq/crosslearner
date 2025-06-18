import optuna
from crosslearner.datasets.toy import get_toy_dataloader
from crosslearner.experiments import ExperimentManager, cross_validate_acx
from crosslearner.training import ModelConfig, TrainingConfig


def test_cross_validate(tmp_path):
    loader, (mu0, mu1) = get_toy_dataloader(batch_size=4, n=16, p=3)
    model_cfg = ModelConfig(p=3)
    train_cfg = TrainingConfig(epochs=1)
    metric = cross_validate_acx(
        loader,
        mu0,
        mu1,
        model_config=model_cfg,
        training_config=train_cfg,
        folds=2,
        device="cpu",
        log_dir=str(tmp_path),
    )
    assert metric >= 0.0
    assert (tmp_path / "fold_0").exists()


def test_experiment_manager_optuna(tmp_path):
    loader, (mu0, mu1) = get_toy_dataloader(batch_size=4, n=16, p=3)
    manager = ExperimentManager(
        loader, mu0, mu1, p=3, folds=2, device="cpu", log_dir=str(tmp_path)
    )

    def space(trial: optuna.Trial) -> dict:
        return {"epochs": 1, "rep_dim": trial.suggest_int("rep_dim", 4, 8)}

    study = manager.optimize(space, n_trials=1)
    assert len(study.trials) == 1
