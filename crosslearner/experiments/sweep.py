import argparse
import os
from typing import Callable, Dict, Any, Optional

import optuna
import yaml

from .manager import ExperimentManager
from ..benchmarks.run_benchmarks import DATASET_LOADERS

_DEFAULT_SPACE = os.path.join(os.path.dirname(__file__), "../configs/optuna_sweep.yaml")


def _load_space(path: str) -> Callable[[optuna.Trial], Dict[str, Any]]:
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}

    def space(trial: optuna.Trial) -> Dict[str, Any]:
        params: Dict[str, Any] = {}
        for name, spec in cfg.items():
            typ = spec.get("type", "float")
            if typ == "int":
                params[name] = trial.suggest_int(name, spec["low"], spec["high"])
            elif typ == "float":
                params[name] = trial.suggest_float(name, spec["low"], spec["high"])
            elif typ == "loguniform":
                params[name] = trial.suggest_float(
                    name, spec["low"], spec["high"], log=True
                )
            elif typ == "choice":
                params[name] = trial.suggest_categorical(name, spec["values"])
            elif typ == "bool":
                params[name] = trial.suggest_categorical(name, [True, False])
            else:  # pragma: no cover - unexpected
                raise ValueError(f"unknown parameter type {typ}")
        return params

    return space


def run_sweep(
    dataset: str,
    *,
    n_trials: int = 50,
    sampler: str | optuna.samplers.BaseSampler = "tpe",
    space_config: Optional[str] = None,
) -> optuna.Study:
    """Run an Optuna sweep for ``dataset``.

    Args:
        dataset: Name of dataset as understood by ``crosslearner-benchmarks``.
        n_trials: Number of trials to evaluate.
        sampler: Sampler name (``tpe`` or ``random``) or a sampler instance.
        space_config: Optional path to YAML configuration for the search space.
    """
    if dataset not in DATASET_LOADERS:
        raise ValueError(f"unknown dataset {dataset}")

    loader_fn = DATASET_LOADERS[dataset]
    loader, (mu0, mu1) = loader_fn(0)
    p = loader.dataset.tensors[0].size(1)
    manager = ExperimentManager(loader, mu0, mu1, p=p, folds=3, device="cpu")

    space_fn = _load_space(space_config or _DEFAULT_SPACE)

    if isinstance(sampler, str):
        name = sampler.lower()
        if name == "tpe":
            sampler_obj = optuna.samplers.TPESampler()
        elif name == "random":
            sampler_obj = optuna.samplers.RandomSampler()
        else:  # pragma: no cover - unexpected
            raise ValueError(f"unknown sampler {sampler}")
    else:
        sampler_obj = sampler

    return manager.optimize(space_fn, n_trials=n_trials, sampler=sampler_obj)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an Optuna sweep for ACX")
    parser.add_argument(
        "dataset",
        choices=list(DATASET_LOADERS.keys()),
        help="dataset to run the sweep on",
    )
    parser.add_argument("--trials", type=int, default=50, help="number of trials")
    parser.add_argument(
        "--sampler",
        default="tpe",
        help="Optuna sampler to use (tpe|random)",
    )
    parser.add_argument(
        "--config",
        help="YAML file specifying search space",
    )
    args = parser.parse_args()
    run_sweep(
        args.dataset,
        n_trials=args.trials,
        sampler=args.sampler,
        space_config=args.config,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
