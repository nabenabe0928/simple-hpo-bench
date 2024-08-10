from __future__ import annotations

import optuna

from hpo_benchmarks import NASBench201


bench = NASBench201(dataset_name="imagenet")


def objective(trial: optuna.Trial) -> float:
    params = {}
    for param_name, choices in bench.search_space.items():
        params[param_name] = trial.suggest_categorical(param_name, choices)

    return bench(params)


study = optuna.create_study(direction=bench.direction)
study.optimize(objective, n_trials=30)
