from __future__ import annotations

import optuna

from hpo_benchmarks import NASBench201


bench = NASBench201(dataset_name="imagenet")


def objective(trial: optuna.Trial) -> float:
    params = {}
    for param_name, choices in bench.search_space.items():
        params[param_name] = trial.suggest_categorical(param_name, choices)

    return bench(params)[bench.metric_names[0]]


study = optuna.create_study(direction=bench.directions[bench.metric_names[0]])
study.optimize(objective, n_trials=30)
