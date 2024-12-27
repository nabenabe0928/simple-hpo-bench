from __future__ import annotations

import optuna

from hpo_benchmarks import NASBench201


bench = NASBench201(dataset_name="imagenet")


def objective(trial: optuna.Trial) -> list[float]:
    params = {}
    for param_name, choices in bench.search_space.items():
        choice = trial.suggest_categorical(param_name, choices)
        assert choice is not None, "MyPy Redefinition."
        params[param_name] = choice

    results = bench(params)
    return [results[name] for name in bench.metric_names]

study = optuna.create_study(directions=[bench.directions[name] for name in bench.metric_names])
study.optimize(objective, n_trials=30)
