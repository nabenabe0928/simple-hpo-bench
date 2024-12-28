from __future__ import annotations

import optuna

from hpo_benchmarks import HPOBench


bench = HPOBench(dataset_name=HPOBench.available_dataset_names[0])
print(bench)


def objective(trial: optuna.Trial) -> list[float]:
    params = {}
    for param_name, choices in bench.search_space.items():
        params[param_name] = choices[trial.suggest_int(f"{param_name}_index", low=0, high=len(choices) - 1)]

    results = bench(params)
    return [results[name] for name in bench.metric_names]


study = optuna.create_study(directions=[bench.directions[name] for name in bench.metric_names])
study.optimize(objective, n_trials=30)
