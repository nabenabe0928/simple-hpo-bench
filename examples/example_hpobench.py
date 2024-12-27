import optuna

from hpo_benchmarks import HPOBench


bench = HPOBench(dataset_name="australian")


def objective(trial: optuna.Trial) -> float:
    params = {}
    for param_name, choices in bench.search_space.items():
        params[param_name] = choices[trial.suggest_int(f"{param_name}_index", low=0, high=len(choices) - 1)]

    return bench(params)[bench.metric_names[0]]


study = optuna.create_study(direction=bench.directions[bench.metric_names[0]])
study.optimize(objective, n_trials=30)
