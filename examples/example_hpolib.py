import optuna

from hpo_benchmarks import HPOLib


bench = HPOLib(dataset_name="naval_propulsion")
print(bench)


def objective(trial: optuna.Trial) -> float:
    param_types = bench.param_types
    params = {}
    for param_name, choices in bench.search_space.items():
        if param_types[param_name] == str:
            params[param_name] = trial.suggest_categorical(param_name, choices)
        else:
            params[param_name] = choices[trial.suggest_int(f"{param_name}_index", low=0, high=len(choices) - 1)]

    return bench(params)


study = optuna.create_study(direction=bench.direction)
study.optimize(objective, n_trials=30)
