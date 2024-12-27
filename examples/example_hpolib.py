import optuna

from hpo_benchmarks import HPOLib


bench = HPOLib(dataset_name="naval_propulsion")
print(bench)


def objective(trial: optuna.Trial) -> float:
    param_types = bench.param_types
    params = {}
    for param_name, choices in bench.search_space.items():
        if param_types[param_name] == str:
            choice = trial.suggest_categorical(param_name, choices)
            assert choice is not None, "MyPy Redefinition."
        else:
            choice = choices[trial.suggest_int(f"{param_name}_index", low=0, high=len(choices) - 1)]

        params[param_name] = choice

    return bench(params)[bench.metric_names[0]]


study = optuna.create_study(direction=bench.directions[bench.metric_names[0]])
study.optimize(objective, n_trials=30)
