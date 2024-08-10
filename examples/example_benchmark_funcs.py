import numpy as np
import optuna

from hpo_benchmarks import Ackley
from hpo_benchmarks import DifferentPower
from hpo_benchmarks import DixonPrice
from hpo_benchmarks import Griewank
from hpo_benchmarks import KTablet
from hpo_benchmarks import Langermann
from hpo_benchmarks import Levy
from hpo_benchmarks import Michalewicz
from hpo_benchmarks import Perm
from hpo_benchmarks import Powell
from hpo_benchmarks import Rastrigin
from hpo_benchmarks import Rosenbrock
from hpo_benchmarks import Schwefel
from hpo_benchmarks import Sphere
from hpo_benchmarks import Styblinski
from hpo_benchmarks import WeightedSphere
from hpo_benchmarks import XinSheYang


for func_cls in [
    Ackley,
    DifferentPower,
    DixonPrice,
    Griewank,
    KTablet,
    Levy,
    Michalewicz,
    Perm,
    Powell,
    Rastrigin,
    Rosenbrock,
    Schwefel,
    Sphere,
    Styblinski,
    WeightedSphere,
    XinSheYang,
    Langermann,
]:
    func = func_cls(dim=2)

    def objective(trial: optuna.Trial) -> float:
        (low, high) = func.param_range
        X = np.asarray([trial.suggest_float(f"x{d}", low=low, high=high) for d in range(func.dim)])
        return func(X)

    print(func)
    study = optuna.create_study()
    study.optimize(objective, n_trials=10)
