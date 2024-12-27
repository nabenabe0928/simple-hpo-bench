from __future__ import annotations

import itertools

import optuna
import pytest

from hpo_benchmarks import HPOBench
from hpo_benchmarks import HPOLib
from hpo_benchmarks import NASBench201
from hpo_benchmarks.base import BaseHPOBench


def _get_metric_choices(metric_names: list[str]) -> list[None | list[str]]:
    return [None] + list(
        itertools.chain(
            *[[list(it) for it in itertools.combinations(metric_names, i + 1)] for i in range(len(metric_names))]
        )
    )


HPOBENCH_DATASET_NAMES = HPOBench.available_dataset_names.copy()
HPOLIB_DATASET_NAMES = HPOLib.available_dataset_names.copy()
NB201_DATASET_NAMES = NASBench201.available_dataset_names.copy()
HPOBENCH_METRIC_CHOICES = _get_metric_choices(list(HPOBench.available_metric_names))
HPOLIB_METRIC_CHOICES = _get_metric_choices(list(HPOLib.available_metric_names))
NB201_METRIC_CHOICES = _get_metric_choices(list(NASBench201.available_metric_names))


def _validate_metric_names(metric_choices: list[str] | None, bench: BaseHPOBench) -> None:
    if metric_choices is None:
        metric_names = bench.metric_names
        assert metric_names is not None and len(metric_names) == 1
    else:
        metric_names = bench.metric_names
        assert all(m1 == m2 for m1, m2 in zip(metric_choices, metric_names))


@pytest.mark.parametrize("dataset_name", HPOBENCH_DATASET_NAMES)
@pytest.mark.parametrize("metric_choices", HPOBENCH_METRIC_CHOICES)
def test_hpobench(dataset_name: str, metric_choices: list[str] | None) -> None:
    bench = HPOBench(dataset_name=dataset_name, metric_names=metric_choices)
    _validate_metric_names(metric_choices, bench)

    def objective(trial: optuna.Trial) -> list[float]:
        params = {}
        for param_name, choices in bench.search_space.items():
            params[param_name] = choices[trial.suggest_int(f"{param_name}_index", low=0, high=len(choices) - 1)]

        results = bench(params)
        return [results[name] for name in bench.metric_names]

    study = optuna.create_study(directions=[bench.directions[name] for name in bench.metric_names])
    study.optimize(objective, n_trials=30)


@pytest.mark.parametrize("dataset_name", HPOLIB_DATASET_NAMES)
@pytest.mark.parametrize("metric_choices", HPOLIB_METRIC_CHOICES)
def test_hpolib(dataset_name: str, metric_choices: list[str] | None) -> None:
    bench = HPOLib(dataset_name=dataset_name, metric_names=metric_choices)
    _validate_metric_names(metric_choices, bench)

    def objective(trial: optuna.Trial) -> list[float]:
        param_types = bench.param_types
        params = {}
        for param_name, choices in bench.search_space.items():
            if param_types[param_name] == str:
                choice = trial.suggest_categorical(param_name, choices)
                assert choice is not None, "MyPy Redefinition."
            else:
                choice = choices[trial.suggest_int(f"{param_name}_index", low=0, high=len(choices) - 1)]

            params[param_name] = choice

        results = bench(params)
        return [results[name] for name in bench.metric_names]

    study = optuna.create_study(directions=[bench.directions[name] for name in bench.metric_names])
    study.optimize(objective, n_trials=30)


@pytest.mark.parametrize("dataset_name", NB201_DATASET_NAMES)
@pytest.mark.parametrize("metric_choices", NB201_METRIC_CHOICES)
def test_nasbench201(dataset_name: str, metric_choices: list[str] | None) -> None:
    bench = NASBench201(dataset_name=dataset_name, metric_names=metric_choices)
    _validate_metric_names(metric_choices, bench)

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


@pytest.mark.parametrize("bench_cls", (HPOBench, HPOLib, NASBench201))
def test_bench_properties(bench_cls: type[BaseHPOBench]) -> None:
    assert all(d in ("maximize", "minimize") for d in bench_cls._metric_directions.values())
    assert all(name in bench_cls._metric_directions for name in bench_cls.available_metric_names)
    assert set(bench_cls.search_space.keys()) == set(bench_cls.param_types.keys())
    assert all(isinstance(choices[0], bench_cls.param_types[k]) for k, choices in bench_cls.search_space.items())


@pytest.mark.parametrize("bench_cls", (HPOBench, HPOLib, NASBench201))
def test_bench_reproducibility(bench_cls: type[BaseHPOBench]) -> None:
    bench = bench_cls(bench_cls.available_dataset_names[0])
    params = {k: choices[0] for k, choices in bench_cls.search_space.items()}

    bench.reseed(42)
    out1 = [bench(params) for _ in range(10)]
    bench.reseed(42)
    out2 = [bench(params) for _ in range(10)]
    assert all(o1 == o2 for o1, o2 in zip(out1, out2))

    bench.reseed()
    out1 = [bench(params) for _ in range(10)]
    bench.reseed()
    out2 = [bench(params) for _ in range(10)]
    assert any(o1 != o2 for o1, o2 in zip(out1, out2))
