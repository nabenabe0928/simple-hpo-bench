from __future__ import annotations

import itertools

from hpo_benchmarks import HPOBench
from hpo_benchmarks import HPOLib
from hpo_benchmarks import NASBench201
from hpo_benchmarks.base import BaseHPOBench
import optuna
import pytest


def _get_metric_choices(metric_names: list[str]) -> list[None | list[str]]:
    return [None] + list(
        itertools.chain(*[[list(it) for it in itertools.combinations(metric_names, i + 1)] for i in range(len(metric_names))])
    )


HPOBENCH_DATASET_NAMES = HPOBench("australian")._dataset_names.copy()
HPOLIB_DATASET_NAMES = HPOLib("naval_propulsion")._dataset_names.copy()
NB201_DATASET_NAMES = NASBench201("imagenet")._dataset_names.copy()
HPOBENCH_METRIC_CHOICES = _get_metric_choices(list(HPOBench("australian")._metric_directions.keys()))
HPOLIB_METRIC_CHOICES = _get_metric_choices(list(HPOLib("naval_propulsion")._metric_directions.keys()))
NB201_METRIC_CHOICES = _get_metric_choices(list(NASBench201("imagenet")._metric_directions.keys()))


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
                params[param_name] = trial.suggest_categorical(param_name, choices)
            else:
                params[param_name] = choices[trial.suggest_int(f"{param_name}_index", low=0, high=len(choices) - 1)]

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
            params[param_name] = trial.suggest_categorical(param_name, choices)

        results = bench(params)
        return [results[name] for name in bench.metric_names]

    study = optuna.create_study(directions=[bench.directions[name] for name in bench.metric_names])
    study.optimize(objective, n_trials=30)


def _validate_bench_properties(bench: BaseHPOBench) -> None:
    all(d in ("maximize", "minimize") for d in bench.directions)
    assert set(bench.search_space.keys()) == set(bench.param_types.keys())
    assert all(isinstance(choices[0], bench.param_types[k]) for k, choices in bench.search_space.items())


def test_hpobench_properties() -> None:
    _validate_bench_properties(HPOBench("australian"))


def test_hpolib_properties() -> None:
    _validate_bench_properties(HPOLib("naval_propulsion"))


def test_nasbench201_properties() -> None:
    _validate_bench_properties(NASBench201("imagenet"))


def _validate_bench_reproducibility(bench: BaseHPOBench) -> None:
    params = {k: choices[0] for k, choices in bench.search_space.items()}

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


def test_hpobench_reproducibility() -> None:
    _validate_bench_reproducibility(HPOBench("australian"))


def test_hpolib_reproducibility() -> None:
    _validate_bench_reproducibility(HPOLib("naval_propulsion"))


def test_nasbench201_reproducibility() -> None:
    _validate_bench_reproducibility(NASBench201("imagenet"))