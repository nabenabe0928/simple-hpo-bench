from __future__ import annotations

from abc import ABCMeta
import os
import pickle
from typing import Protocol

import numpy as np


class BaseDatasetProperties(Protocol):
    @property
    def available_dataset_names(self) -> list[str]:
        raise NotImplementedError

    @property
    def available_metric_names(self) -> list[str]:
        raise NotImplementedError

    @property
    def param_types(self) -> dict[str, type[int | float | str]]:
        raise NotImplementedError

    @property
    def search_space(self) -> dict[str, list[int | float | str]]:
        raise NotImplementedError

    @property
    def _bench_name(self) -> str:
        raise NotImplementedError

    @property
    def _main_metric_name(self) -> str:
        raise NotImplementedError

    @property
    def _metric_directions(self) -> dict[str, str]:
        raise NotImplementedError


class HPOBenchmarkInterface(Protocol):
    available_metric_names: list[str]
    search_space: dict[str, list[int | float | str]]
    param_types: dict[str, type[int | float | str]]
    available_dataset_names: list[str]

    def __init__(self, dataset_name: str, seed: int | None = None, metric_names: list[str] | None = None):
        raise NotImplementedError

    def reseed(self, seed: int | None = None) -> None:
        raise NotImplementedError

    def __call__(self, params: dict[str, int | float | str]) -> dict[str, float]:
        raise NotImplementedError

    @property
    def directions(self) -> dict[str, str]:
        raise NotImplementedError

    @property
    def metric_names(self) -> list[str]:
        raise NotImplementedError


class BaseHPOBenchmark(metaclass=ABCMeta):
    def __init__(
        self,
        dataset_properties: BaseDatasetProperties,
        dataset_name: str,
        seed: int | None,
        metric_names: list[str] | None,
    ):
        self._dataset_properties = dataset_properties
        available_dataset_names = self._dataset_properties.available_dataset_names
        if dataset_name not in available_dataset_names:
            raise ValueError(f"dataset_name must be in {available_dataset_names}, but got {dataset_name}.")
        if metric_names is not None and any(mn not in self._metric_directions for mn in metric_names):
            raise ValueError(f"metric_names must be in {list(self._metric_directions.keys())}, but got {metric_names}.")

        curdir = os.path.dirname(os.path.abspath(__file__))
        self._dataset = pickle.load(
            open(os.path.join(curdir, f"datasets/{self._bench_name}/{dataset_name}.pkl"), mode="rb")
        )
        self._dataset_name = dataset_name
        self._rng = np.random.default_rng(seed)
        self._metric_names = metric_names.copy() if metric_names is not None else [self._main_metric_name]

    def reseed(self, seed: int | None = None) -> None:
        self._rng = np.random.default_rng(seed)

    def __str__(self) -> str:
        return f'{self.__class__.__name__}(dataset_name="{self._dataset_name}", metric_names={self._metric_names})'

    def __call__(self, params: dict[str, int | float | str]) -> dict[str, float]:
        search_space = self._dataset_properties.search_space
        param_types = self._dataset_properties.param_types
        param_indices = []
        for param_name, choices in search_space.items():
            value = params[param_name]
            if param_types[param_name] == float:
                param_indices.append(str(np.arange(len(choices))[np.isclose(value, choices)][0]))
            else:
                param_indices.append(str(choices.index(value)))

        param_id = "".join(param_indices)
        vals = self._dataset[param_id]
        seed = self._rng.integers(len(vals[self._main_metric_name]))
        return {name: vals[name][seed] for name in self._metric_names}

    @property
    def directions(self) -> dict[str, str]:
        metric_directions = self._metric_directions
        return {name: metric_directions[name] for name in self._metric_names}

    @property
    def metric_names(self) -> list[str]:
        return self._metric_names.copy()

    @staticmethod
    def _get_available_metric_names(metric_directions: dict[str, str]) -> list[str]:
        return list(metric_directions.keys())

    @property
    def _bench_name(self) -> str:
        return self._dataset_properties._bench_name

    @property
    def _metric_directions(self) -> dict[str, str]:
        return self._dataset_properties._metric_directions.copy()

    @property
    def _main_metric_name(self) -> str:
        return self._dataset_properties._main_metric_name
