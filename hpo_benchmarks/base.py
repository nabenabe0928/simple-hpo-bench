from __future__ import annotations

from abc import ABCMeta
from abc import abstractmethod
import os
import pickle

import numpy as np


class BaseHPOBench(metaclass=ABCMeta):
    def __init__(self, dataset_name: str, seed: int | None = None, metric_names: list[str] | None = None):
        if dataset_name not in self._dataset_names:
            raise ValueError(f"dataset_name must be in {self._dataset_names}, but got {dataset_name}.")
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
        search_space = self.search_space
        param_types = self.param_types
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
    @abstractmethod
    def _dataset_names(self) -> list[str]:
        raise NotImplementedError

    @property
    @abstractmethod
    def _bench_name(self) -> str:
        raise NotImplementedError

    @property
    def metric_names(self) -> list[str]:
        return self._metric_names.copy()

    @property
    @abstractmethod
    def _metric_directions(self) -> dict[str, str]:
        raise NotImplementedError

    @property
    def directions(self) -> dict[str, str]:
        metric_directions = self._metric_directions
        return {name: metric_directions[name] for name in self._metric_names}

    @property
    @abstractmethod
    def search_space(self) -> dict[str, list[int | float | str]]:
        raise NotImplementedError

    @property
    @abstractmethod
    def param_types(self) -> dict[str, type[int | float | str]]:
        raise NotImplementedError

    @property
    @abstractmethod
    def _main_metric_name(self) -> str:
        raise NotImplementedError
