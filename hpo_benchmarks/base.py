from __future__ import annotations

from abc import ABCMeta
from abc import abstractmethod
import pickle

import numpy as np


class BaseHPOBench(metaclass=ABCMeta):
    def __init__(self, dataset_name: str, seed: int | None = None):
        if dataset_name not in self._dataset_names:
            raise ValueError(f"dataset_name must be in {self._dataset_names}, but got {dataset_name}.")

        self._dataset = pickle.load(open(f"hpo_benchmarks/datasets/{self._bench_name}/{dataset_name}.pkl", mode="rb"))
        self._dataset_name = dataset_name
        self._rng = np.random.default_rng(seed)

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self._dataset_name})"

    def __call__(self, params: dict[str, int | float | str]) -> float:
        search_space = self.search_space
        param_types = self.param_types
        param_indices = [
            (
                str(np.arange(len(search_space[param_name]))[np.isclose(value, search_space[param_name])][0])
                if param_types[param_name] == float
                else str(search_space[param_name].index(value))
            )
            for param_name, value in params.items()
        ]
        param_id = "".join(param_indices)
        vals = self._dataset[param_id]
        seed = self._rng.integers(len(vals))
        return vals[seed]

    @property
    @abstractmethod
    def _dataset_names(self) -> list[str]:
        raise NotImplementedError

    @property
    @abstractmethod
    def _bench_name(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def direction(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def search_space(self) -> dict[str, list[int | float | str]]:
        raise NotImplementedError

    @property
    @abstractmethod
    def param_types(self) -> dict[str, type[int | float | str]]:
        raise NotImplementedError
