from __future__ import annotations

import pickle

import numpy as np


class NASBench201:
    def __init__(self, dataset_name: str, seed: int | None = None):
        dataset_names = ["cifar10", "cifar100", "imagenet"]
        if dataset_name not in dataset_names:
            raise ValueError(f"dataset_name must be in {dataset_names}, but got {dataset_name}.")

        self._dataset = pickle.load(open(f"nasbench201/{dataset_name}.pkl", mode="rb"))
        self._dataset_name = dataset_name
        self._rng = np.random.RandomState(seed)

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self._dataset_name})"

    def __call__(self, params: dict[str, int | float | str]) -> float:
        search_space = self.search_space
        param_types = self.param_types
        param_indices = [
            str(np.arange(len(search_space[param_name]))[np.isclose(value, search_space[param_name])][0])
            if param_types[param_name] == float
            else str(search_space[param_name].index(value))
            for param_name, value in params.items()
        ]
        param_id = "".join(param_indices)
        vals = self._dataset[param_id]
        seed = self._rng.randint(len(vals))
        return vals[seed]

    @property
    def direction(self) -> str:
        return "maximize"

    @property
    def search_space(self) -> dict[str, list[int | float | str]]:
        return {f"Op{i}": ["none", "skip_connect", "nor_conv_1x1", "nor_conv_3x3", "avg_pool_3x3"] for i in range(6)}

    @property
    def param_types(self) -> dict[str, type[int | float | str]]:
        return {f"Op{i}": str for i in range(6)}
