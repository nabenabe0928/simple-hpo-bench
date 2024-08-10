from __future__ import annotations

import pickle

import numpy as np


class HPOBench:
    def __init__(self, dataset_name: str, seed: int | None = None):
        dataset_names = ["car", "phoneme", "vehicle", "australian", "kc1", "segment", "blood_transfusion", "credit_g"]
        if dataset_name not in dataset_names:
            raise ValueError(f"dataset_name must be in {dataset_names}, but got {dataset_name}.")

        self._dataset = pickle.load(open(f"hpobench/{dataset_name}.pkl", mode="rb"))
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
        seed = self._rng.randint(5)
        return self._dataset[param_id][seed]

    @property
    def direction(self) -> str:
        return "maximize"

    @property
    def search_space(self) -> dict[str, list[int | float | str]]:
        return {
            "alpha": [
                1e-8,
                7.742637e-8,
                5.994842e-7,
                4.641589e-6,
                3.5938137e-5,
                2.7825593e-4,
                2.1544348e-3,
                1.6681006e-2,
                1.2915497e-1,
                1
            ],
            "batch_size": [
                4,
                6,
                10,
                16,
                25,
                40,
                64,
                101,
                161,
                256
            ],
            "depth": [
                1,
                2,
                3
            ],
            "learning_rate_init": [
                1e-5,
                3.5938137e-5,
                1.2915497e-4,
                4.641589e-4,
                1.6681006e-3,
                5.9948424e-3,
                2.1544347e-2,
                7.742637e-2,
                2.7825594e-1,
                1
            ],
            "width": [
                16,
                25,
                40,
                64,
                101,
                161,
                256,
                406,
                645,
                1024
            ],
        }

    @property
    def param_types(self) -> dict[str, type[int | float | str]]:
        return {
            "alpha": float,
            "batch_size": int,
            "depth": int,
            "learning_rate_init": float,
            "width": int,
        }
