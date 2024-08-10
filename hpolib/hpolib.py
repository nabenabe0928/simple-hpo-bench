from __future__ import annotations

import pickle

import numpy as np


class HPOLib:
    def __init__(self, dataset_name: str, seed: int | None = None):
        dataset_names = ["naval_propulsion", "parkinsons_telemonitoring", "protein_structure", "slice_localization"]
        if dataset_name not in dataset_names:
            raise ValueError(f"dataset_name must be in {dataset_names}, but got {dataset_name}.")

        self._dataset = pickle.load(open(f"hpolib/{dataset_name}.pkl", mode="rb"))
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
        seed = self._rng.randint(4)
        return self._dataset[param_id][seed]

    @property
    def direction(self) -> str:
        return "minimize"

    @property
    def search_space(self) -> dict[str, list[int | float | str]]:
        return {
            "activation_fn_1": ["relu", "tanh"],
            "activation_fn_2": ["relu", "tanh"],
            "batch_size": [8, 16, 32, 64],
            "dropout_1": [0.0, 0.3, 0.6],
            "dropout_2": [0.0, 0.3, 0.6],
            "init_lr": [5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1],
            "lr_schedule": ["cosine", "const"],
            "n_units_1": [16, 32, 64, 128, 256, 512],
            "n_units_2": [16, 32, 64, 128, 256, 512],
        }

    @property
    def param_types(self) -> dict[str, type[int | float | str]]:
        return {
            "activation_fn_1": str,
            "activation_fn_2": str,
            "batch_size": int,
            "dropout_1": float,
            "dropout_2": float,
            "init_lr": float,
            "lr_schedule": str,
            "n_units_1": int,
            "n_units_2": int,
        }
