from __future__ import annotations

from dataclasses import dataclass

from hpo_benchmarks.base import BaseHPOBenchmark


@dataclass(frozen=True)
class _HPOLibPropertiesClass:
    available_dataset_names: list[str]
    available_metric_names: list[str]
    param_types: dict[str, type[int | float | str]]
    search_space: dict[str, list[int | float | str]]
    _metric_directions: dict[str, str]
    _bench_name: str = "hpolib"
    _main_metric_name: str = "val_loss"


_metric_directions = {"train_time": "minimize", "val_loss": "minimize", "model_size": "minimize"}
hpolib_properties = _HPOLibPropertiesClass(
    available_dataset_names=[
        "naval_propulsion",
        "parkinsons_telemonitoring",
        "protein_structure",
        "slice_localization",
    ],
    available_metric_names=BaseHPOBenchmark._get_available_metric_names(_metric_directions),
    param_types={
        "activation_fn_1": str,
        "activation_fn_2": str,
        "batch_size": int,
        "dropout_1": float,
        "dropout_2": float,
        "init_lr": float,
        "lr_schedule": str,
        "n_units_1": int,
        "n_units_2": int,
    },
    search_space={
        "activation_fn_1": ["relu", "tanh"],
        "activation_fn_2": ["relu", "tanh"],
        "batch_size": [8, 16, 32, 64],
        "dropout_1": [0.0, 0.3, 0.6],
        "dropout_2": [0.0, 0.3, 0.6],
        "init_lr": [5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1],
        "lr_schedule": ["cosine", "const"],
        "n_units_1": [16, 32, 64, 128, 256, 512],
        "n_units_2": [16, 32, 64, 128, 256, 512],
    },
    _metric_directions=_metric_directions,
)


class HPOLib(BaseHPOBenchmark):
    available_metric_names: list[str] = hpolib_properties.available_metric_names
    search_space: dict[str, list[int | float | str]] = hpolib_properties.search_space
    param_types: dict[str, type[int | float | str]] = hpolib_properties.param_types
    available_dataset_names: list[str] = hpolib_properties.available_dataset_names

    def __init__(self, dataset_name: str, seed: int | None = None, metric_names: list[str] | None = None):
        super().__init__(hpolib_properties, dataset_name=dataset_name, seed=seed, metric_names=metric_names)
