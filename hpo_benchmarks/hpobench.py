from __future__ import annotations

from dataclasses import dataclass

from hpo_benchmarks.base import BaseHPOBenchmark


@dataclass(frozen=True)
class _HPOBenchPropertiesClass:
    available_dataset_names: list[str]
    available_metric_names: list[str]
    param_types: dict[str, type[int | float | str]]
    search_space: dict[str, list[int | float | str]]
    _metric_directions: dict[str, str]
    _bench_name: str = "hpobench"
    _main_metric_name: str = "val_acc"


_metric_directions = {
    "train_time": "minimize",
    "val_acc": "maximize",
    "val_precision": "maximize",
    "val_f1": "maximize",
}
_available_metric_names = BaseHPOBenchmark._get_available_metric_names(_metric_directions)
hpobench_properties = _HPOBenchPropertiesClass(
    available_dataset_names=[
        "car",
        "phoneme",
        "vehicle",
        "australian",
        "kc1",
        "segment",
        "blood_transfusion",
        "credit_g",
    ],
    available_metric_names=BaseHPOBenchmark._get_available_metric_names(_metric_directions),
    param_types={"alpha": float, "batch_size": int, "depth": int, "learning_rate_init": float, "width": int},
    search_space={
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
            1,
        ],
        "batch_size": [4, 6, 10, 16, 25, 40, 64, 101, 161, 256],
        "depth": [1, 2, 3],
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
            1,
        ],
        "width": [16, 25, 40, 64, 101, 161, 256, 406, 645, 1024],
    },
    _metric_directions=_metric_directions,
)


class HPOBench(BaseHPOBenchmark):
    available_metric_names: list[str] = hpobench_properties.available_metric_names
    search_space: dict[str, list[int | float | str]] = hpobench_properties.search_space
    param_types: dict[str, type[int | float | str]] = hpobench_properties.param_types
    available_dataset_names: list[str] = hpobench_properties.available_dataset_names

    def __init__(self, dataset_name: str, seed: int | None = None, metric_names: list[str] | None = None):
        super().__init__(hpobench_properties, dataset_name=dataset_name, seed=seed, metric_names=metric_names)
