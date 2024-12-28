from __future__ import annotations

from dataclasses import dataclass

from hpo_benchmarks.base import BaseHPOBenchmark


@dataclass(frozen=True)
class _NASBench201PropertiesClass:
    available_dataset_names: list[str]
    available_metric_names: list[str]
    param_types: dict[str, type[int | float | str]]
    search_space: dict[str, list[int | float | str]]
    _metric_directions: dict[str, str]
    _bench_name: str = "nasbench201"
    _main_metric_name: str = "val_acc"


_metric_directions = {"train_time": "minimize", "val_acc": "maximize", "model_size": "minimize", "latency": "minimize"}
nb201_properties = _NASBench201PropertiesClass(
    available_dataset_names=["cifar10", "cifar100", "imagenet"],
    available_metric_names=BaseHPOBenchmark._get_available_metric_names(_metric_directions),
    param_types={f"Op{i}": str for i in range(6)},
    search_space={f"Op{i}": ["none", "skip_connect", "nor_conv_1x1", "nor_conv_3x3", "avg_pool_3x3"] for i in range(6)},
    _metric_directions=_metric_directions,
)


class NASBench201(BaseHPOBenchmark):
    available_metric_names: list[str] = nb201_properties.available_metric_names
    search_space: dict[str, list[int | float | str]] = nb201_properties.search_space
    param_types: dict[str, type[int | float | str]] = nb201_properties.param_types
    available_dataset_names: list[str] = nb201_properties.available_dataset_names

    def __init__(self, dataset_name: str, seed: int | None = None, metric_names: list[str] | None = None):
        super().__init__(nb201_properties, dataset_name=dataset_name, seed=seed, metric_names=metric_names)
