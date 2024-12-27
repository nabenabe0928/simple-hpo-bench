from __future__ import annotations

from hpo_benchmarks.base import BaseHPOBench


class NASBench201(BaseHPOBench):
    @property
    def _dataset_names(self) -> list[str]:
        return ["cifar10", "cifar100", "imagenet"]

    @property
    def _bench_name(self) -> str:
        return "nasbench201"

    @property
    def search_space(self) -> dict[str, list[int | float | str]]:
        return {f"Op{i}": ["none", "skip_connect", "nor_conv_1x1", "nor_conv_3x3", "avg_pool_3x3"] for i in range(6)}

    @property
    def param_types(self) -> dict[str, type[int | float | str]]:
        return {f"Op{i}": str for i in range(6)}

    @property
    def _metric_directions(self) -> dict[str, str]:
        return {"train_time": "minimize", "val_acc": "maximize", "model_size": "minimize", "latency": "minimize"}

    @property
    def _main_metric_name(self) -> str:
        return "val_acc"
