from __future__ import annotations

from hpo_benchmarks.base import BaseHPOBench


class HPOBench(BaseHPOBench):
    @property
    def _dataset_names(self) -> list[str]:
        return ["car", "phoneme", "vehicle", "australian", "kc1", "segment", "blood_transfusion", "credit_g"]

    @property
    def _bench_name(self) -> str:
        return "hpobench"

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
        }

    @property
    def param_types(self) -> dict[str, type[int | float | str]]:
        return {"alpha": float, "batch_size": int, "depth": int, "learning_rate_init": float, "width": int}

    @property
    def _metric_directions(self) -> dict[str, str]:
        return {"train_time": "minimize", "val_acc": "maximize", "val_precision": "maximize", "val_f1": "maximize"}

    @property
    def _main_metric_name(self) -> str:
        return "val_acc"
