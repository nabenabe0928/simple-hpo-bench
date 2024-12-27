from __future__ import annotations

from hpo_benchmarks.base import BaseHPOBench


class HPOLib(BaseHPOBench):
    @property
    def _dataset_names(self) -> list[str]:
        return ["naval_propulsion", "parkinsons_telemonitoring", "protein_structure", "slice_localization"]

    @property
    def _bench_name(self) -> str:
        return "hpolib"

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

    @property
    def _metric_directions(self) -> dict[str, str]:
        return {"train_time": "minimize", "val_loss": "minimize", "model_size": "minimize"}

    @property
    def _main_metric_name(self) -> str:
        return "val_loss"
