# Simple HPO Benchmark Datasets

<div>
    <img src="https://img.shields.io/badge/python-3.8%20%7C%203.9%20%7C%203.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue" alt="Python"/>
</div>

This repository provides a set of simple single-objective HPO benchmark datasets:
- [HPOBench](https://github.com/automl/hpobench) (Apache-2.0 License)
- [HPOLib](https://arxiv.org/abs/1905.04970) (BSD-3-Clause License)
- [NAS-Bench-201](https://github.com/D-X-Y/NATS-Bench) (MIT License)

# Installation & Requirements

The requirements of this repository are:
- Python 3.8 or later
- NumPy

You can simply install the package via:

```shell
$ pip install simple-hpo-bench
```

# Examples

Examples are available at [examples/](./examples/).

For example, `HPOBench` can be optimized by Optuna as follows:

```python
from __future__ import annotations

from hpo_benchmarks import HPOBench

import optuna


# Instatiate the benchmark function.
bench = HPOBench(dataset_name=HPOBench.available_dataset_names[0])
print(bench)


def objective(trial: optuna.Trial) -> list[float]:
    params = {}
    for param_name, choices in bench.search_space.items():
        params[param_name] = choices[trial.suggest_int(f"{param_name}_index", low=0, high=len(choices) - 1)]

    results = bench(params)
    return [results[name] for name in bench.metric_names]


study = optuna.create_study(direction=[bench.directions[name] for name in bench.metric_names])
study.optimize(objective, n_trials=30)

```

For tabular benchmarks (`HPOLib`, `HPOBench`, and `NASBench201`), the arguments are:
- `dataset_name`: one of the dataset names of the benchmark dataset of interest, and
- `seed`: the random seed for the benchmark dataset.
- `metric_names`: a list of metric names to fetch from the benchmark datasets. The available metric names can be found in the `available_metric_names` attribute. If not specified, we return only the main metric.

Class attributes available to users are:
- `available_metric_names`: A list of available metric names.
- `available_dataset_names`: A list of available dataset names.
- `search_space`: The choices of each parameter.
- `param_types`: The data type of each parameter.

Instance attributes available to users are:
- `directions`: The directions (`minimize` or `maximize`) of each metric specified by the user.
- `metric_names`: The user-specified list of metrics to optimize.

The methods provided to users are:
- `reseed`: Reset the random seed to the provided seed.
- `__call__`: The method to evaluate the provided parameters and return the corresponding metrics.

The available dataset names for each benchmark dataset are as follows:

|Benchmark Dataset| Available Dataset Names |
|:--|:--|
|`HPOLib`|`naval_propulsion`, `parkinsons_telemonitoring`, `protein_structure`, `slice_localization`|
|`HPOBench`|`australian`, `blood_transfusion`, `car`, `credit_g`, `kc1`, `phoneme`, `segment`, `vehicle`|
|`NASBench201`|`cifar10`, `cifar100`, `imagenet`|

# Search Space

In this section, `bench` is an instantiated benchmark.

The search space of the benchmark functions are `[bench.param_range[0], bench.param_range[1]]^bench.dim`.

The search space of `HPOLib`, `HPOBench`, and `NASBench201` is defined in `bench.search_space`.
`bench.search_space` is a `dict` that takes `param_name` as a key and the corresponding possible choices as a value.
`bench.param_types` defines the types of each parameter.
`int` is for an integer parameter, `str` is for a categorical parameter, and `float` is for a non-integer numerical parameter.
