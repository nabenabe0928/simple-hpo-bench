# Simple HPO Benchmark Datasets

This repository provides a set of simple single-objective HPO benchmark datasets:
- [HPOBench](https://github.com/automl/hpobench)
- [HPOLib](https://arxiv.org/abs/1905.04970)
- [NAS-Bench-201](https://github.com/D-X-Y/NATS-Bench)
- Continuous Famous Benchmark Functions

# Installation & Requirements

The requirements of this repository are:
- Python 3.7 or later
- NumPy

You can simply install the package via:

```shell
$ pip install simple-hpo-bench
```

# Examples

Examples are available at [examples/](./examples/).

For example, `HPOBench` can be optimized by Optuna as follows:

```python
import optuna

from hpo_benchmarks import HPOBench


# Instatiate the benchmark function.
bench = HPOBench(dataset_name="australian")


def objective(trial: optuna.Trial) -> float:
    params = {}
    # search_space is a dict that takes a parameter name as a key and the corresponding parameter's choices as a value.
    for param_name, choices in bench.search_space.items():
        params[param_name] = choices[trial.suggest_int(f"{param_name}_index", low=0, high=len(choices) - 1)]

    return bench(params)


study = optuna.create_study(direction=bench.direction)
study.optimize(objective, n_trials=30)

```

For benchmark functions, the argument for these classes is only `dim`, which determines the dimension of the function.

For tabular benchmarks (`HPOLib`, `HPOBench`, and `NASBench201`), the arguments are:
- `dataset_name`: one of the dataset names of the benchmark dataset of interest, and
- `seed`: the random seed for the benchmark dataset.

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
