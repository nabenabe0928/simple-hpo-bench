from hpo_benchmarks import hpobench
from hpo_benchmarks import hpolib
from hpo_benchmarks import nasbench201


def test_instantiation() -> None:
    hpobench.HPOBench(dataset_name="car")
    hpolib.HPOLib(dataset_name="naval_propulsion")
    nasbench201.NASBench201(dataset_name="imagenet")
