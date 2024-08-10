from __future__ import annotations

from abc import ABCMeta
from abc import abstractmethod

import numpy as np


class BaseFunc(metaclass=ABCMeta):
    def __init__(self, dim: int):
        self._dim = dim

    def __call__(self, X: np.ndarray) -> float:
        raise NotImplementedError

    def _validate_dim(self, X: np.ndarray) -> None:
        if X.size != self._dim:
            raise ValueError(f"The shape of X must be ({self._dim}, ), but got {X.shape}.")

    def __str__(self) -> str:
        return self.__class__.__name__

    @property
    def dim(self) -> int:
        return self._dim

    @property
    @abstractmethod
    def _param_range(self) -> float:
        raise NotImplementedError

    @property
    def param_range(self) -> tuple[float, float]:
        return (-self._param_range, self._param_range)


class Sphere(BaseFunc):
    @property
    def _param_range(self) -> float:
        return 5.0

    def __call__(self, X: np.ndarray) -> float:
        self._validate_dim(X)
        return np.sum(X**2)


class Styblinski(BaseFunc):
    @property
    def _param_range(self) -> float:
        return 5.0

    def __call__(self, X: np.ndarray) -> float:
        self._validate_dim(X)
        t1 = np.sum(X**4)
        t2 = -16 * np.sum(X**2)
        t3 = 5 * np.sum(X)
        return 0.5 * (t1 + t2 + t3)


class Rastrigin(BaseFunc):
    @property
    def _param_range(self) -> float:
        return 5.12

    def __call__(self, X: np.ndarray) -> float:
        self._validate_dim(X)
        dim = len(X)
        t1 = 10 * dim
        t2 = np.sum(X**2)
        t3 = -10 * np.sum(np.cos(2 * np.pi * X))
        return t1 + t2 + t3


class Schwefel(BaseFunc):
    @property
    def _param_range(self) -> float:
        return 500.0

    def __call__(self, X: np.ndarray) -> float:
        self._validate_dim(X)
        return -np.sum(X * np.sin(np.sqrt(np.abs(X))))


class Ackley(BaseFunc):
    @property
    def _param_range(self) -> float:
        return 32.768

    def __call__(self, X: np.ndarray) -> float:
        self._validate_dim(X)
        t1 = -20 * np.exp(-0.2 * np.sqrt(np.mean(X**2)))
        t2 = -np.exp(np.mean(np.cos(2 * np.pi * X)))
        return 20 + np.e + t1 + t2


class Griewank(BaseFunc):
    @property
    def _param_range(self) -> float:
        return 600.0

    def __call__(self, X: np.ndarray) -> float:
        self._validate_dim(X)
        dim = len(X)
        t1 = np.sum(X**2) / 4000
        t2 = -np.prod(np.cos(X / np.sqrt(np.arange(1, dim + 1))))
        return 1 + t1 + t2


class Perm(BaseFunc):
    @property
    def _param_range(self) -> float:
        return 1.0

    def __call__(self, X: np.ndarray) -> float:
        self._validate_dim(X)
        dim = len(X)
        indices = np.arange(dim) + 1
        ret = 0
        for d in indices:
            center = (1 / indices) ** d
            factor = X**d
            ret += ((indices + 1.0) @ (factor - center)) ** 2

        return ret


class KTablet(BaseFunc):
    @property
    def _param_range(self) -> float:
        return 5.12

    def __call__(self, X: np.ndarray) -> float:
        self._validate_dim(X)
        dim = len(X)
        k = (dim + 3) // 4
        X[k:] *= 100
        return np.sum(X**2)


class WeightedSphere(BaseFunc):
    @property
    def _param_range(self) -> float:
        return 5.0

    def __call__(self, X: np.ndarray) -> float:
        self._validate_dim(X)
        dim = len(X)
        weights = np.arange(dim) + 1
        return float(weights @ (X**2))


class Rosenbrock(BaseFunc):
    @property
    def _param_range(self) -> float:
        return 5.0

    def __call__(self, X: np.ndarray) -> float:
        self._validate_dim(X)
        t1 = np.sum(100 * (X[1:] - X[:-1] ** 2) ** 2)
        t2 = np.sum((X[:-1] - 1) ** 2)
        return t1 + t2


class DifferentPower(BaseFunc):
    @property
    def _param_range(self) -> float:
        return 1.0

    def __call__(self, X: np.ndarray) -> float:
        self._validate_dim(X)
        dim = len(X)
        indices = np.arange(dim) + 2
        return np.sum(np.abs(X) ** indices)


class XinSheYang(BaseFunc):
    @property
    def _param_range(self) -> float:
        return 2 * np.pi

    def __call__(self, X: np.ndarray) -> float:
        self._validate_dim(X)
        return np.sum(np.abs(X)) * np.exp(-np.sum(np.sin(X**2)))


class Levy(BaseFunc):
    @property
    def _param_range(self) -> float:
        return 10.0

    def __call__(self, X: np.ndarray) -> float:
        self._validate_dim(X)
        weights = 1 + (X - 1) / 4
        t1 = np.sin(np.pi * weights[0]) ** 2
        t2 = (weights[-1] - 1) ** 2 * (1 + np.sin(2 * np.pi * weights[-1]) ** 2)
        t3 = np.sum((weights[:-1] - 1) ** 2 * (1 + 10 * np.sin(np.pi * weights[:-1] + 1) ** 2))
        return t1 + t2 + t3


class Michalewicz(BaseFunc):
    @property
    def _param_range(self) -> float:
        return np.pi / 2

    def __call__(self, X: np.ndarray) -> float:
        self._validate_dim(X)
        t1 = np.sin(X)
        t2 = np.sin((np.arange(len(X)) + 1) * (X**2) / np.pi) ** 20
        return -t1 @ t2


class DixonPrice(BaseFunc):
    @property
    def _param_range(self) -> float:
        return 10.0

    def __call__(self, X: np.ndarray) -> float:
        self._validate_dim(X)
        d = X.size
        t1 = (X[0] - 1) ** 2
        t2 = np.arange(2, d + 1) @ (2 * X[1:] ** 2 - X[:-1]) ** 2
        return t1 + t2


class Powell(BaseFunc):
    @property
    def _param_range(self) -> float:
        return 5.0

    def __call__(self, X: np.ndarray) -> float:
        self._validate_dim(X)
        size = X.size // 4
        x1 = X[::4][:size]  # 4i - 3
        x2 = X[1::4][:size]  # 4i - 2
        x3 = X[2::4][:size]  # 4i - 1
        x4 = X[3::4][:size]  # 4i
        t1 = np.sum((x1 + 10 * x2) ** 2)
        t2 = np.sum(5 * (x3 - x4) ** 2)
        t3 = np.sum((x2 + 2 * x3) ** 4)
        t4 = np.sum(10 * (x1 - x4) ** 4)
        return t1 + t2 + t3 + t4


class Langermann(BaseFunc):
    _rng = np.random.default_rng(42)

    @property
    def _param_range(self) -> float:
        return 5.0

    def __call__(self, X: np.ndarray) -> float:
        self._validate_dim(X)
        dim = X.size
        m = 5
        C = self._rng.integers(low=1, high=5, size=m).astype(np.float64)
        A = self._rng.integers(low=1, high=10, size=(m, dim)).astype(np.float64)
        exps = np.sum((X - A) ** 2, axis=-1)
        return (C * np.exp(-1 / np.pi * exps)) @ np.cos(np.pi * exps)
