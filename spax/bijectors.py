from typing import Optional, Callable

import abc

from jax import nn
from jax import numpy as jnp

__all__ = [
    "positive",
    "triangular",
]

DEFAULT_POSITIVE_MINIMUM = lambda: 0.0
DEFAULT_POSITIVE_BIJECTOR = lambda: "softplus"
BIJECTOR_TYPE_MAP = lambda: {
    "exp": lambda *args, **kwargs: Exp(*args, **kwargs),
    "softplus": lambda *args, **kwargs: Softplus(*args, **kwargs),
}


class Bijector:
    @abc.abstractmethod
    def __call__(self, x):
        pass

    @abc.abstractmethod
    def inverse(self, x):
        pass


class PositiveBijector(Bijector):
    base: Callable
    base_inv: Callable

    def __init__(self, lower: float = 0.):
        super().__init__()
        self.lower = lower

    def __call__(self, x):
        return self.lower + self.base(x)

    def inverse(self, x):
        return self.base_inv(x - self.lower)


class Exp(PositiveBijector):
    base = lambda _, x: jnp.exp(x)
    base_inv = lambda _, x: jnp.log(x)


class Softplus(PositiveBijector):
    base = lambda _, x: nn.softplus(x)
    base_inv = lambda _, x: jnp.where(x < 20., jnp.log(jnp.expm1(x)), x)


def positive(lower: Optional[float] = None, base: Optional[str] = None) -> PositiveBijector:
    lower_bound = lower if lower is not None else DEFAULT_POSITIVE_MINIMUM()

    bijector = base if base is not None else DEFAULT_POSITIVE_BIJECTOR()
    bijector = BIJECTOR_TYPE_MAP()[bijector.lower()](lower_bound)

    return bijector


def triangular():
    raise NotImplementedError
