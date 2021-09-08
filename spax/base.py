from typing import Optional, Callable

from objax.typing import JaxArray
from objax.module import Module
from objax.variable import TrainVar, reduce_mean


__all__ = [
    "Module",
    "TrainVar",
    "ConstraintTrainVar",
]


class ConstraintTrainVar(TrainVar):
    """A constraint trainable variable."""

    def __init__(self, tensor: JaxArray, constraint: Callable, reduce: Optional[Callable[[JaxArray], JaxArray]] = reduce_mean):
        inv_tensor = constraint.inverse(tensor)
        super().__init__(inv_tensor, reduce)
        self.constraint = constraint

    @property
    def safe_value(self) -> JaxArray:
        return self.constraint(self._value)

    def __repr__(self):
        return super().__repr__()[:-1] + f", constraint={self.constraint__class__.__name__})"
