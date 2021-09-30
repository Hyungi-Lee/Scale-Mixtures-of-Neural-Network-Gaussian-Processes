from jax import numpy as jnp

from neural_tangents.predict import gradient_descent_mse_ensemble

from .base import Module, ConstraintTrainVar
from .bijectors import positive


class NNGPKernel(Module):
    def __init__(
        self,
        get_kernel_fn,
        w_std: float = 1.0,
        b_std: float = 1.0,
        last_w_std: float = 1.0,
    ):
        super().__init__()
        self._get_kernel_fn = get_kernel_fn
        self.w_std = ConstraintTrainVar(jnp.array(w_std), constraint=positive())
        self.b_std = ConstraintTrainVar(jnp.array(b_std), constraint=positive())
        self.last_w_std = ConstraintTrainVar(jnp.array(last_w_std), constraint=positive())

    def K(self, kernel_fn, x, x2=None):
        if x2 is None:
            return kernel_fn(x, x, get="nngp")
        else:
            return kernel_fn(x, x2, get="nngp")

    def predict(self, kernel_fn, x, y, x_test, eps=1e-6):
        predict_fn = gradient_descent_mse_ensemble(kernel_fn, x, y, diag_reg=eps)
        mean, cov = predict_fn(x_test=x_test, get="nngp", compute_cov=True)
        return mean, cov

    def get_params(self):
        return (self.w_std.safe_value, self.b_std.safe_value, self.last_w_std.safe_value)

    def get_kernel_fn(self):
        w_std = self.w_std.safe_value
        b_std = self.b_std.safe_value
        last_w_std = self.last_w_std.safe_value
        return self._get_kernel_fn(w_std, b_std, last_w_std)
