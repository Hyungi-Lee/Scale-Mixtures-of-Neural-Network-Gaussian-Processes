import numpy as np

from jax import numpy as jnp
from jax.scipy import stats

from .base import Module, ConstraintTrainVar
from .bijectors import positive
from .utils import multivariate_t_logpdf, jitter


__all__ = [
    "Likelihood",
    "GaussianLikelihood",
    "StudentTLikelihood",
]


class Likelihood(Module):
    pass


class GaussianLikelihood(Likelihood):
    require = None

    def prior_logpdf(self, x, cov):
        zero = np.zeros_like(x)
        logpdf = stats.multivariate_normal.logpdf(x, zero, cov)
        return logpdf

    def logpdf(self, x, mean, cov, aux):
        sigma = jnp.sqrt(jnp.diag(cov))
        logpdf = stats.norm.logpdf(x, mean, sigma)
        return logpdf



class StudentTLikelihood(Likelihood):
    require = ["cov_data", "y_data"]

    def __init__(self, alpha, beta):
        super().__init__()
        self.a = ConstraintTrainVar(jnp.array(alpha), constraint=positive())
        self.b = ConstraintTrainVar(jnp.array(beta), constraint=positive())

    def prior_logpdf(self, x, cov):
        a = self.a.safe_value
        b = self.b.safe_value
        zero = np.zeros_like(x)
        log_prob = multivariate_t_logpdf(x, zero, (b / a) * cov, 2 * a)
        return log_prob

    def logpdf(self, x, mean, cov, aux):
        a = self.a.safe_value
        b = self.b.safe_value
        cov_data, y_data = aux
        num_data = cov_data.shape[-1]

        df = 2 * a
        cond_df = df + num_data
        inv_cov_data = jnp.linalg.inv(b / a * cov_data + jitter(num_data))
        d = df + jnp.einsum("i,ij,j->", y_data, inv_cov_data, y_data)
        sigma = jnp.sqrt(jnp.diag(d / cond_df * b / a * cov))

        logpdf = stats.t.logpdf(x, cond_df, mean, sigma)
        return logpdf
