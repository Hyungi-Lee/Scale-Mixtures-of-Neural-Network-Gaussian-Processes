from jax import random
from jax import numpy as jnp
from jax.scipy.special import gammaln, digamma

from .base import Module, ConstraintTrainVar
from .bijectors import positive
from .utils import *


class Prior(Module):
    pass


class GaussianPrior(Prior):
    def sample_f_b(self, key, num_batch, num_classes, num_samples):
        mean = jnp.zeros((num_batch * num_classes))
        covariance = jnp.eye(num_batch * num_classes)
        sampled_f = random.multivariate_normal(key, mean, covariance, shape=(num_samples,))
        return sampled_f

    def sample_f(self, key, mean, cov, num_samples):
        sampled_f = random.multivariate_normal(key, mean, cov, shape=(num_samples,))
        return sampled_f

    def kl_divergence(self, k_ii, k_ii_inv, q_mu, q_sigma, num_inducing, num_classes):
        k_ii_inv = kron_diag(k_ii_inv, n=num_classes)
        q_mu_t = q_mu[None, ...]
        q_mu_vec = q_mu[..., None]

        kl = 1 / 2 * ((logdet(k_ii) - logdet(q_sigma)) \
                    - (num_inducing * num_classes) \
                    + jnp.trace(jnp.matmul(k_ii_inv, q_sigma)) \
                    + matmul3(q_mu_t, k_ii_inv, q_mu_vec))
        return jnp.sum(kl)


class InverseGammaPrior(Prior):
    def __init__(self, alpha, beta):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.a = ConstraintTrainVar(jnp.array(alpha), constraint=positive())
        self.b = ConstraintTrainVar(jnp.array(beta), constraint=positive())

    def sample_f_b(self, key, num_batch, num_classes, num_samples):
        key_normal, key_gamma = random.split(key)

        mean = jnp.zeros((num_batch * num_classes))
        covariance = jnp.eye(num_batch * num_classes)
        sampled_f = random.multivariate_normal(key_normal, mean, covariance, shape=(num_samples,))

        gamma_pure = random.gamma(key_gamma, a=self.a.safe_value)
        # gamma_rho = gamma_pure / self.b.safe_value
        # invgamma = 1 / gamma_rho  # invgamma ~ invgamma(a = nu_q/2, scale = rho_q/2)
        invgamma = self.b.safe_value / gamma_pure  # invgamma ~ invgamma(a = nu_q/2, scale = rho_q/2)
        sigma = jnp.sqrt(invgamma)
        sampled_f = sampled_f * sigma

        return sampled_f

    def sample_f(self, key, mean, cov, num_samples):
        # sampled_f = stats.multivariate_t.rvs(
        #     random_state=key[1],
        #     loc=mean,
        #     shape=cov,
        #     df=(2 * self.a.safe_value),
        #     size=(num_samples,),
        # )
        df = 2 * self.a.safe_value
        sampled_f = multivariate_t(key, df, mean, cov, shape=(num_samples,))
        return sampled_f

    def kl_divergence(self, k_ii, k_ii_inv, q_mu, q_sigma, num_inducing, num_classes):
        k_ii_inv = kron_diag(k_ii_inv, n=num_classes)
        invgamma_a = self.a.safe_value
        invgamma_b = self.b.safe_value
        q_mu_t = q_mu[None, ...]
        q_mu_vec = q_mu[..., None]

        kl = 1 / 2 * (invgamma_a / invgamma_b * matmul3(q_mu_t, k_ii_inv, q_mu_vec) \
                    + jnp.trace(jnp.matmul(k_ii_inv, q_sigma)) \
                    + (logdet(k_ii) - logdet(q_sigma)) \
                    - (num_inducing * num_classes)) \
        + self.alpha * jnp.log(invgamma_b / self.beta) \
        - gammaln(invgamma_a) + gammaln(self.alpha) \
        + (invgamma_a - self.alpha) * digamma(invgamma_a) \
        + (self.beta - invgamma_b) * invgamma_a / invgamma_b

        return jnp.sum(kl)
