from jax import random
from jax import numpy as jnp
from jax.scipy.special import gammaln, digamma

from .base import Module, ConstraintTrainVar
from .bijectors import positive
from .utils import logdet, trace, multivariate_t


class Prior(Module):
    pass


class GaussianPrior(Prior):
    def sample_f_b(self, key, num_class, num_batch, num_samples):
        sampled_f_b = random.normal(key, shape=(num_class, num_batch, num_samples))
        return sampled_f_b

    def sample_f(self, key, mean, cov, num_samples):
        num_class = mean.shape[0]
        sampled_f = random.multivariate_normal(key, mean, cov, shape=(num_samples, num_class))
        sampled_f = sampled_f.transpose(1, 2, 0)
        return sampled_f

    def kl_divergence(self, k_ii, k_ii_inv, q_mu, q_sigma, num_inducing, num_class):
        kl = 1 / 2 * ((logdet(k_ii) * num_class - logdet(q_sigma)) \
                      - (num_inducing * num_class) \
                      + trace(jnp.matmul(k_ii_inv[None, :, :], q_sigma)) \
                      + jnp.einsum("ci,ij,jc->", q_mu, k_ii_inv, q_mu.T))
        return kl


class InverseGammaPrior(Prior):
    def __init__(self, alpha, beta):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.a = ConstraintTrainVar(jnp.array(alpha), constraint=positive())
        self.b = ConstraintTrainVar(jnp.array(beta), constraint=positive())

    def sample_f_b(self, key, num_class, num_batch, num_samples):
        key_normal, key_gamma = random.split(key)
        sampled_f_b = random.normal(key_normal, shape=(num_class, num_batch, num_samples))
        gamma_pure = random.gamma(key_gamma, a=self.a.safe_value)
        invgamma = self.b.safe_value / gamma_pure  # invgamma ~ invgamma(a = nu_q/2, scale = rho_q/2)
        sampled_f_b = sampled_f_b * jnp.sqrt(invgamma)
        return sampled_f_b

    def sample_f(self, key, mean, cov, num_samples):
        df = 2 * self.a.safe_value
        num_class = mean.shape[0]
        sampled_f = multivariate_t(key, df, mean, cov, shape=(num_samples, num_class))
        sampled_f = sampled_f.transpose(1, 2, 0)
        return sampled_f

    def kl_divergence(self, k_ii, k_ii_inv, q_mu, q_sigma, num_inducing, num_class):
        invgamma_a = self.a.safe_value
        invgamma_b = self.b.safe_value
        a_by_b = invgamma_a / invgamma_b
        kl = 1 / 2 * ((logdet(k_ii) * num_class - logdet(q_sigma)) \
                      - (num_inducing * num_class) \
                      + trace(jnp.matmul(k_ii_inv[None, :, :], q_sigma)) \
                      + jnp.einsum("ci,ij,jc->", q_mu, k_ii_inv, q_mu.T) * a_by_b) \
           + self.alpha * jnp.log(invgamma_b / self.beta) \
           - gammaln(invgamma_a) + gammaln(self.alpha) \
           + (invgamma_a - self.alpha) * digamma(invgamma_a) \
           + (self.beta - invgamma_b) * a_by_b
        return kl
