from jax import random
from jax import numpy as jnp
from jax.scipy.special import gammaln, digamma

from .base import Module, ConstraintTrainVar
from .bijectors import positive
from .utils import logdet, trace, multivariate_t


__all__ = [
    "Prior",
    "GaussianPrior",
    "InverseGammaPrior",
]


class Prior(Module):
    pass


class GaussianPrior(Prior):
    def sample_f(self, key, mean, cov, num_samples):
        num_class = mean.shape[0]
        sampled_f = random.multivariate_normal(key, mean, cov, shape=(num_samples, num_class))
        sampled_f = sampled_f.transpose(1, 2, 0)
        return sampled_f

    def sample_f_iid(self, key, mean, cov, num_samples):
        num_class, num_batch = mean.shape
        sigma = jnp.sqrt(jnp.diagonal(cov, axis1=-2, axis2=-1))
        # sigma = jnp.full_like(mean, 1e-3)
        sampled_f = random.normal(key, shape=(num_class, num_batch, num_samples))
        sampled_f = sampled_f * sigma[..., None] + mean[..., None]
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

    def sample_f(self, key, mean, cov, num_samples):
        a = self.a.safe_value
        b = self.b.safe_value
        num_class = mean.shape[0]
        sampled_f = multivariate_t(key, 2 * a, mean, b / a * cov, shape=(num_samples, num_class))
        sampled_f = sampled_f.transpose(1, 2, 0)
        return sampled_f

    def sample_f_iid(self, key, mean, cov, num_samples):
        a = self.a.safe_value
        b = self.b.safe_value
        num_class, num_batch = mean.shape
        sigma = jnp.sqrt(jnp.diagonal(b / a * cov, axis1=-2, axis2=-1))
        # sigma = jnp.full_like(mean, 1e-3)
        sampled_f = random.t(key, 2 * a, shape=(num_class, num_batch, num_samples))
        sampled_f = sampled_f * sigma[..., None] + mean[..., None]
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

    def sample_f2(self, key, mean, cov, num_samples, a, b):
        num_class = mean.shape[0]
        sampled_f = multivariate_t(key, 2 * a, mean, b / a * cov, shape=(num_samples, num_class))
        sampled_f = sampled_f.transpose(1, 2, 0)
        return sampled_f

    def kl_divergence2(self, k_ii, k_ii_inv, q_mu, q_sigma, num_inducing, num_class, a, b):
        invgamma_a = a
        invgamma_b = b
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
