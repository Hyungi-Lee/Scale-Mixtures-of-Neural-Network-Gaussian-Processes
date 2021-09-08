from jax import numpy as jnp

from .base import Module, TrainVar, ConstraintTrainVar
from .bijectors import positive
from .utils import *


__all__ = [
    "SVSP",
]


class SVSP(Module):
    def __init__(
        self,
        prior,
        kernel,
        inducing_variable,
        *,
        num_latent_gps: int = 1,
    ):
        super().__init__()
        self.prior = prior
        self.kernel = kernel
        self.num_latent_gps = num_latent_gps
        self.inducing_variable = TrainVar(jnp.array(inducing_variable))
        self.num_inducing = self.inducing_variable.shape[0]
        self.q_mu = TrainVar(jnp.zeros((self.num_inducing * self.num_latent_gps)))
        self.q_sqrt = ConstraintTrainVar(
            # jnp.ones((self.num_inducing * self.num_latent_gps)),
            jnp.full((self.num_inducing * self.num_latent_gps), 1e-6),
            constraint=positive(),
        )

    def mean_cov(self, x_batch, q_mu, q_sigma, kernel_fn, k_bi, k_ii_inv):
        inducing_variable = self.inducing_variable.value
        _, B_B = self.kernel.predict(kernel_fn, inducing_variable, jnp.zeros((self.num_inducing, self.num_latent_gps)), x_batch)

        A_B = jnp.matmul(k_bi, k_ii_inv)
        # L = cholesky(k_i_i)
        # A_B_L = matmul(A_B, L)
        A_B_L = A_B
        A_B_L = kron_diag(A_B_L, n=self.num_latent_gps)
        B_B = kron_diag(B_B, n=self.num_latent_gps)

        mean = jnp.matmul(A_B_L, q_mu)
        cov = matmul3(A_B_L, q_sigma, A_B_L.T) + B_B
        cov_L = jnp.linalg.cholesky(cov)

        return mean, cov_L

    def loss(self, key, x_batch, y_batch, num_batch, num_train, num_samples):
        inducing_variable = self.inducing_variable.value
        q_mu = self.q_mu.value
        q_sigma = jnp.diag(self.q_sqrt.safe_value)
        kernel_fn = self.kernel.get_kernel_fn()

        k_batch_induced = self.kernel.K(kernel_fn, jnp.concatenate((inducing_variable, x_batch), axis=0))
        k_ii, _, k_bi, _ = split_kernel(k_batch_induced, self.num_inducing)
        k_ii_inv = jnp.linalg.inv(k_ii + jitter(self.num_inducing))

        mean, cov_L = self.mean_cov(x_batch, q_mu, q_sigma, kernel_fn, k_bi, k_ii_inv)

        sampled_f = self.prior.sample_f_b(key, num_batch, self.num_latent_gps, num_samples)
        sampled_f = (mean.reshape(-1, 1) + jnp.matmul(cov_L, sampled_f.T)).T
        sampled_f = jnp.transpose(sampled_f.reshape(num_samples, self.num_latent_gps, num_batch), axes=(0, 2, 1))

        ll = log_likelihood(y_batch, sampled_f)
        kl = self.prior.kl_divergence(k_ii, k_ii_inv, q_mu, q_sigma, self.num_inducing, self.num_latent_gps)

        n_elbo = -ll + kl / num_train
        return n_elbo

    def test_acc_nll(self, key, x_batch, y_batch, num_batch, num_samples):
        inducing_variable = self.inducing_variable.value
        q_mu = self.q_mu.value
        q_sigma = jnp.diag(self.q_sqrt.safe_value)
        kernel_fn = self.kernel.get_kernel_fn()

        k_batch_induced = self.kernel.K(kernel_fn, jnp.concatenate((x_batch, inducing_variable), axis=0))
        k_ii, _, k_bi, _ = split_kernel(k_batch_induced, self.num_inducing)
        k_ii_inv = jnp.linalg.inv(k_ii + jitter(self.num_inducing))

        # L_induced = jnp.linalg.cholesky(k_ii)
        # L_induced = kron_diag(L_induced, n=self.num_inducing)
        # L_mu = matmul(L_induced, inducing_mu)
        L_mu = q_mu

        # A_L = matmul3(k_bi, k_ii_inv, L_induced)
        A_L = jnp.matmul(k_bi, k_ii_inv)
        A_L = kron_diag(A_L, n=self.num_latent_gps)

        mean, cov = self.kernel.predict(kernel_fn, inducing_variable, jnp.transpose(L_mu.reshape(-1, self.num_inducing)), x_batch)

        mean = mean.T.flatten()
        cov = kron_diag(cov, n=self.num_latent_gps)
        test_sigma = matmul3(A_L, q_sigma, A_L.T) + cov

        sampled_f = self.prior.sample_f(key, mean, test_sigma, num_samples)
        sampled_f = sampled_f.reshape(num_samples, self.num_latent_gps, num_batch)
        sampled_f = jnp.transpose(sampled_f, axes=(0, 2, 1))

        nll = -test_log_likelihood(y_batch, sampled_f, num_samples)
        correct_count = get_correct_count(y_batch, sampled_f)
        return nll, correct_count
