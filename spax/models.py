import numpy as np
from jax import numpy as jnp

from .base import Module, TrainVar, ConstraintTrainVar
from .bijectors import positive
from .utils import jitter, log_likelihood, test_log_likelihood, get_correct_count

from jax import random  # TODO: Remove this


__all__ = [
    "SVSP",
    "SPR",
]


class SVSP(Module):
    def __init__(self, prior, kernel, inducing_variable, *, num_latent_gps: int = 1):
        super().__init__()
        self.prior = prior
        self.kernel = kernel
        self.num_latent_gps = num_latent_gps
        self.inducing_variable = TrainVar(np.array(inducing_variable))
        self.num_inducing = self.inducing_variable.shape[0]
        self.q_mu = TrainVar(np.zeros((self.num_latent_gps, self.num_inducing)))
        self.q_sqrt = ConstraintTrainVar(
            np.ones((self.num_latent_gps, self.num_inducing)),
            constraint=positive(),
        )

    def loss(self, key, x_batch, y_batch, num_train, num_samples, aux=False):
        inducing_variable = self.inducing_variable.value  # [I, D...]
        q_mu = self.q_mu.value  # [C, I]
        q_sqrt = self.q_sqrt.safe_value  # [C, I]
        q_sigma = jnp.einsum("ci,ij->cij", q_sqrt, np.eye(self.num_inducing))  # [C, I, I]
        kernel_fn = self.kernel.get_kernel_fn()

        k_bi = self.kernel.K(kernel_fn, x_batch, inducing_variable)  # [B, I]
        k_ii = self.kernel.K(kernel_fn, inducing_variable)  # [I, I]
        k_ii_inv = jnp.linalg.inv(k_ii + jitter(self.num_inducing))  # [I, I]

        zero = np.zeros((self.num_inducing, self.num_latent_gps))  # [I, C]
        _, B_B = self.kernel.predict(kernel_fn, inducing_variable, zero, x_batch)  # [B, B]
        A_B = jnp.matmul(k_bi, k_ii_inv)  # [B, I]

        mean = jnp.matmul(q_mu, A_B.T)  # [C, B]
        cov = jnp.einsum("ij,cjk,kl->cil", A_B, q_sigma, A_B.T) + B_B[None, :, :]  # [C, B, B]
        sampled_f = self.prior.sample_f(key, mean, cov, num_samples)  # [C, B, S]

        ll = log_likelihood(sampled_f, y_batch)
        kl = self.prior.kl_divergence(k_ii, k_ii_inv, q_mu, q_sigma, self.num_inducing, self.num_latent_gps)
        n_elbo = -ll + kl / num_train
        if aux:
            return n_elbo, -ll, kl / num_train
        else:
            return n_elbo

    def test_acc_nll(self, key, x_batch, y_batch, num_samples):
        inducing_variable = self.inducing_variable.value  # [I, D...]
        q_mu = self.q_mu.value  # [C, I]
        q_sqrt = self.q_sqrt.safe_value  # [C, I]
        q_sigma = jnp.einsum("ci,ij->cij", q_sqrt, np.eye(self.num_inducing))  # [C, I, I]
        kernel_fn = self.kernel.get_kernel_fn()

        k_bi = self.kernel.K(kernel_fn, x_batch, inducing_variable)  # [B, I]
        k_ii = self.kernel.K(kernel_fn, inducing_variable)  # [I, I]
        k_ii_inv = jnp.linalg.inv(k_ii + jitter(self.num_inducing))  # [I, I]

        mean, cov = self.kernel.predict(kernel_fn, inducing_variable, q_mu.T, x_batch)  # [B, C], [B, B]
        A_B = jnp.matmul(k_bi, k_ii_inv)  # [B, I]

        test_cov = jnp.einsum("ij,cjk,kl->cil", A_B, q_sigma, A_B.T) + cov[None, :, :]  # [C, B, B]
        sampled_f = self.prior.sample_f_iid(key, mean.T, test_cov, num_samples)  # [C, B, S]

        nll = -test_log_likelihood(sampled_f, y_batch)
        correct_count = get_correct_count(sampled_f, y_batch)
        return nll, correct_count

    # TODO: Remove this
    def loss2(self, key, x_batch, y_batch, num_train, num_samples, alphas, betas):
        inducing_variable = self.inducing_variable.value  # [I, D...]
        q_mu = self.q_mu.value  # [C, I]
        q_sqrt = self.q_sqrt.safe_value  # [C, I]
        q_sigma = jnp.einsum("ci,ij->cij", q_sqrt, np.eye(self.num_inducing))  # [C, I, I]
        kernel_fn = self.kernel.get_kernel_fn()

        k_bi = self.kernel.K(kernel_fn, x_batch, inducing_variable)  # [B, I]
        k_ii = self.kernel.K(kernel_fn, inducing_variable)  # [I, I]
        k_ii_inv = jnp.linalg.inv(k_ii + jitter(self.num_inducing))  # [I, I]

        y = np.zeros((self.num_inducing, self.num_latent_gps))  # [I, C]
        _, B_B = self.kernel.predict(kernel_fn, inducing_variable, y, x_batch)  # [B, B]
        A_B = jnp.matmul(k_bi, k_ii_inv)  # [B, I]

        mean = jnp.matmul(q_mu, A_B.T)  # [C, B]
        cov = jnp.einsum("ij,cjk,kl->cil", A_B, q_sigma, A_B.T) + B_B[None, :, :]  # [C, B, B]

        nelbos = []
        for a in alphas:
            for b in betas:
                sampled_f = self.prior.sample_f2(key, mean, cov, num_samples, a, b)  # [C, B, S]

                ll = log_likelihood(sampled_f, y_batch)
                kl = self.prior.kl_divergence2(k_ii, k_ii_inv, q_mu, q_sigma, self.num_inducing, self.num_latent_gps, a, b)
                n_elbo = -ll + kl / num_train
                nelbos.append(n_elbo)

        return nelbos

    # TODO: Remove this
    def test_acc_nll2(self, key, x_batch, y_batch, num_samples, alphas, betas):
        inducing_variable = self.inducing_variable.value  # [I, D...]
        q_mu = self.q_mu.value  # [C, I]
        q_sqrt = self.q_sqrt.safe_value  # [C, I]
        q_sigma = jnp.einsum("ci,ij->cij", q_sqrt, np.eye(self.num_inducing))  # [C, I, I]
        kernel_fn = self.kernel.get_kernel_fn()

        k_bi = self.kernel.K(kernel_fn, x_batch, inducing_variable)  # [B, I]
        k_ii = self.kernel.K(kernel_fn, inducing_variable)  # [I, I]
        k_ii_inv = jnp.linalg.inv(k_ii + jitter(self.num_inducing))  # [I, I]

        mean, cov = self.kernel.predict(kernel_fn, inducing_variable, q_mu.T, x_batch)  # [B, C], [B, B]
        A_L = jnp.matmul(k_bi, k_ii_inv)  # [B, I]

        test_cov = jnp.einsum("ij,cjk,kl->cil", A_L, q_sigma, A_L.T) + cov[None, :, :]  # [C, B, B]

        nlls = []
        ccs = []
        num_class, num_batch = mean.T.shape
        for a in alphas:
            for b in betas:
                sigma = jnp.sqrt(jnp.diagonal(b / a * test_cov, axis1=-2, axis2=-1))
                sampled_f = random.t(key, 2 * a, shape=(num_class, num_batch, num_samples))
                sampled_f = sampled_f * sigma[..., None] + mean.T[..., None]

                nll = -test_log_likelihood(sampled_f, y_batch)
                correct_count = get_correct_count(sampled_f, y_batch)
                nlls.append(nll)
                ccs.append(correct_count)

        return nlls, ccs


class SPR(Module):
    def __init__(self, kernel, likelihood, x_data, y_data):
        super().__init__()
        self.kernel = kernel
        self.likelihood = likelihood
        self.x_data = x_data
        self.y_data = y_data
        self.num_data = x_data.shape[0]

    def loss(self):
        kernel_fn = self.kernel.get_kernel_fn()
        cov = self.kernel.K(kernel_fn, self.x_data) + jitter(self.num_data)
        log_prob = self.likelihood.prior_logpdf(self.y_data, cov)
        return -log_prob / self.num_data

    def test_nll(self, x, y):
        kernel_fn = self.kernel.get_kernel_fn()
        mean, cov = self.kernel.predict(kernel_fn, self.x_data, self.y_data[:, None], x)
        require = self.likelihood.require
        if require:
            if "cov_data" in require:
                cov_data = self.kernel.K(kernel_fn, self.x_data)
            aux_dict = dict(cov_data=cov_data, y_data=self.y_data, num_data=self.num_data)
            aux = tuple(aux_dict[k] for k in require)
        else:
            aux = None

        log_prob = self.likelihood.logpdf(y, mean.flatten(), cov, aux)
        ll = jnp.mean(log_prob)
        return -ll
