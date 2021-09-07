import numpy as np

from jax import numpy as jnp
# from jax.nn import log_softmax, logsumexp


__all__ = [
    "matmul3",
    "jitter",
    "kron_diag",
    "split_kernel",
    "get_true_values",
    "logsumexp",
    "log_softmax",
    "logdet",
    "log_likelihood",
    "test_log_likelihood",
    "get_correct_count",
    "multivariate_t",
]


def matmul3(mat0, mat1, mat2):
    return jnp.matmul(jnp.matmul(mat0, mat1), mat2)


def jitter(num):
    return 1e-6 * np.eye(num)


def kron_diag(data, n):
    data_expanded = jnp.kron(np.eye(n), data)
    return data_expanded


def split_kernel(kernel, num_11):
    kernel_11 = kernel[:num_11, :num_11]
    kernel_12 = kernel[:num_11, num_11:]
    kernel_21 = kernel[num_11:, :num_11]
    kernel_22 = kernel[num_11:, num_11:]
    return kernel_11, kernel_12, kernel_21, kernel_22


def get_true_values(value, label):
    """
    Parameters
    ----------
    value: [sample_num, batch_num, class_num]
    label: [batch_num, class_num]

    Returns
    -------
    true_values: [sample_num, batch_num]
    """

    sample_num = value.shape[0]
    label_idx = jnp.argmax(label, axis=-1)[jnp.newaxis, :, jnp.newaxis]
    value_idx = jnp.repeat(label_idx, sample_num, axis=0)
    true_values = jnp.take_along_axis(value, value_idx, axis=-1).squeeze(axis=-1)
    return true_values


def logsumexp(data, axis=-1):
    data_max = jnp.max(data, axis=axis, keepdims=True)
    data_exp = jnp.exp(data - data_max)
    data_sum = jnp.log(jnp.sum(data_exp, axis=axis, keepdims=True))
    data_logsumexp = data_sum + data_max
    return data_logsumexp


def log_softmax(data):
    return data - logsumexp(data)


def logdet(data):
    sign, abslogdet = jnp.linalg.slogdet(data)
    return sign * abslogdet


def log_likelihood(label, sampled_f):
    sampled_f_softmax = log_softmax(sampled_f)
    true_label_softmax = get_true_values(sampled_f_softmax, label)
    ll = jnp.mean(jnp.mean(true_label_softmax, axis=1))
    return ll


def test_log_likelihood(label, sampled_f, sample_num):
    sampled_f_softmax = log_softmax(sampled_f)
    true_label_softmax = get_true_values(sampled_f_softmax, label)
    ll = jnp.sum(logsumexp(true_label_softmax.T) - jnp.log(sample_num))
    return ll


def get_correct_count(label, sampled_f):
    sampled_f_softmax = jnp.exp(log_softmax(sampled_f))
    mean_f_softmax = jnp.mean(sampled_f_softmax, axis=0)
    predict_y = jnp.argmax(mean_f_softmax, axis=-1)
    true_y = jnp.argmax(label, axis=-1)
    correct_count = jnp.sum(predict_y == true_y)
    return correct_count


# multivariate_t implementation from jax.random.multivariate_normal

from typing import Any, Sequence, Optional
from functools import partial
from jax import lax
from jax import core
from jax._src import dtypes
from jax._src.api import jit
from jax.numpy.linalg import cholesky, svd, eigh
from jax._src.random import _check_shape, t


Array = Any
RealArray = Array
DTypeLikeFloat = Any


def multivariate_t(key: jnp.ndarray,
                   df: RealArray,
                   mean: RealArray,
                   cov: RealArray,
                   shape: Optional[Sequence[int]] = None,
                   dtype: DTypeLikeFloat = dtypes.float_,
                   method: str = 'cholesky') -> jnp.ndarray:
  if method not in {'svd', 'eigh', 'cholesky'}:
    raise ValueError("method must be one of {'svd', 'eigh', 'cholesky'}")
  if not dtypes.issubdtype(dtype, np.floating):
    raise ValueError(f"dtype argument to `multivariate_t` must be a float "
                     f"dtype, got {dtype}")
  dtype = dtypes.canonicalize_dtype(dtype)
  if shape is not None:
    shape = core.canonicalize_shape(shape)
  return _multivariate_t(key, df, mean, cov, shape, dtype, method)  # type: ignore


@partial(jit, static_argnums=(4, 5, 6))
def _multivariate_t(key, df, mean, cov, shape, dtype, method) -> jnp.ndarray:
  if not np.ndim(mean) >= 1:
    msg = "multivariate_t requires mean.ndim >= 1, got mean.ndim == {}"
    raise ValueError(msg.format(np.ndim(mean)))
  if not np.ndim(cov) >= 2:
    msg = "multivariate_t requires cov.ndim >= 2, got cov.ndim == {}"
    raise ValueError(msg.format(np.ndim(cov)))
  n = mean.shape[-1]
  if np.shape(cov)[-2:] != (n, n):
    msg = ("multivariate_t requires cov.shape == (..., n, n) for n={n}, "
           "but got cov.shape == {shape}.")
    raise ValueError(msg.format(n=n, shape=np.shape(cov)))

  if shape is None:
    shape = lax.broadcast_shapes(mean.shape[:-1], cov.shape[:-2])
  else:
    _check_shape("t", shape, mean.shape[:-1], cov.shape[:-2])

  if method == 'svd':
    (u, s, _) = svd(cov)
    factor = u * jnp.sqrt(s)
  elif method == 'eigh':
    (w, v) = eigh(cov)
    factor = v * jnp.sqrt(w)
  else: # 'cholesky'
    factor = cholesky(cov)
  t_samples = t(key, df, shape + mean.shape[-1:], dtype)
  return mean + jnp.einsum('...ij,...j->...i', factor, t_samples)
