import numpy as np

from jax import numpy as jnp
from jax.nn import log_softmax, logsumexp


__all__ = [
    "matmul3",
    "jitter",
    "split_kernel",
    "get_true_values",
    "logdet",
    "trace",
    "log_likelihood",
    "test_log_likelihood",
    "get_correct_count",
    "multivariate_t",
    "multivariate_t_logpdf",
]


def matmul3(mat0, mat1, mat2):
    return jnp.matmul(jnp.matmul(mat0, mat1), mat2)


def jitter(num, eps=1e-6):
    return eps * np.eye(num)


def split_kernel(kernel, num_11):
    kernel_11 = kernel[:num_11, :num_11]
    kernel_12 = kernel[:num_11, num_11:]
    kernel_21 = kernel[num_11:, :num_11]
    kernel_22 = kernel[num_11:, num_11:]
    return kernel_11, kernel_12, kernel_21, kernel_22


def logdet(data):
    sign, abslogdet = jnp.linalg.slogdet(data)
    return jnp.sum(sign * abslogdet)


def trace(data):
    return jnp.sum(jnp.trace(data, axis1=-2, axis2=-1))


def get_true_values(value, label):
    label = label[jnp.newaxis, :, jnp.newaxis]
    value_idx = jnp.repeat(label, value.shape[2], axis=2)
    true_values = jnp.take_along_axis(value, value_idx, axis=0).squeeze(axis=0)
    return true_values


def log_likelihood(sampled_f, label):
    sampled_f_softmax = log_softmax(sampled_f, axis=0)
    true_label_softmax = get_true_values(sampled_f_softmax, label)
    ll = jnp.mean(jnp.mean(true_label_softmax, axis=0))
    return ll


def test_log_likelihood(sampled_f, label):
    num_samples = sampled_f.shape[2]
    sampled_f_softmax = log_softmax(sampled_f, axis=0)  # [C, B, S]
    true_label_softmax = get_true_values(sampled_f_softmax, label)  # [B, S]
    ll = jnp.mean(logsumexp(true_label_softmax, axis=1) - np.log(num_samples))
    return ll


def get_correct_count(sampled_f, label):
    sampled_f_softmax = log_softmax(sampled_f, axis=0)  # [C, B, S]
    sum_f_softmax = logsumexp(sampled_f_softmax, axis=2)  # [C, B]
    y_pred = jnp.argmax(sum_f_softmax, axis=0)  # [B]
    correct_count = jnp.sum(y_pred == label)
    return correct_count


# multivariate_t implementation fork from jax.random.multivariate_normal

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



# multivariate_t.log_pdf implementation fork from jax.scipy.stats.multivariate_normal.logpdf

import numpy as np
import scipy.stats as osp_stats

from jax import lax
from jax import numpy as jnp
from jax._src.numpy.util import _wraps
from jax._src.numpy.lax_numpy import _promote_dtypes_inexact
from jax._src.scipy.stats import multivariate_normal, t as student_t
from jax._src.scipy.special import gammaln


# @_wraps(osp_stats.multivariate_t.logpdf, update_doc=False, lax_description="""
# In the JAX version, the `allow_singular` argument is not implemented.
# """)
def multivariate_t_logpdf(x, loc, shape, df, allow_singular=None):
  if allow_singular is not None:
    raise NotImplementedError("allow_singular argument of multivariate_t.logpdf")
  # TODO: Properly handle df == np.inf
  # if df == np.inf:
  #   return multivariate_normal.logpdf(x, loc, shape)
  x, loc, shape, df = _promote_dtypes_inexact(x, loc, shape, df)
  if not loc.shape:
    return student_t.logpdf(x, df, loc=loc, scale=jnp.sqrt(shape))
  else:
    n = loc.shape[-1]
    if not np.shape(shape):
      y = x - loc
      # TODO: Implement this
      raise NotImplementedError("multivariate_t.logpdf doesn't support scalar shape")
    else:
      if shape.ndim < 2 or shape.shape[-2:] != (n, n):
        raise ValueError("multivariate_t.logpdf got incompatible shapes")
      t = 1/2 * (df + n)
      L = lax.linalg.cholesky(shape)
      y = lax.linalg.triangular_solve(L, x - loc, lower=True, transpose_a=True)
      return (-t * jnp.log(1 + 1/df * jnp.einsum('...i,...i->...', y, y))
              - n/2*jnp.log(df*np.pi) + gammaln(t) - gammaln(1/2 * df)
              - jnp.log(L.diagonal(axis1=-1, axis2=-2)).sum(-1))
