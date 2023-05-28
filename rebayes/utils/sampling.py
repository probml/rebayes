"""
Utilities for sampling from a distribution.
"""
import jax
import numpy as np
import jax.numpy as jnp
import jax.random as jr
from functools import partial


def sample_dlr_single(key, W, diag, temperature=1.0):
    """
    Sample from an MVG with diagonal + low-rank
    covariance matrix. See §4.2.2, Proposition 1 of
    L-RVGA paper
    """
    key_x, key_eps = jax.random.split(key)
    diag_inv = (1 / diag).ravel()
    diag_inv_mod = diag_inv * temperature
    D, d = W.shape
    
    ID = jnp.eye(D)
    Id = jnp.eye(d)
    
    M = Id + jnp.einsum("ji,j,jk->ik", W, diag_inv, W)
    L = jnp.sqrt(temperature) * jnp.linalg.solve(M.T, jnp.einsum("ji,j->ij", W, diag_inv)).T
    
    x = jax.random.normal(key_x, (D,)) * jnp.sqrt(diag_inv_mod)
    eps = jax.random.normal(key_eps, (d,))
    
    x_plus = jnp.einsum("ij,kj,k->i", L, W, x)
    x_plus = x - x_plus + jnp.einsum("ij,j->i", L, eps)
    
    return x_plus


@partial(jax.jit, static_argnums=(4,))
def sample_dlr(key, W, diag, temperature=1.0, shape=None):
    shape = (1,) if shape is None else shape
    n_elements = np.prod(shape)
    keys = jax.random.split(key, n_elements)
    samples = jax.vmap(sample_dlr_single, in_axes=(0, None, None, None))(keys, W, diag, temperature)
    samples = samples.reshape(*shape, -1)
    
    return samples
