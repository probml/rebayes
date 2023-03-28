"""
Utilities for sampling from a distribution.
"""
import jax
import numpy as np
import jax.numpy as jnp
from functools import partial


def sample_dlr_single(key, W, diag):
    """
    Sample from an MVG with diagonal + low-rank
    covariance matrix. See §4.2.2, Proposition 1 of
    L-RVGA paper
    """
    key_x, key_eps = jax.random.split(key)
    diag_inv = (1 / diag).ravel()
    D, d = W.shape
    
    ID = jnp.eye(D)
    Id = jnp.eye(d)
    
    M = Id + jnp.einsum("ji,j,jk->ik", W, diag_inv, W)
    L = jnp.linalg.solve(M.T, jnp.einsum("ji,j->ij", W, diag_inv)).T
    
    x = jax.random.normal(key_x, (D,)) * jnp.sqrt(diag_inv)
    eps = jax.random.normal(key_eps, (d,))
    
    x_plus = (ID - jnp.einsum("ij,kj->ik", L, W))
    x_plus = jnp.einsum("ij,j->i", x_plus, x) + jnp.einsum("ij,j->i", L, eps)
    return x_plus


@partial(jax.jit, static_argnums=(3,))
def sample_dlr(key, W, diag, shape=None):
    shape = (1,) if shape is None else shape
    n_elements = np.prod(shape)
    keys = jax.random.split(key, n_elements)
    samples = jax.vmap(sample_dlr_single, in_axes=(0, None, None))(keys, W, diag)
    samples = samples.reshape(shape + W.shape)
    return samples
