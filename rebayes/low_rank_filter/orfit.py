from functools import partial

import jax.numpy as jnp
import chex
from jax import jacrev, jit, vmap

from rebayes.base import Rebayes, RebayesParams


# Helper functions
_jacrev_2d = lambda f, x: jnp.atleast_2d(jacrev(f)(x))
_stable_division = lambda a, b: jnp.where(b.any(), a / b, jnp.zeros(shape=a.shape))
_normalize = lambda v: jnp.where(v.any(), v / jnp.linalg.norm(v), jnp.zeros(shape=v.shape))
_project = lambda a, x: _stable_division(a * (a.T @ x), (a.T @ a))
_project_to_columns = lambda A, x: \
    jnp.where(A.any(), vmap(_project, (1, None))(A, x).sum(axis=0), jnp.zeros(shape=x.shape))


@chex.dataclass
class ORFitBel:
    mean: chex.Array
    basis: chex.Array
    singular_values: chex.Array

    
class RebayesORFit(Rebayes):
    def __init__(
        self,
        params: RebayesParams,
        memory_size: int = 10,
    ):
        self.params = params
        self.memory_size = memory_size
    
    def init_bel(self):
        init_basis = jnp.zeros((len(self.params.initial_mean), self.memory_size))
        init_singular_values = jnp.zeros((self.memory_size,))
        
        return ORFitBel(
            mean=self.params.initial_mean,
            basis=init_basis,
            singular_values=init_singular_values,
        )
    
    @partial(jit, static_argnums=(0,))
    def predict_obs(
        self,
        bel: ORFitBel,
        x: float
    ):
        m = bel.mean
        m_Y = lambda z: self.params.emission_mean_function(z, x)
        y_pred = jnp.atleast_1d(m_Y(m))
        
        return y_pred
    
    @partial(jit, static_argnums=(0,))
    def predict_obs_cov(
        self,
        bel: ORFitBel,
        x: float,
    ):
        m, U = bel.mean, bel.basis
        m_Y = lambda z: self.params.emission_mean_function(z, x)
        
        # Compute predicted observation covariance
        H = _jacrev_2d(m_Y, m)
        Sigma_obs = H @ H.T - (H @ U) @ (H @ U).T
        
        return Sigma_obs
    
    @partial(jit, static_argnums=(0,))
    def update_state(
        self,
        bel: ORFitBel,
        x: float,
        y: float,
    ):
        m, U, Lambda = bel.mean, bel.basis, bel.singular_values
        
        # Update the state
        m_cond, U_cond, Lambda_cond = _orfit_condition_on(
            m, U, Lambda, self.params.emission_mean_function, x, y
        )
        bel_cond =  bel.replace(
            mean = m_cond,
            basis = U_cond,
            singular_values = Lambda_cond,
        )
        
        return bel_cond


def _orfit_condition_on(m, U, Lambda, apply_fn, x, y):
    """Condition on the emission using ORFit.

    Args:
        m (D_hid,): Prior mean.
        U (D_hid, D_mem,): Prior basis.
        Lambda (D_mem,): Prior singular values.
        apply_fn (Callable): Apply function.
        x (D_in,): Control input.
        y (D_obs,): Emission.

    Returns:
        m_cond (D_hid,): Posterior mean.
        U_cond (D_hid, D_mem,): Posterior basis.
        Lambda_cond (D_mem,): Posterior singular values.
    """    
    f_fn = lambda w: apply_fn(w, x)

    # Compute Jacobians and project out the orthogonal components
    v = jacrev(f_fn)(m).squeeze()
    v_prime = v - _project_to_columns(U, v)

    # Update the U matrix
    u = _normalize(v_prime)
    U_cond = jnp.where(Lambda.min() < u @ v_prime, U.at[:, Lambda.argmin()].set(u), U)
    Lambda_cond = jnp.where(Lambda.min() < u @ v_prime, Lambda.at[Lambda.argmin()].set(u.T @ v_prime), Lambda)
    
    # Update the parameters
    m_cond = m - _stable_division((f_fn(m) - y) * v_prime, v.T @ v_prime)

    return m_cond, U_cond, Lambda_cond