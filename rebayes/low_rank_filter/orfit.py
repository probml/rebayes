from functools import partial
from typing import Any, Callable, Union

import chex
from jax import jacrev, jit, vmap
import jax.numpy as jnp
from jaxtyping import Float, Array
import tensorflow_probability.substrates.jax as tfp

from rebayes.base import (
    EmissionDistFn,
    FnStateToEmission,
    FnStateToEmission2,
    CovMat,
    FnStateAndInputToEmission,
    FnStateAndInputToEmission2,
    Rebayes,
)
from rebayes.low_rank_filter.lofi_core import _jacrev_2d, _normalize
from rebayes.utils.sampling import sample_dlr


tfd = tfp.distributions
MVN = tfd.MultivariateNormalTriL

# Helper functions
_stable_division = lambda a, b: jnp.where(b.any(), a / b, jnp.zeros(shape=a.shape))
_project = lambda a, x: _stable_division(a * (a.T @ x), (a.T @ a))
_project_to_columns = lambda A, x: \
    jnp.where(A.any(), vmap(_project, (1, None))(A, x).sum(axis=0), jnp.zeros(shape=x.shape))


@chex.dataclass
class ORFitBel:
    mean: chex.Array
    basis: chex.Array
    svs: chex.Array

    
class RebayesORFit(Rebayes):
    def __init__(
        self,
        dynamics_weights: CovMat,
        dynamics_covariance: CovMat,
        emission_mean_function: Union[FnStateToEmission, FnStateAndInputToEmission],
        emission_cov_function: Union[FnStateToEmission2, FnStateAndInputToEmission2],
        emission_dist: EmissionDistFn = \
            lambda mean, cov: MVN(loc=mean, scale_tril=jnp.linalg.cholesky(cov)),
        memory_size: int = 10,
    ):
        super().__init__(dynamics_covariance, emission_mean_function, emission_cov_function, emission_dist)
        self.dynamics_weights = dynamics_weights
        self.memory_size = memory_size
    
    def init_bel(
        self,
        initial_mean: Float[Array, "state_dim"],
        initial_covariance: CovMat=None,
        X: Float[Array, "input_dim"]=None,
        y: Float[Array, "output_dim"]=None,
    ):
        init_basis = jnp.zeros((len(initial_mean), self.memory_size))
        init_svs = jnp.zeros((self.memory_size,))
        
        return ORFitBel(
            mean=initial_mean,
            basis=init_basis,
            svs=init_svs,
        )
    
    @partial(jit, static_argnums=(0,))
    def predict_obs(
        self,
        bel: ORFitBel,
        x: Float[Array, "input_dim"],
    ) -> Union[Float[Array, "output_dim"], Any]: 
        m = bel.mean
        m_Y = lambda z: self.emission_mean_function(z, x)
        y_pred = jnp.atleast_1d(m_Y(m))
        
        return y_pred
    
    @partial(jit, static_argnums=(0,))
    def predict_obs_cov(
        self,
        bel: ORFitBel,
        x: Float[Array, "input_dim"],
    ) -> Union[Float[Array, "output_dim output_dim"], Any]: 
        m, U = bel.mean, bel.basis
        m_Y = lambda z: self.emission_mean_function(z, x)
        
        # Compute predicted observation covariance
        H = _jacrev_2d(m_Y, m)
        Sigma_obs = H @ H.T - (H @ U) @ (H @ U).T
        
        return Sigma_obs
    
    @partial(jit, static_argnums=(0,))
    def update_state(
        self,
        bel: ORFitBel,
        x: Float[Array, "input_dim"],
        y: Float[Array, "obs_dim"],
    ) -> ORFitBel:
        m, U, Lambda = bel.mean, bel.basis, bel.svs
        
        # Update the state
        m_cond, U_cond, Lambda_cond = _orfit_condition_on(
            m, U, Lambda, self.emission_mean_function, x, y
        )
        bel_cond =  bel.replace(
            mean = m_cond,
            basis = U_cond,
            svs = Lambda_cond,
        )
        
        return bel_cond
    
    @partial(jit, static_argnums=(0,))
    def sample_state(
        self,
        bel: ORFitBel,
        key: Array,
        n_samples: int=100
    ) -> Float[Array, "n_samples state_dim"]:
        bel = self.predict_state(bel)
        P, *_ = bel.mean.shape
        diag = jnp.ones((P,))
        shape = (n_samples,)
        params_sample = sample_dlr(key, bel.basis, diag, shape) + bel.mean
        
        return params_sample


def _orfit_condition_on(
    m: Float[Array, "state_dim"],
    U: Float[Array, "state_dim memory_size"],
    Lambda: Float[Array, "memory_size"],
    apply_fn: Callable,
    x: Float[Array, "input_dim"],
    y: Float[Array, "obs_dim"]
):
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

    # Update the basis and singular values
    u = _normalize(v_prime)
    U_cond = jnp.where(Lambda.min() < u @ v_prime, U.at[:, Lambda.argmin()].set(u), U)
    Lambda_cond = jnp.where(Lambda.min() < u @ v_prime, Lambda.at[Lambda.argmin()].set(u.T @ v_prime), Lambda)
    
    # Update the mean
    m_cond = m - _stable_division((f_fn(m) - y) * v_prime, v.T @ v_prime)

    return m_cond, U_cond, Lambda_cond