from typing import Callable, Union

import chex
from functools import partial
import jax
from jax import jit, vmap
from jax.flatten_util import ravel_pytree
import jax.numpy as jnp
from jaxtyping import Array, Float
import tensorflow_probability.substrates.jax as tfp

from rebayes.base import (
    CovMat,
    EmissionDistFn,
    FnStateAndInputToEmission,
    FnStateAndInputToEmission2,
    FnStateToEmission,
    FnStateToEmission2,
    FnStateToState,
)
import rebayes.extended_kalman_filter.ekf_core as core
from rebayes.extended_kalman_filter.ekf import RebayesEKF


tfd = tfp.distributions
MVN = tfd.MultivariateNormalTriL
MVD = tfd.MultivariateNormalDiag


@chex.dataclass
class ReplayEKFBel:
    mean: chex.Array
    cov: chex.Array


class RebayesReplayEKF(RebayesEKF):
    def __init__(
        self,
        dynamics_weights_or_function: Union[float, FnStateToState],
        dynamics_covariance: CovMat,
        emission_mean_function: Union[FnStateToEmission, FnStateAndInputToEmission],
        emission_cov_function: Union[FnStateToEmission2, FnStateAndInputToEmission2],
        emission_dist: EmissionDistFn = \
            lambda mean, cov: MVN(loc=mean, scale_tril=jnp.linalg.cholesky(cov)),
        log_likelihood_input_processing_fn: Callable=lambda y: y,
        dynamics_covariance_inflation_factor: float = 0.0,
        method: str="fcekf",
        n_replay: int=10,
        learning_rate: float=0.01,
    ):  
        super().__init__(
            dynamics_weights_or_function, dynamics_covariance, 
            emission_mean_function, emission_cov_function, emission_dist, False,
            dynamics_covariance_inflation_factor, method
        )
        self.log_likelihood = lambda params, x, y: \
            emission_dist(self.emission_mean_function(params, x),
                          self.emission_cov_function(params, x)).log_prob(
                              log_likelihood_input_processing_fn(y)
                            )
        self.n_replay = n_replay
        self.learning_rate = learning_rate
        
    def _update_mean(
        self,
        bel: ReplayEKFBel,
        m_prev: Float[Array, "state_dim"],
        x: Float[Array, "input_dim"],
        y: Float[Array, "output_dim"],
    ) -> ReplayEKFBel:
        m, P = bel.mean, bel.cov
        gll = jax.grad(self.log_likelihood, argnums=0)(m, x, y)
        m_cond = m - self.learning_rate * (m - m_prev + P @ gll)
        bel_cond = bel.replace(mean=m_cond)
        
        return bel_cond
        
    def _update_cov(
        self,
        bel: ReplayEKFBel,
        x: Float[Array, "input_dim"],
        y: Float[Array, "output_dim"],
    ) -> ReplayEKFBel:
        m_prev, P_prev = bel.mean, bel.cov
        m_Y = lambda m: self.emission_mean_function(m, x)
        Cov_Y = lambda m: self.emission_cov_function(m, x)
        R = jnp.atleast_2d(Cov_Y(m_prev))
        H = core._jacrev_2d(m_Y, m_prev)
        S = R + (H @ P_prev @ H.T)
        C = P_prev @ H.T
        K = jnp.linalg.lstsq(S, C.T)[0].T
        P_cond = P_prev - K @ S @ K.T
        bel_cond = bel.replace(cov=P_cond)
        
        return bel_cond

    @partial(jit, static_argnums=(0,))
    def update_state(
        self, 
        bel: ReplayEKFBel,
        x: Float[Array, "input_dim"],
        y: Float[Array, "output_dim"],
    ) -> ReplayEKFBel:
        m_prev = bel.mean
        
        def partial_step(_, bel):
            bel = self._update_mean(bel, m_prev, x, y)
            return bel
        bel = jax.lax.fori_loop(0, self.n_replay-1, partial_step, bel)
        bel = self._update_mean(bel, m_prev, x, y)
        bel_cond = self._update_cov(bel, x, y)
        
        return bel_cond