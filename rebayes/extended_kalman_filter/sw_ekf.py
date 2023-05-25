from functools import partial
from typing import Union

import chex
from jax import jit, vmap
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
    Rebayes,
)
from rebayes.extended_kalman_filter.ekf import _process_ekf_cov
from rebayes.extended_kalman_filter.ekf_core import (
    _jacrev_2d,
    _full_covariance_dynamics_predict,
    _full_covariance_condition_on,
    _swvakf_compute_auxiliary_matrices,
    _swvakf_estimate_noise
)

tfd = tfp.distributions
MVN = tfd.MultivariateNormalTriL

_make_symmetrical = lambda x: (x + x.T) / 2


@chex.dataclass
class SWEKFBel:
    mean: chex.Array
    cov: chex.Array
    dynamics_cov: chex.Array
    dynamics_cov_scale: chex.Array
    emission_cov: chex.Array
    emission_cov_scale: chex.Array
    mean_buffer: chex.Array
    cov_buffer: chex.Array
    input_buffer: chex.Array
    emission_buffer: chex.Array
    dynamics_cov_dof: float = 0.0
    emission_cov_dof: float = 0.0
    counter: int = 0
    
    def _update_buffer(self, buffer, elt):
        buffer_new = jnp.concatenate([buffer[1:], jnp.expand_dims(elt, 0)], axis=0)

        return buffer_new
    
    def apply_buffers(self, m, P, u, y):
        mean_buffer = self._update_buffer(self.mean_buffer, m)
        cov_buffer = self._update_buffer(self.cov_buffer, P)
        input_buffer = self._update_buffer(self.input_buffer, u)
        emission_buffer = self._update_buffer(self.emission_buffer, y)
        
        return self.replace(
            mean_buffer = mean_buffer,
            cov_buffer = cov_buffer,
            input_buffer = input_buffer,
            emission_buffer = emission_buffer,
            counter = self.counter + 1
        )


class RebayesSWEKF(Rebayes):
    def __init__(
        self,
        dim_input: int,
        dim_output: int,
        dynamics_weights_or_function: Union[float, FnStateToState],
        dynamics_covariance: CovMat,
        emission_mean_function: Union[FnStateToEmission, FnStateAndInputToEmission],
        emission_cov_function: Union[FnStateToEmission2, FnStateAndInputToEmission2],
        emission_dist: EmissionDistFn = \
            lambda mean, cov: MVN(loc=mean, scale_tril=jnp.linalg.cholesky(cov)),
        emission_covariance: CovMat = None,
        adaptive_dynamics_cov: bool = True, # Whether or not to adapt dynamics covariance
        adaptive_emission_cov: bool = True, # Whether or not to adapt emission covariance
        covariance_inflation_factor: float = 1e-4, # Covariance inflation factor
        window_length: int = 5, # Window length for adaptive covariances
    ):
        super().__init__(dynamics_covariance, emission_mean_function, emission_cov_function, emission_dist)
        self.dim_input, self.dim_output = dim_input, dim_output
        self.dynamics_weights = dynamics_weights_or_function
        if isinstance(dynamics_weights_or_function, float):
            self.dynamics_weights = lambda x: dynamics_weights_or_function * x
        self.emission_covariance = _process_ekf_cov(emission_covariance, dim_output, "fcekf")
        self.ada_dynamics_cov, self.ada_emission_cov = adaptive_dynamics_cov, adaptive_emission_cov
        self.rho = 1 - covariance_inflation_factor
        self.window_length = window_length
    
    def init_bel(
        self, 
        initial_mean: Float[Array, "state_dim"],
        initial_covariance: CovMat,
        Xinit: Float[Array, "input_dim"]=None,
        Yinit: Float[Array, "output_dim"]=None,
    ) -> SWEKFBel:
        P, *_ = initial_mean.shape
        P0, Q0 = (_process_ekf_cov(cov, P, "fcekf") 
                  for cov in (initial_covariance, self.dynamics_covariance))
        D, C, L = self.dim_input, self.dim_output, self.window_length
        mean_buffer, cov_buffer = jnp.zeros((L + 1, P)), jnp.zeros((L + 1, P, P))
        input_buffer, emission_buffer = jnp.zeros((L + 1, D)), jnp.zeros((L + 1, C))
        
        bel = SWEKFBel(
            mean = initial_mean,
            cov = P0,
            dynamics_cov = Q0,
            dynamics_cov_scale = 0.0 * jnp.eye(P),
            emission_cov = self.emission_covariance,
            emission_cov_scale = 0.0 * jnp.eye(C),
            mean_buffer = mean_buffer,
            cov_buffer = cov_buffer,
            input_buffer = input_buffer,
            emission_buffer = emission_buffer,
        )
        if Xinit is not None:
            bel, _ = self.scan(initial_mean, initial_covariance, Xinit, Yinit, bel=bel)
        
        return bel
    
    @partial(jit, static_argnums=(0,))
    def predict_obs(
        self, 
        bel: SWEKFBel, 
        x: Float[Array, "input_dim"],
    ) -> Float[Array, "output_dim"]:
        y_pred = jnp.atleast_1d(self.emission_mean_function(bel.mean, x))
        
        return y_pred
    
    @partial(jit, static_argnums=(0,))
    def predict_obs_cov(
        self, 
        bel: SWEKFBel,
        x: Float[Array, "input_dim"],
    ) -> Float[Array, "output_dim output_dim"]:
        m, P = bel.mean, bel.cov
        m_Y = lambda z: self.emission_mean_function(z, x)
        H =  _jacrev_2d(m_Y, m)
        V_epi = H @ P @ H.T
        if self.ada_emission_cov:
            R = bel.emission_cov
        else:
            R = jnp.atleast_2d(self.emission_cov_function(m, x))
        P_obs = V_epi + R
        
        return P_obs
    
    @partial(jit, static_argnums=(0,))
    def predict_state(
        self, 
        bel: SWEKFBel,
    ) -> SWEKFBel:
        m, P, Q = bel.mean, bel.cov, bel.dynamics_cov
        m_pred, P_pred = _full_covariance_dynamics_predict(m, P, self.dynamics_weights, Q, 0.0)
        bel_pred = bel.replace(
            mean = m_pred,
            cov = P_pred,
        )
        
        return bel_pred

    @partial(jit, static_argnums=(0,))
    def update_state(
        self, 
        bel: SWEKFBel, 
        x: Float[Array, "input_dim"],
        y: Float[Array, "output_dim"],
    ) -> SWEKFBel:
        # KF Update
        m, P, R = bel.mean, bel.cov, bel.emission_cov
        f, h, L = self.dynamics_weights, self.emission_mean_function, self.window_length
        m_cond, P_cond = \
            _full_covariance_condition_on(m, P, h, None, x, y, 1, True, R)
        P_cond = _make_symmetrical(P_cond)
        
        bel_cond = bel.replace(
            mean = m_cond,
            cov = P_cond,
        )
        
        # Covariance Estimation
        if self.ada_dynamics_cov or self.ada_emission_cov:
            bel_cond = bel_cond.apply_buffers(m_cond, P_cond, x, y)
            Q, q_nu, q_psi, R, r_nu, r_psi = \
                bel_cond.dynamics_cov, bel_cond.dynamics_cov_dof, bel_cond.dynamics_cov_scale, \
                    bel_cond.emission_cov, bel_cond.emission_cov_dof, bel_cond.emission_cov_scale
            m_prevs, P_prevs, x_prevs, y_prevs = \
                bel_cond.mean_buffer, bel_cond.cov_buffer, bel_cond.input_buffer, bel_cond.emission_buffer
            L_eff = jnp.minimum(L+1, bel_cond.counter)
            A, B = _swvakf_compute_auxiliary_matrices(f, Q, h, m_prevs, P_prevs, 
                                                      x_prevs, y_prevs, L_eff)
            
            Q_cond, q_nu_cond, q_psi_cond, R_cond, r_nu_cond, r_psi_cond = \
                _swvakf_estimate_noise(Q, q_nu, q_psi, R, r_nu, r_psi, A, B, L_eff, 
                                    self.rho, bel_cond.counter)
            if not self.ada_dynamics_cov:
                Q_cond = Q
            if not self.ada_emission_cov:
                R_cond = R
            
            bel_cond = bel_cond.replace(
                dynamics_cov = Q_cond,
                dynamics_cov_scale = q_psi_cond,
                dynamics_cov_dof = q_nu_cond,
                emission_cov = R_cond,
                emission_cov_scale = r_psi_cond,
                emission_cov_dof = r_nu_cond,
            )
        
        return bel_cond

    @partial(jit, static_argnums=(0,3))
    def sample_state(
        self, 
        bel: SWEKFBel,
        key: Array, 
        n_samples: int=100
    ) -> Float[Array, "n_samples state_dim"]:
        bel = self.predict_state(bel)
        shape = (n_samples,)
        mvn = MVN(loc=bel.mean, scale_tril=jnp.linalg.cholesky(bel.cov))
        params_sample = mvn.sample(seed=key, sample_shape=shape)
        
        return params_sample
    
    @partial(jit, static_argnums=(0,4))
    def pred_obs_mc(
        self,
        bel: SWEKFBel,
        key,
        x: Float[Array, "input_dim"], 
        n_samples: int=1,
    ) -> Float[Array, "n_samples output_dim"]:
        """
        Sample observations from the posterior predictive distribution.
        """
        params_sample = self.sample_state(bel, key, n_samples)
        yhat_samples = vmap(self.emission_mean_function, (0, None))(params_sample, x)
        
        return yhat_samples
    