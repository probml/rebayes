from functools import partial
from typing import NamedTuple, Union

import chex
from jax import jit
import jax.numpy as jnp
from jaxtyping import Array, Float

from rebayes.base import Rebayes
from rebayes.extended_kalman_filter.ekf import (
    CovMat,
    _process_ekf_cov,
    FnStateAndInputToEmission,
    FnStateAndInputToEmission2,
    FnStateToEmission,
    FnStateToEmission2,
    FnStateToState,
)
from rebayes.extended_kalman_filter.ekf_core import (
    _full_covariance_dynamics_predict,
    _full_covariance_condition_on,
    _swvakf_compute_auxiliary_matrices,
    _swvakf_estimate_noise
)

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


class SWEKFParams(NamedTuple):
    dim_input: int
    dim_output: int
    initial_mean: Float[Array, "state_dim"]
    initial_covariance: CovMat
    dynamics_function: FnStateToState
    dynamics_covariance: CovMat
    emission_mean_function: Union[FnStateToEmission, FnStateAndInputToEmission]
    emission_cov_function: Union[FnStateToEmission2, FnStateAndInputToEmission2]
    emission_covariance: CovMat = None
    adaptive_dynamics_cov: bool = True
    adaptive_emission_cov: bool = True
    covariance_inflation_factor: float = 1e-4
    window_length: int = 5


class RebayesSWEKF(Rebayes):
    def __init__(
        self,
        params: SWEKFParams,
    ):
        self.dim_input, self.dim_output, self.m0, self.P0, self.f, self.Q0, \
            self.h, self.r, self.R0, self.ada_dynamics_cov, self.ada_emission_cov, \
                rho_factor, self.L = params
        self.rho = 1 - rho_factor
        P, *_ = self.m0.shape
        self.P0, self.Q0 = (_process_ekf_cov(cov, P, "fcekf") for cov in (self.P0, self.Q0))
        self.R0 = _process_ekf_cov(self.R0, self.dim_output, "fcekf")
        assert self.L > 0 or not self.ada_dynamics_cov and not self.ada_emission_cov, \
            "Window length must be positive if adaptive covariances are used."
    
    def init_bel(self, Xinit=None, Yinit=None):
        P, *_ = self.m0.shape
        D, C = self.dim_input, self.dim_output
        mean_buffer, cov_buffer = jnp.zeros((self.L + 1, P)), jnp.zeros((self.L + 1, P, P))
        input_buffer, emission_buffer = jnp.zeros((self.L + 1, D)), jnp.zeros((self.L + 1, C))
        
        bel = SWEKFBel(
            mean = self.m0,
            cov = self.P0,
            dynamics_cov = self.Q0,
            dynamics_cov_scale = 0.0 * jnp.eye(P),
            emission_cov = self.R0,
            emission_cov_scale = 0.0 * jnp.eye(C),
            mean_buffer = mean_buffer,
            cov_buffer = cov_buffer,
            input_buffer = input_buffer,
            emission_buffer = emission_buffer,
        )
        if Xinit is not None:
            bel, _ = self.scan(Xinit, Yinit, bel=bel)
        
        return bel
    
    @partial(jit, static_argnums=(0,))
    def predict_state(self, bel):
        m, P, Q = bel.mean, bel.cov, bel.dynamics_cov
        m_pred, P_pred = _full_covariance_dynamics_predict(m, P, self.f, Q, 0.0)
        bel_pred = bel.replace(
            mean = m_pred,
            cov = P_pred,
        )
        
        return bel_pred
    
    @partial(jit, static_argnums=(0,))
    def predict_obs(self, bel, u):
        y_pred = jnp.atleast_1d(self.h(bel.mean, u))
        
        return y_pred

    @partial(jit, static_argnums=(0,))
    def update_state(self, bel, u, y):
        # KF Update
        m, P, R = bel.mean, bel.cov, bel.emission_cov
        m_cond, P_cond = \
            _full_covariance_condition_on(m, P, self.h, None, u, y, 1, True, R)
        P_cond = _make_symmetrical(P_cond)
        
        bel_cond = bel.replace(
            mean = m_cond,
            cov = P_cond,
        )
        
        # Covariance Estimation
        if self.ada_dynamics_cov or self.ada_emission_cov:
            bel_cond = bel_cond.apply_buffers(m_cond, P_cond, u, y)
            Q, q_nu, q_psi, R, r_nu, r_psi = \
                bel_cond.dynamics_cov, bel_cond.dynamics_cov_dof, bel_cond.dynamics_cov_scale, \
                    bel_cond.emission_cov, bel_cond.emission_cov_dof, bel_cond.emission_cov_scale
            m_prevs, P_prevs, u_prevs, y_prevs = \
                bel_cond.mean_buffer, bel_cond.cov_buffer, bel_cond.input_buffer, bel_cond.emission_buffer
            L_eff = jnp.minimum(self.L+1, bel_cond.counter)
            A, B = _swvakf_compute_auxiliary_matrices(self.f, Q, self.h, m_prevs, 
                                                    P_prevs, u_prevs, y_prevs, L_eff)
            
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
