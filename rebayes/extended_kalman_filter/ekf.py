from typing import Callable, NamedTuple, Union

import chex
from functools import partial
import jax
from jax import jit, vmap
from jax import numpy as jnp
from jaxtyping import Array, Float

from rebayes.base import Rebayes
from rebayes.extended_kalman_filter.ekf_core import (
    _jacrev_2d,
    _ekf_estimate_noise,
    _full_covariance_dynamics_predict, 
    _full_covariance_condition_on,
    _diagonal_dynamics_predict,
    _variational_diagonal_ekf_condition_on,  
    _fully_decoupled_ekf_condition_on
)


FnStateToState = Callable[ [Float[Array, "state_dim"]], Float[Array, "state_dim"]]
FnStateToEmission = Callable[ [Float[Array, "state_dim"]], Float[Array, "emission_dim"]]
FnStateAndInputToEmission = Callable[ [Float[Array, "state_dim"], Float[Array, "input_dim"] ], Float[Array, "emission_dim"]]
FnStateToEmission2 = Callable[[Float[Array, "state_dim"]], Float[Array, "emission_dim emission_dim"]]
FnStateAndInputToEmission2 = Callable[[Float[Array, "state_dim"], Float[Array, "input_dim"]], Float[Array, "emission_dim emission_dim"]]
CovMat = Union[float, Float[Array, "dim"], Float[Array, "dim dim"]]


PREDICT_FNS = {
    "fcekf": _full_covariance_dynamics_predict,
    "fdekf": _diagonal_dynamics_predict,
    "vdekf": _diagonal_dynamics_predict,
}

UPDATE_FNS = {
    "fcekf": _full_covariance_condition_on,
    "fdekf": _fully_decoupled_ekf_condition_on,
    "vdekf": _variational_diagonal_ekf_condition_on,
}


# Helper functions
def _process_ekf_cov(cov, P, method):
    if method == "fcekf":
        if isinstance(cov, float):
            cov = cov * jnp.eye(P)
        elif cov.ndim == 1:
            cov = jnp.diag(cov)
    else:
        if isinstance(cov, float):
            cov = cov * jnp.ones(P)

    return cov


@chex.dataclass
class EKFBel:
    mean: chex.Array
    cov: chex.Array
    nobs: int=None
    obs_noise_var: float=None


class EKFParams(NamedTuple):
    initial_mean: Float[Array, "state_dim"]
    initial_covariance: CovMat
    dynamics_weights_or_function: Union[float, FnStateToState]
    dynamics_covariance: CovMat
    emission_mean_function: Union[FnStateToEmission, FnStateAndInputToEmission]
    emission_cov_function: Union[FnStateToEmission2, FnStateAndInputToEmission2]
    adaptive_emission_cov: bool = False
    dynamics_covariance_inflation_factor: float = 0.0


class RebayesEKF(Rebayes):
    def __init__(
        self,
        params: EKFParams,
        method: str,
    ):  
        self.m0, self.P0, self.d, self.Q0, self.h, self.R, self.ada_var, self.alpha = params
        self.method = method
        if method == "fcekf":
            if isinstance(self.d, float):
                gamma = self.d
                self.d = lambda x: gamma * x
        elif method in ["fdekf", "vdekf"]:
            assert isinstance(self.d, float) # Dynamics should be a scalar
        else:
            raise ValueError('unknown method ', method)        
        self.pred_fn, self.update_fn = PREDICT_FNS[method], UPDATE_FNS[method]
        self.P0, self.Q0 = \
            (_process_ekf_cov(cov, self.m0.shape[0], method) for cov in [self.P0, self.Q0])
        self.nobs, self.obs_noise_var = 0, 0.0

    def init_bel(self, Xinit=None, Yinit=None):
        bel = EKFBel(
            mean = self.m0,
            cov = self.P0,
            nobs = self.nobs,
            obs_noise_var = self.obs_noise_var,
        )
        if Xinit is not None: # warmup sequentially
            bel, _ = self.scan(Xinit, Yinit, bel=bel)
            
        return bel
       
    @partial(jit, static_argnums=(0,))
    def predict_state(self, bel):
        m, P = bel.mean, bel.cov
        m_pred, P_pred = self.pred_fn(m, P, self.d, self.Q0, self.alpha)    
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
    def predict_obs_cov(self, bel, u):
        m, P, obs_noise_var = bel.mean, bel.cov, bel.obs_noise_var
        m_Y = lambda z: self.h(z, u)
        Cov_Y = lambda z: self.R(z, u)
        H =  _jacrev_2d(m_Y, m)
        y_pred = jnp.atleast_1d(m_Y(m))
        C, *_ = y_pred.shape
        
        if self.ada_var:
            R = obs_noise_var * jnp.eye(C)
        else:
            R = jnp.atleast_2d(Cov_Y(m))
        if self.method == 'fcekf':
            V_epi = H @ P @ H.T
        else:
            V_epi = (P * H) @ H.T
        P_obs = V_epi + R
        
        return P_obs

    @partial(jit, static_argnums=(0,))
    def update_state(self, bel, u, y):
        m, P, nobs, obs_noise_var = bel.mean, bel.cov, bel.nobs, bel.obs_noise_var
        m_cond, P_cond = self.update_fn(m, P, self.h, self.R, u, y, 1, self.ada_var, obs_noise_var)
        nobs_cond, obs_noise_var_cond = _ekf_estimate_noise(m_cond, self.h, u, y, 
                                                            nobs, obs_noise_var,
                                                            self.ada_var)
        bel_cond = EKFBel(
            mean = m_cond,
            cov = P_cond,
            nobs = nobs_cond,
            obs_noise_var = obs_noise_var_cond
        )
        
        return bel_cond
    
    @partial(jit, static_argnums=(0,4))
    def pred_obs_mc(self, key, bel, x, shape=None):
        """
        Sample observations from the posterior predictive distribution.
        """
        shape = shape or (1,)
        # Belief posterior predictive.
        bel = self.predict_state(bel)
        if self.method != "fcekf":
            cov = jnp.diagflat(bel.cov)
        else:
            cov = bel.cov
        params_sample = jax.random.multivariate_normal(key, bel.mean, cov, shape)
        yhat_samples = vmap(self.h, (0, None))(params_sample, x)
        
        return yhat_samples
