from typing import NamedTuple, Union

import chex
from functools import partial
import jax
from jax import jit, vmap
from jax import numpy as jnp
import jax.random as jr
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
import rebayes.extended_kalman_filter.ekf_core as core


tfd = tfp.distributions
MVN = tfd.MultivariateNormalTriL
MVD = tfd.MultivariateNormalDiag


PREDICT_FNS = {
    "fcekf": core._full_covariance_dynamics_predict,
    "fdekf": core._diagonal_dynamics_predict,
    "vdekf": core._diagonal_dynamics_predict,
}

UPDATE_FNS = {
    "fcekf": core._full_covariance_condition_on,
    "fdekf": core._fully_decoupled_ekf_condition_on,
    "vdekf": core._variational_diagonal_ekf_condition_on,
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


class RebayesEKF(Rebayes):
    def __init__(
        self,
        dynamics_weights_or_function: Union[float, FnStateToState],
        dynamics_covariance: CovMat,
        emission_mean_function: Union[FnStateToEmission, FnStateAndInputToEmission],
        emission_cov_function: Union[FnStateToEmission2, FnStateAndInputToEmission2],
        emission_dist: EmissionDistFn = \
            lambda mean, cov: MVN(loc=mean, scale_tril=jnp.linalg.cholesky(cov)),
        adaptive_emission_cov: bool = False,
        dynamics_covariance_inflation_factor: float = 0.0,
        method: str="fcekf",
    ):  
        super().__init__(dynamics_covariance, emission_mean_function, emission_cov_function, emission_dist)
        self.dynamics_weights = dynamics_weights_or_function
        self.adaptive_emission_cov = adaptive_emission_cov
        self.dynamics_covariance_inflation_factor = dynamics_covariance_inflation_factor
        
        self.method = method
        if method == "fcekf":
            if isinstance(self.dynamics_weights, float):
                gamma = self.dynamics_weights
                self.dynamics_weights = lambda x: gamma * x
        elif method in ["fdekf", "vdekf"]:
            assert isinstance(self.dynamics_weights, float) # Dynamics should be a scalar
        else:
            raise ValueError('unknown method ', method)        
        self.pred_fn, self.update_fn = PREDICT_FNS[method], UPDATE_FNS[method]
        self.nobs, self.obs_noise_var = 0, 0.0

    def init_bel(
        self,
        initial_mean: Float[Array, "state_dim"],
        initial_covariance: CovMat,
        Xinit: Float[Array, "input_dim"]=None,
        Yinit: Float[Array, "output_dim"]=None,
    ) -> EKFBel:
        P, *_ = initial_mean.shape
        P0 = _process_ekf_cov(initial_covariance, P, self.method)
        bel = EKFBel(
            mean = initial_mean,
            cov = P0,
            nobs = self.nobs,
            obs_noise_var = self.obs_noise_var,
        )
        if Xinit is not None: # warmup sequentially
            bel, _ = self.scan(Xinit, Yinit, bel=bel)
            
        return bel

    @partial(jit, static_argnums=(0,))
    def predict_obs(
        self, 
        bel: EKFBel, 
        x: Float[Array, "input_dim"],
    ) -> Float[Array, "output_dim"]:
        m = bel.mean
        m_Y = lambda z: self.emission_mean_function(z, x)
        y_pred = jnp.atleast_1d(m_Y(m))
        
        return y_pred
    
    @partial(jit, static_argnums=(0,))
    def predict_obs_cov(
        self, 
        bel: EKFBel,
        x: Float[Array, "input_dim"],
    ) -> Float[Array, "output_dim output_dim"]:
        m, P = bel.mean, bel.cov
        m_Y = lambda z: self.emission_mean_function(z, x)
        H =  core._jacrev_2d(m_Y, m)
        if self.method == 'fcekf':
            V_epi = H @ P @ H.T
        else:
            V_epi = (P * H) @ H.T
        R = self.obs_cov(bel, x)
        P_obs = V_epi + R
        
        return P_obs
    
    @partial(jit, static_argnums=(0,))
    def predict_state(
        self, 
        bel: EKFBel,
    ) -> EKFBel:
        m, P = bel.mean, bel.cov
        dynamics_covariance = _process_ekf_cov(self.dynamics_covariance, m.shape[0], self.method)
        m_pred, P_pred = self.pred_fn(m, P, self.dynamics_weights, dynamics_covariance, 
                                      self.dynamics_covariance_inflation_factor)
        bel_pred = bel.replace(
            mean = m_pred,
            cov = P_pred,
        )
        return bel_pred

    @partial(jit, static_argnums=(0,))
    def update_state(
        self, 
        bel: EKFBel,
        x: Float[Array, "input_dim"],
        y: Float[Array, "output_dim"],
    ) -> EKFBel:
        m, P, nobs, obs_noise_var = bel.mean, bel.cov, bel.nobs, bel.obs_noise_var
        m_cond, P_cond = self.update_fn(m, P, self.emission_mean_function, 
                                        self.emission_cov_function, x, y, 
                                        1, self.adaptive_emission_cov, obs_noise_var)
        nobs_cond, obs_noise_var_cond = \
            core._ekf_estimate_noise(m_cond, self.emission_mean_function, x, y, 
                                     nobs, obs_noise_var, self.adaptive_emission_cov)
        bel_cond = EKFBel(
            mean = m_cond,
            cov = P_cond,
            nobs = nobs_cond,
            obs_noise_var = obs_noise_var_cond
        )
        
        return bel_cond
    
    @partial(jit, static_argnums=(0,3))
    def sample_state(
        self, 
        bel: EKFBel,
        key: Array, 
        n_samples: int=100,
        temperature: float=1.0,
    ) -> Float[Array, "n_samples state_dim"]:
        bel = self.predict_state(bel)
        shape = (n_samples,)
        cooled_cov = bel.cov * temperature
        if self.method != "fcekf":
            mvn = MVD(loc=bel.mean, scale_diag=jnp.sqrt(cooled_cov))
        else:
            mvn = MVN(loc=bel.mean, scale_tril=jnp.linalg.cholesky(cooled_cov))
        params_sample = mvn.sample(seed=key, sample_shape=shape)
        
        return params_sample
    
    @partial(jit, static_argnums=(0,4,))
    def pred_obs_mc(
        self, 
        bel: EKFBel,
        key: Array,
        x: Float[Array, "input_dim"],
        n_samples: int=1
    ) -> Float[Array, "n_samples output_dim"]:
        """
        Sample observations from the posterior predictive distribution.
        """
        params_sample = self.sample_state(bel, key, n_samples)
        yhat_samples = vmap(self.emission_mean_function, (0, None))(params_sample, x)
        
        return yhat_samples
