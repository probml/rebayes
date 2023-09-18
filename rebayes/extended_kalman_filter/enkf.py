from typing import Callable, Union

import chex
from functools import partial
import jax
from jax import jit, vmap
from jax.flatten_util import ravel_pytree
import jax.numpy as jnp
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


@chex.dataclass
class EnKFBel:
    ensemble: chex.Array
    mean: chex.Array
    perturbations: chex.Array
    key: chex.Array
    nobs: int=None
    obs_noise_var: float=None
    
    
def _process_enkf_cov(cov, P):
    if isinstance(cov, float):
        cov = cov * jnp.ones(P)

    return cov


def _compute_ensemble_mean_and_perturbations(ensemble):
    mean = jnp.mean(ensemble, axis=0)
    perturbations = (ensemble - mean).T
    
    return mean, perturbations


class RebayesEnKF(Rebayes):
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
        n_particles: int = 10,
        key: int = 0,
    ):  
        super().__init__(dynamics_covariance, emission_mean_function, 
                         emission_cov_function, emission_dist)
        self.dynamics_weights = dynamics_weights_or_function
        self.adaptive_emission_cov = adaptive_emission_cov
        self.dynamics_covariance_inflation_factor = \
            dynamics_covariance_inflation_factor
        self.n_particles = n_particles
        if isinstance(key, int):
            key = jr.PRNGKey(key)
        self.key = key
        assert isinstance(self.dynamics_weights, float)
        
        # self.pred_fn, self.update_fn = PREDICT_FNS[method], UPDATE_FNS[method]
        self.nobs, self.obs_noise_var = 0, 0.0

    def init_bel(
        self,
        initial_mean: Float[Array, "state_dim"],
        initial_covariance: CovMat,
        Xinit: Float[Array, "input_dim"]=None,
        Yinit: Float[Array, "output_dim"]=None,
    ) -> EnKFBel:
        P, *_ = initial_mean.shape
        P0 = _process_enkf_cov(initial_covariance, P)
        self.dynamics_covariance = \
            _process_enkf_cov(self.dynamics_covariance, P)
        ensemble = MVD(loc=initial_mean, scale_diag=jnp.sqrt(P0)).sample(
            seed=self.key, sample_shape=(self.n_particles,)
        )
        mean, perturbations = _compute_ensemble_mean_and_perturbations(ensemble)
        bel = EnKFBel(
            ensemble = ensemble,
            mean = mean,
            perturbations = perturbations,
            key = self.key,
            nobs = self.nobs,
            obs_noise_var = self.obs_noise_var,
        )
        if Xinit is not None: # warmup sequentially
            bel, _ = self.scan(Xinit, Yinit, bel=bel)
            
        return bel

    @partial(jit, static_argnums=(0,))
    def predict_obs(
        self, 
        bel: EnKFBel, 
        x: Float[Array, "input_dim"],
    ) -> Float[Array, "output_dim"]:
        m = bel.mean
        m_Y = lambda z: self.emission_mean_function(z, x)
        y_pred = jnp.atleast_1d(m_Y(m))
        
        return y_pred
    
    @partial(jit, static_argnums=(0,4))
    def predict_obs_cov(
        self, 
        bel: EnKFBel,
        x: Float[Array, "input_dim"],
        aleatoric_factor: float = 1.0,
        apply_fn: Callable = None,
    ) -> Float[Array, "output_dim output_dim"]:
        m, pert = bel.mean, bel.perturbations
        if apply_fn is None:
            m_Y = lambda z: self.emission_mean_function(z, x)
        else:
            m_Y = lambda z: apply_fn(z, x)
        H =  core._jacrev_2d(m_Y, m)
        HP = H @ pert
        V_epi = HP @ HP.T / (self.n_particles - 1)
        R = self.obs_cov(bel, x) * aleatoric_factor
        P_obs = V_epi + R
        
        return P_obs
    
    @partial(jit, static_argnums=(0,))
    def predict_state(
        self, 
        bel: EnKFBel,
    ) -> EnKFBel:
        ens, key = bel.ensemble, bel.key
        key, subkey = jr.split(key)
        ens_pred = core._ensemble_predict(
            key, ens, self.dynamics_weights, self.dynamics_covariance,
        )
        mean_pred, pert_pred = _compute_ensemble_mean_and_perturbations(ens_pred)
        bel_pred = bel.replace(
            ensemble = ens_pred,
            mean = mean_pred,
            perturbations = pert_pred,
            key = subkey,
        )
        
        return bel_pred

    @partial(jit, static_argnums=(0,))
    def update_state(
        self, 
        bel: EnKFBel,
        x: Float[Array, "input_dim"],
        y: Float[Array, "output_dim"],
    ) -> EnKFBel:
        ens, key = bel.ensemble, bel.key
        nobs, obs_noise_var = bel.nobs, bel.obs_noise_var
        
        key, subkey = jr.split(key)
        ens_cond = core._ensemble_stochastic_condition_on(
            key, ens, self.emission_mean_function, self.emission_cov_function,
            x, y, self.adaptive_emission_cov, obs_noise_var,
        )
        mean_cond, pert_cond = _compute_ensemble_mean_and_perturbations(ens_cond)
        nobs_cond, obs_noise_var_cond = \
            core._ekf_estimate_noise(mean_cond, self.emission_mean_function, x, y, 
                                     nobs, obs_noise_var, self.adaptive_emission_cov)
        bel_cond = bel.replace(
            ensemble = ens_cond,
            mean = mean_cond,
            perturbations = pert_cond,
            key = subkey,
            nobs = nobs_cond,
            obs_noise_var = obs_noise_var_cond
        )
        
        return bel_cond
    
    @partial(jit, static_argnums=(0,3))
    def sample_state(
        self, 
        bel: EnKFBel,
        key: Array, 
        n_samples: int=100,
        temperature: float=1.0,
    ) -> Float[Array, "n_samples state_dim"]:
        pass
    
    @partial(jit, static_argnums=(0,4,))
    def pred_obs_mc(
        self, 
        bel: EnKFBel,
        key: Array,
        x: Float[Array, "input_dim"],
        n_samples: int=1
    ) -> Float[Array, "n_samples output_dim"]:
        """
        Sample observations from the posterior predictive distribution.
        """
        pass
