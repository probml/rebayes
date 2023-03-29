from functools import partial
from typing import Callable, Literal

import chex
from flax.training.train_state import TrainState
from jax import jacrev, jit
import jax.numpy as jnp
import jax.random as jr

from rebayes.base import Gaussian
from rebayes.dual_base import (
    DualRebayesParams, 
    ObsModel, 
    RebayesEstimator,
    form_tril_matrix,
)
from rebayes.extended_kalman_filter.ekf_core import (
    _full_covariance_dynamics_predict, 
    _diagonal_dynamics_predict,
    _full_covariance_condition_on, 
    _variational_diagonal_ekf_condition_on,
    _fully_decoupled_ekf_condition_on,
)


_jacrev_2d = lambda f, x: jnp.atleast_2d(jacrev(f)(x))

EKFMethods = Literal["fcekf", "vdekf", "fdekf"]


@chex.dataclass
class EKFParams:
    method: EKFMethods
    obs_noise_estimator: str = None
    obs_noise_lr_fn: Callable = None # fn(t) -> lr


def make_dual_ekf_estimator(params: DualRebayesParams, obs: ObsModel, ekf_params: EKFParams):

    def init():
        D = params.mu0.shape[0]
        if ekf_params.method == 'fcekf':
            cov = 1/params.eta0 * jnp.eye(D)
        else: # store diagonal cov as a vector
            cov = 1/params.eta0 * jnp.ones(D)
        bel =  Gaussian(mean=params.mu0, cov=cov)
        return params, bel
    
    def init_optimizer_params(tx, C, key=0):
        """Initialize optimizer parameters for the observation noise.

        Args:
            tx (optimizer): Optax optimizer.
            C (int): Emission dimension.
            key (int, optional): PRNG key. Defaults to 0.
        """
        if isinstance(key, int):
            key = jr.PRNGKey(key)
        
        params_bel = TrainState.create(
            apply_fn = obs.emission_mean_function,
            params = jr.normal(key, (int(C*(C+1)/2),)),
            tx = tx,
        )
        return params_bel
    
    @jit
    def predict_state(params, bel):
        m, P = bel.mean, bel.cov
        if ekf_params.method == 'fcekf':
            pred_mean, pred_cov = _full_covariance_dynamics_predict(m, P, params.dynamics_noise, params.dynamics_scale_factor, params.cov_inflation_factor)
        else:
            pred_mean, pred_cov = _diagonal_dynamics_predict(m, P, params.dynamics_noise, params.dynamics_scale_factor, params.cov_inflation_factor)
        return Gaussian(mean=pred_mean, cov=pred_cov)
    
    @jit
    def update_state(params, bel, X, Y):
        m, P = bel.mean, bel.cov
        if ekf_params.method == 'fcekf':
            update_fn = _full_covariance_condition_on
        elif ekf_params.method == 'vdekf':
            update_fn = _variational_diagonal_ekf_condition_on
        elif ekf_params.method == 'fdekf':
            update_fn = _fully_decoupled_ekf_condition_on
        adapt_obs_noise = (ekf_params.obs_noise_estimator is not None)
        mu, Sigma = update_fn(m, P, obs.emission_mean_function,
                            obs.emission_cov_function, X, Y, num_iter=1,
                            adaptive_variance=adapt_obs_noise,
                            obs_noise_var=params.obs_noise)
        return Gaussian(mean=mu, cov=Sigma)

    @jit
    def predict_obs(params, bel, X):
        prior_mean, prior_cov = bel.mean, bel.cov
        m_Y = lambda z: obs.emission_mean_function(z, X)
        y_pred = jnp.atleast_1d(m_Y(prior_mean))
        return y_pred

    @jit
    def predict_obs_cov(params, bel, X):
        prior_mean, prior_cov = bel.mean, bel.cov
        m_Y = lambda z: obs.emission_mean_function(z, X)
        H =  _jacrev_2d(m_Y, prior_mean)
        y_pred = jnp.atleast_1d(m_Y(prior_mean))
        if ekf_params.obs_noise_estimator is not None:
            R = jnp.eye(y_pred.shape[0]) * params.obs_noise
        else:
            R = jnp.atleast_2d(obs.emission_cov_function(prior_mean, X))
        if ekf_params.method == 'fcekf':
            V_epi = H @ prior_cov @ H.T
        else:
            V_epi = (prior_cov * H) @ H.T 
        Sigma_obs = V_epi + R
        return Sigma_obs

    @jit
    def update_params(params, t, X, y, ypred, bel):
        if ekf_params.obs_noise_estimator is None:
            return params
        nobs = params.nobs + 1
        #lr = ekf_params.obs_noise_var_lr/nobs # decay learning rate over time
        lr = ekf_params.obs_noise_lr_fn(t)
        if ekf_params.obs_noise_estimator == "post":
            # prediction after the belief update 
            yhat = obs.emission_mean_function(bel.mean, X)
        else:
            yhat = ypred
            # use yhat before the belief update

        yhat = jnp.atleast_1d(yhat)
        obs_noise = jnp.atleast_2d(params.obs_noise)
        sqerr = jnp.outer((yhat - y), (yhat - y)) / yhat.shape[0]
          
        r = (1-lr)*obs_noise + lr*sqerr
        obs_noise = jnp.where(jnp.linalg.norm(r) < 1e-6, 1e-6 * jnp.eye(r.shape[0]), r)
        params = params.replace(nobs = nobs, obs_noise = obs_noise)
        return params
    
    @partial(jit, static_argnums=(2,))
    def update_optimizer_params(params, params_bel, C):
        theta = params_bel.params
        L = form_tril_matrix(theta, C)
        params = params.replace(obs_noise = L @ L.T)
        return params
    
    return RebayesEstimator(init, init_optimizer_params, predict_state, update_state, 
                            predict_obs, predict_obs_cov, update_params, update_optimizer_params,)