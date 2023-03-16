

from jaxtyping import Float, Array
from typing import Callable, NamedTuple, Union, Tuple, Any, Literal
from functools import partial
import chex
import optax
import jax
import jax.numpy as jnp
import jax.random as jr
from jax import lax, jacfwd, vmap, grad, jit, jacrev
from jax.tree_util import tree_map, tree_reduce
from jax.flatten_util import ravel_pytree

import flax
import flax.linen as nn
from tensorflow_probability.substrates.jax.distributions import MultivariateNormalFullCovariance as MVN

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from dataclasses import dataclass
from collections import namedtuple
from itertools import cycle

from rebayes.dual_filter.dual_estimator import RebayesHParams, RebayesObsModel, GaussBel, RebayesEstimator
from rebayes.extended_kalman_filter.ekf import _full_covariance_dynamics_predict, _diagonal_dynamics_predict
from rebayes.extended_kalman_filter.ekf import _full_covariance_condition_on,  _variational_diagonal_ekf_condition_on,  _fully_decoupled_ekf_condition_on

_jacrev_2d = lambda f, x: jnp.atleast_2d(jacrev(f)(x))

EKFMethods = Literal["fcekf", "vdekf", "fdekf"]

def make_dual_ekf_estimator(params: RebayesHParams, obs: RebayesObsModel, method: EKFMethods):
    if obs.emission_cov_function is None:
        scalar_obs_var = True # value is stored in params.r
    else:
        scalar_obs_var = False

    def init():
        D = params.mu0.shape[0]
        if method == 'fcekf':
            cov = 1/params.eta0 * jnp.eye(D)
        else: # store diagonal cov as a vector
            cov = 1/params.eta0 * jnp.ones(D)
        bel =  GaussBel(mean=params.mu0, cov=cov)
        return params, bel
    
    def predict_state(params, bel):
        m, P = bel.mean, bel.cov
        if method == 'fcekf':
            pred_mean, pred_cov = _full_covariance_dynamics_predict(m, P, params.q, params.gamma, params.alpha)
        else:
            pred_mean, pred_cov = _diagonal_dynamics_predict(m, P, params.q, params.gamma, params.alpha)
        return GaussBel(mean=pred_mean, cov=pred_cov)
    
    def update_state(params, bel, X, Y):
        m, P = bel.mean, bel.cov
        if method == 'fcekf':
            update_fn = _full_covariance_condition_on
        elif method == 'vdekf':
            update_fn = _variational_diagonal_ekf_condition_on
        elif method == 'fdekf':
            update_fn = _fully_decoupled_ekf_condition_on
        mu, Sigma = update_fn(m, P, obs.emission_mean_function,
                            obs.emission_cov_function, X, Y, 
                            num_iter=1, adaptive_variance=scalar_obs_var, obs_noise_var=params.r)
        return GaussBel(mean=mu, cov=Sigma)
    
    def predict_obs(params, bel, X):
        prior_mean, prior_cov = bel.mean, bel.cov
        m_Y = lambda z: obs.emission_mean_function(z, X)
        y_pred = jnp.atleast_1d(m_Y(prior_mean))
        return y_pred

    def predict_obs_cov(params, bel, X):
        prior_mean, prior_cov = bel.mean, bel.cov
        m_Y = lambda z: obs.emission_mean_function(z, X)
        H =  _jacrev_2d(m_Y, prior_mean)
        y_pred = jnp.atleast_1d(m_Y(prior_mean))
        if scalar_obs_var:
            R = jnp.eye(y_pred.shape[0]) * params.r
        else:
            R = jnp.atleast_2d(obs.emission_cov_function(prior_mean, X))
        if method == 'fcekf':
            V_epi = H @ prior_cov @ H.T
        else:
            V_epi = (prior_cov * H) @ H.T
        Sigma_obs = V_epi + R
        return Sigma_obs


    
    def update_params(params, t, X, Y, Yhat):
        #jax.debug.print("t={t}", t=t)
        return params
    
    return RebayesEstimator(init, predict_state, update_state, predict_obs, predict_obs_cov, update_params)