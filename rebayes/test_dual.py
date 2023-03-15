#pytest test_dual.py  -rP

from jaxtyping import Float, Array
from typing import Callable, NamedTuple, Union, Tuple, Any
from functools import partial
import chex
import optax
import jax
import jax.numpy as jnp
import jax.random as jr
from jax import lax, jacfwd, vmap, grad, jit
from jax.tree_util import tree_map, tree_reduce
from jax.flatten_util import ravel_pytree

import flax
import flax.linen as nn
from tensorflow_probability.substrates.jax.distributions import MultivariateNormalFullCovariance as MVN
import haiku as hk

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from dataclasses import dataclass
from collections import namedtuple
from itertools import cycle

from rebayes.base import RebayesParams, Rebayes, Belief, make_rebayes_params


RebayesEstimator = namedtuple("RebayesEstimator", ["init", "predict_state", "update_state", "predict_obs", "update_params"])

def rebayes_scan(
        estimator,
        X: Float[Array, "ntime input_dim"],
        Y: Float[Array, "ntime emission_dim"],
        callback=None
    ) -> Tuple[Belief, Any]:
        """Apply filtering to entire sequence of data. Return final belief state and outputs from callback."""
        num_timesteps = X.shape[0]
        def step(carry, t):
            params, bel = carry
            pred_bel = estimator.predict_state(params, bel)
            pred_obs = estimator.predict_obs(params, bel, X[t])
            bel = estimator.update_state(params, pred_bel, X[t], Y[t])
            params = estimator.update_params(params, t,  X[t], Y[t], pred_obs)
            out = None
            if callback is not None:
                out = callback(bel, pred_obs, t, X[t], Y[t], pred_bel)
            return (params, bel), out
        params, bel = estimator.init()
        carry, outputs = jax.lax.scan(step, (params, bel), jnp.arange(num_timesteps))
        return carry, outputs

def make_my_estimator(model_params, est_params):
    """The belief state is the sum of all the scaled input X_t values.
    The model parameters sets the dynamics covariance at time t to t."""

    ndim_in, ndim_out, scale_factor = est_params 

    def init():
        bel = Belief(dummy = jnp.zeros((ndim_in,)))
        return model_params, bel
    
    def predict_state(params, bel):
        return bel
    
    def update_state(params, bel, X, Y):
        return Belief(dummy = bel.dummy + scale_factor * X)
    
    def predict_obs(params, bel, X):
        return None
    
    def update_params(params, t, X, Y, Yhat):
        #jax.debug.print("t={t}", t=t)
        params.dynamics_covariance = t*1.0 # abritrary update
        return params
    
    return RebayesEstimator(init, predict_state, update_state, predict_obs, update_params)

def make_data():
    keys = hk.PRNGSequence(42)
    ndim_in = 5
    nclasses = 10
    ntime = 12
    X = jr.normal(next(keys), (ntime, ndim_in))
    labels = jr.randint(next(keys), (ntime,), 0,  nclasses-1)
    Y = jax.nn.one_hot(labels, nclasses)
    return X, Y

def test():
    X, Y = make_data()
    ntime = X.shape[0]
    ndim_in = X.shape[1]
    ndim_out = Y.shape[1]

    ssm_params = make_rebayes_params()
    ssm_params.dynamics_covariance = 0
    print(ssm_params)

    scale_factor = 2
    est_params = (ndim_in, ndim_out, scale_factor)
    estimator = make_my_estimator(ssm_params, est_params)

    carry, outputs = rebayes_scan(estimator,  X, Y)
    params, bel = carry
    print('final belief ', bel)
    print('final params ', params)
    print('outputs ', outputs)
    Xsum = jnp.sum(X, axis=0)
    assert jnp.allclose(bel.dummy, Xsum*scale_factor)
    assert jnp.allclose(params.dynamics_covariance, ntime-1)

