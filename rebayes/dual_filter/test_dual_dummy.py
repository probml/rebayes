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

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from dataclasses import dataclass
from collections import namedtuple
from itertools import cycle

from rebayes.dual_filter.dual_estimator import rebayes_scan, RebayesEstimator, DummyBel, DualBayesParams, ObsModel, make_dual_bayes_params

def make_my_estimator(params: DualBayesParams, obs: ObsModel, est_params: Any):
    """The belief state is the sum of all the scaled input X_t values.
    The model parameters sets the dynamics covariance at time t to t."""

    del obs # ignored
    ndim_in, ndim_out, scale_factor = est_params 

    def init():
        bel = DummyBel(dummy = jnp.zeros((ndim_in,)))
        return params, bel
    
    def predict_state(params, bel):
        return bel
    
    def update_state(params, bel, X, Y):
        return DummyBel(dummy = bel.dummy + scale_factor * X)
    
    def predict_obs(params, bel, X):
        return None

    def predict_obs_cov(params, bel, X):
        return None
    
    def update_params(params, t, X, Y, Yhat, bel):
        #jax.debug.print("t={t}", t=t)
        params.q = t*1.0 # abritrary update
        return params
    
    return RebayesEstimator(init, predict_state, update_state, predict_obs, predict_obs_cov, update_params)

def make_data():
    keys = jr.split(jr.PRNGKey(0), 2)
    ndim_in = 5
    nclasses = 10
    ntime = 12
    X = jr.normal(keys[0], (ntime, ndim_in))
    labels = jr.randint(keys[1], (ntime,), 0,  nclasses-1)
    Y = jax.nn.one_hot(labels, nclasses)
    return X, Y

def test():
    X, Y = make_data()
    ntime = X.shape[0]
    ndim_in = X.shape[1]
    ndim_out = Y.shape[1]

    scale_factor = 2
    est_params = (ndim_in, ndim_out, scale_factor)
    params, obs = make_dual_bayes_params()
    params.q = 0
    estimator = make_my_estimator(params, obs, est_params)

    carry, outputs = rebayes_scan(estimator,  X, Y)
    params, bel = carry
    print('final belief ', bel)
    print('final params ', params)
    print('outputs ', outputs)
    Xsum = jnp.sum(X, axis=0)
    assert jnp.allclose(bel.dummy, Xsum*scale_factor)
    assert jnp.allclose(params.q, ntime-1)

if __name__ == "__main__":
    test()