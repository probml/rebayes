
#pytest test_base.py  -rP

import pytest
import numpy as np
import jax.numpy as jnp
import jax.random as jr
import time

from functools import partial

import jax.numpy as jnp
from jax import jit
from jax.lax import scan
from jaxtyping import Float, Array
from typing import Callable, NamedTuple, Union, Tuple, Any
import chex

from rebayes.base import RebayesParams, Rebayes, Belief

class RebayesDummy(Rebayes):
    def __init__(
        self,
        params: RebayesParams,
        ndim_out: int
    ):
        self.params = params
        self.ndim_out = ndim_out

    def init_bel(self) -> Belief:
        bel = Belief(dummy = 0.0)
        return bel

    def predict_state(
        self,
        bel: Belief
    ) -> Belief:
        return bel

    def predict_obs(
        self,
        bel: Belief,
        X: Float[Array, "input_dim"]
    ) -> Float[Array, "output_dim"]: 
        Yhat = jnp.zeros((1, self.ndim_out))
        return Yhat

    def update_state(
        self,
        bel: Belief,
        X: Float[Array, "input_dim"],
        y: Float[Array, "obs_dim"]
    ) -> Belief:
        return Belief(dummy = bel.dummy + 1)
    

def make_params():
    model_params = RebayesParams(
        initial_mean=None,
        initial_covariance=None,
        dynamics_weights=None,
        dynamics_covariance=None,
        emission_mean_function=None,
        emission_cov_function=None,
    ) 
    return model_params

def callback(bel, pred_obs, t, X, y, **kwargs):
    print('belief at time ', t, ' is ', bel.dummy)
    return t

def test():
    ndim_in, ndim_out = 2, 10
    ntime = 6
    key_root = jr.PRNGKey(42)
    key, key_root = jr.split(key_root)
    X = jr.normal(key, (ntime, ndim_in))
    key, key_root = jr.split(key_root)
    Y = jr.normal(key, (ntime, ndim_out))
    print(X.shape, Y.shape)
    estimator = RebayesDummy(make_params(), ndim_out)
    bel, outputs = estimator.scan(X, Y, callback=callback, progress_bar=False)
    print('final belief ', bel)
    print('outputs ', outputs)
    
    