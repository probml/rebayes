
#pytest test_base.py  -rP

import pytest
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
import time
from functools import partial

from jax import jit
from jax.lax import scan
from jaxtyping import Float, Array
from typing import Callable, NamedTuple, Union, Tuple, Any
import chex

import haiku as hk

from rebayes.base import RebayesParams, Rebayes, Belief, make_rebayes_params


class RebayesSum(Rebayes):
    """The belief state is the sum of all the input X_t values."""
    def __init__(
        self,
        params: RebayesParams,
        ndim_in: int, 
        ndim_out: int
    ):
        self.params = params
        self.ndim_in = ndim_in
        self.ndim_out = ndim_out

    def init_bel(self) -> Belief:
        bel = Belief(dummy = jnp.zeros((self.ndim_in,)))
        return bel
    
    def update_state(
        self,
        bel: Belief,
        X: Float[Array, "input_dim"],
        Y: Float[Array, "obs_dim"]
    ) -> Belief:
        return Belief(dummy = bel.dummy + X)

    
def make_data():
    keys = hk.PRNGSequence(42)
    ndim_in = 5
    nclasses = 10
    ntime = 12
    #X = jnp.arange(ntime).reshape((ntime, 1)) # 1d
    X = jr.normal(next(keys), (ntime, ndim_in))
    labels = jr.randint(next(keys), (ntime,), 0,  nclasses-1)
    Y = jax.nn.one_hot(labels, nclasses)
    return X, Y


def callback_scan(bel, pred_obs, t, X, Y, bel_pred, **kwargs):
    jax.debug.print("callback with t={t}", t=t)
    return t

def test_scan():
    print('test scan')
    X, Y = make_data()
    ndim_in = X.shape[1]
    ndim_out = Y.shape[1]
    estimator = RebayesSum(make_rebayes_params(), ndim_in, ndim_out)
    bel, outputs = estimator.scan(X, Y, callback=callback_scan, progress_bar=False)
    print('final belief ', bel)
    print('outputs ', outputs)
    Xsum = jnp.sum(X, axis=0)
    assert jnp.allclose(bel.dummy, Xsum)

def test_update_batch():
    print('test update batch')
    X, Y = make_data()
    ndim_in = X.shape[1]
    ndim_out = Y.shape[1]
    estimator = RebayesSum(make_rebayes_params(), ndim_in, ndim_out)
    Xsum = jnp.sum(X, axis=0)

    bel = estimator.init_bel()
    bel = estimator.update_state_batch(bel, X, Y)
    assert(jnp.allclose(bel.dummy, Xsum))

    bel = estimator.init_bel()
    N = X.shape[0]
    for n in range(N):
        bel = estimator.predict_state(bel)
        bel = estimator.update_state(bel, X[n], Y[n])
    assert(jnp.allclose(bel.dummy, Xsum))


