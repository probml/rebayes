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

from abc import ABC
from abc import abstractmethod
from functools import partial

import jax.numpy as jnp
from jax import jacrev, jit
from jax.lax import scan
from jaxtyping import Float, Array
from typing import Callable, NamedTuple, Union, Tuple, Any
import chex
from jax_tqdm import scan_tqdm
from itertools import cycle

_jacrev_2d = lambda f, x: jnp.atleast_2d(jacrev(f)(x))

import tensorflow_probability.substrates.jax as tfp
tfd = tfp.distributions
MVN = tfd.MultivariateNormalFullCovariance


FnStateToState = Callable[ [Float[Array, "state_dim"]], Float[Array, "state_dim"]]
FnStateAndInputToState = Callable[ [Float[Array, "state_dim"], Float[Array, "input_dim"]], Float[Array, "state_dim"]]
FnStateToEmission = Callable[ [Float[Array, "state_dim"]], Float[Array, "emission_dim"]]
FnStateAndInputToEmission = Callable[ [Float[Array, "state_dim"], Float[Array, "input_dim"] ], Float[Array, "emission_dim"]]

FnStateToEmission2 = Callable[[Float[Array, "state_dim"]], Float[Array, "emission_dim emission_dim"]]
FnStateAndInputToEmission2 = Callable[[Float[Array, "state_dim"], Float[Array, "input_dim"]], Float[Array, "emission_dim emission_dim"]]
EmissionDistFn = Callable[ [Float[Array, "state_dim"], Float[Array, "state_dim state_dim"]], tfd.Distribution]

CovMat = Union[float, Float[Array, "dim"], Float[Array, "dim dim"]]


RebayesEstimator = namedtuple("RebayesEstimator", ["init", "predict_state", "update_state", "predict_obs", "update_params"])

@chex.dataclass
class Gaussian:
    mean: chex.Array
    cov: chex.Array

@chex.dataclass
class Belief:
    dummy: float
    # The belief state can be a Gaussian or some other representation (eg samples)
    # This must be a chex dataclass so that it works with lax.scan as a return type for carry

@chex.dataclass
class RebayesParams:
    initial_mean: Float[Array, "state_dim"]
    initial_covariance: CovMat
    dynamics_weights: CovMat
    dynamics_covariance: CovMat
    emission_mean_function: FnStateAndInputToEmission
    emission_cov_function: FnStateAndInputToEmission2
    adaptive_emission_cov: bool=False
    dynamics_covariance_inflation_factor: float=0.0

def make_rebayes_params():
    # dummy constructor
    model_params = RebayesParams(
        initial_mean=None,
        initial_covariance=None,
        dynamics_weights=None,
        dynamics_covariance=None,
        emission_mean_function=None,
        emission_cov_function=None,
    ) 
    return model_params


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