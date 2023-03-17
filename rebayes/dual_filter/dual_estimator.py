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



import matplotlib.pyplot as plt
import matplotlib.cm as cm
from dataclasses import dataclass
from collections import namedtuple
from itertools import cycle

from functools import partial
import jax.numpy as jnp
from jax import jacrev, jit
from jax.lax import scan
from jaxtyping import Float, Array
from typing import Callable, NamedTuple, Union, Tuple, Any
import chex
from jax_tqdm import scan_tqdm
from itertools import cycle

RebayesEstimator = namedtuple("RebayesEstimator", ["init", "predict_state", "update_state", "predict_obs", "predict_obs_cov", "update_params"])

@chex.dataclass
class DummyBel: # for debugging
    dummy: float

@chex.dataclass
class GaussBel:
    mean: chex.Array
    cov: chex.Array

@chex.dataclass
class DualBayesParams:
    mu0: chex.Array
    eta0: float
    dynamics_scale_factor: float = 1.0 # gamma
    dynamics_noise: float = 0.0 # Q
    obs_noise: float = 1.0 # R
    cov_inflation_factor: float = 0.0 # alpha  
    nobs: int = 0 # counts number of observations seen so far (for adaptive estimation)

# immutable set of functions for observation model
ObsModel = namedtuple("ObsModel", ["emission_mean_function", "emission_cov_function"])

def make_dual_bayes_params():
    # dummy constructor
    params = DualBayesParams(mu0=None, eta0=None, gamma=None, q=None, obs_noise_var=None, alpha=None, nobs=None)
    obs = ObsModel(emission_mean_function=None, emission_cov_function=None)
    return params, obs

def rebayes_scan(
        estimator,
        X: Float[Array, "ntime input_dim"],
        Y: Float[Array, "ntime emission_dim"],
        callback=None
    ) -> Tuple[Any, Any]:
        """Apply filtering to entire sequence of data. Return final belief state and list of outputs from callback."""
        num_timesteps = X.shape[0]
        def step(carry, t):
            params, bel = carry
            pred_bel = estimator.predict_state(params, bel)
            pred_obs = estimator.predict_obs(params, bel, X[t])
            bel = estimator.update_state(params, pred_bel, X[t], Y[t])
            params = estimator.update_params(params, t,  X[t], Y[t], pred_obs, bel)
            out = None
            if callback is not None:
                out = callback(params, bel, pred_obs, t, X[t], Y[t], pred_bel)
            return (params, bel), out
        params, bel = estimator.init()
        carry, outputs = jax.lax.scan(step, (params, bel), jnp.arange(num_timesteps))
        return carry, outputs