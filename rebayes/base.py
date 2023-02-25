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

class GaussianBroken(NamedTuple):
    mean: Float[Array, "state_dim"]
    cov: Union[Float[Array, "state_dim state_dim"], Float[Array, "state_dim"]]

@chex.dataclass
class Gaussian:
    mean: chex.Array
    cov: chex.Array

@chex.dataclass
class Belief:
    dummy: float
    # Can be over-ridden by other representations (e.g., MCMC samples or memory buffer)

class RebayesParams(NamedTuple):
    initial_mean: Float[Array, "state_dim"]
    initial_covariance: CovMat
    dynamics_weights: CovMat
    dynamics_covariance: CovMat
    #emission_function: FnStateAndInputToEmission
    #emission_covariance: CovMat
    emission_mean_function: FnStateAndInputToEmission
    emission_cov_function: FnStateAndInputToEmission2
    emission_dist: EmissionDistFn = lambda mean, cov: MVN(loc=mean, covariance_matrix=cov)
    #emission_dist=lambda mu, Sigma: tfd.Poisson(log_rate = jnp.log(mu))
    adaptive_emission_cov: bool=False
    dynamics_covariance_inflation_factor: float=0.0



class Rebayes(ABC):
    def __init__(
        self,
        params: RebayesParams,
    ):
        self.params = params


    def init_bel(self) -> Belief:
        raise NotImplementedError

    #@partial(jit, static_argnums=(0,))
    def predict_state(
        self,
        bel: Belief
    ) -> Belief:
        """Given bel(t-1|t-1) = p(z(t-1) | D(1:t-1)), return bel(t|t-1) = p(z(t) | z(t-1), D(1:t-1)).
        """
        raise NotImplementedError

    #@partial(jit, static_argnums=(0,))
    def predict_obs(
        self,
        bel: Belief,
        X: Float[Array, "input_dim"]
    ) -> Float[Array, "output_dim"]: 
        """Given bel(t|t-1) = p(z(t) | D(1:t-1)), return predicted-obs(t|t-1) = E(y(t) | u(t), D(1:t-1))"""
        raise NotImplementedError

    def update_state(
        self,
        bel: Belief,
        X: Float[Array, "input_dim"],
        y: Float[Array, "obs_dim"]
    ) -> Belief:
        """Return bel(t|t) = p(z(t) | u(t), y(t), D(1:t-1)) using bel(t|t-1)"""
        raise NotImplementedError

    def scan(
        self,
        X: Float[Array, "ntime input_dim"],
        Y: Float[Array, "ntime emission_dim"],
        callback=None,
        bel=None,
        progress_bar=True,
        **kwargs
    ) -> Tuple[Belief, Any]:
        """Apply filtering to entire sequence of data. Return final belief state and outputs from callback."""
        num_timesteps = X.shape[0]
        def step(bel, t):
            bel = self.predict_state(bel)
            pred_obs = self.predict_obs(bel, X[t])
            bel = self.update_state(bel, X[t], Y[t])
            out = None
            if callback is not None:
                out = callback(bel, pred_obs, t, X[t], Y[t], **kwargs)
            return bel, out
        carry = bel
        if bel is None:
            carry = self.init_bel()

        if progress_bar:
            step = scan_tqdm(num_timesteps)(step)

        bel, outputs = scan(step, carry, jnp.arange(num_timesteps))
        return bel, outputs
    