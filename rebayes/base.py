from abc import ABC
from collections import namedtuple
from functools import partial
from typing import Callable, Union, Tuple, Any

import jax
import chex
from jax import jit
from jax.lax import scan
import jax.numpy as jnp
from jaxtyping import Float, Array
from jax_tqdm import scan_tqdm
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


@chex.dataclass
class Gaussian:
    mean: chex.Array
    cov: chex.Array


@chex.dataclass
class Belief:
    dummy: float
    # The belief state can be a Gaussian or some other representation (eg samples)
    # This must be a chex dataclass so that it works with lax.scan as a return type for carry


# TODO: Deprecate
@chex.dataclass
class RebayesParams:
#class RebayesParams(NamedTuple):
    initial_mean: Float[Array, "state_dim"]
    initial_covariance: CovMat
    dynamics_weights: CovMat
    dynamics_covariance: CovMat
    emission_mean_function: FnStateAndInputToEmission
    emission_cov_function: FnStateAndInputToEmission2
    #emission_dist: EmissionDistFn = lambda mean, cov: MVN(loc=mean, covariance_matrix=cov)
    #emission_dist=lambda mu, Sigma: tfd.Poisson(log_rate = jnp.log(mu))
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


class Rebayes(ABC):
    def __init__(
        self,
        params: RebayesParams,
    ):
        self.params = params


    def init_bel(
        self,
        Xinit = None,
        Yinit = None,
    ) -> Belief:
        raise NotImplementedError

    @partial(jit, static_argnums=(0,))
    def predict_state(
        self,
        bel: Belief
    ) -> Belief:
        """Return bel(t|t-1) = p(z(t) | z(t-1), D(1:t-1)) given bel(t-1|t-1).
        By default, we assume a stationary model, so the belief is unchanged.
        """
        return bel

    @partial(jit, static_argnums=(0,))
    def predict_obs(
        self,
        bel: Belief,
        X: Float[Array, "input_dim"]
    ) -> Union[Float[Array, "output_dim"], Any]: 
        """Return E(y(t) | X(t), D(1:t-1))"""
        return None
    
    @partial(jit, static_argnums=(0,))
    def predict_obs_cov(
        self,
        bel: Belief,
        X: Float[Array, "input_dim"]
    ) -> Union[Float[Array, "output_dim output_dim"], Any]: 
        """Return Cov(y(t) | X(t), D(1:t-1))"""
        return None
    
    @partial(jit, static_argnums=(0,))
    def evaluate_log_prob(
        self,
        bel: Belief,
        X: Float[Array, "input_dim"],
        y: Float[Array, "obs_dim"]
    ) -> float:
        """Return log p(y(t) | X(t), D(1:t-1))"""
        pred_obs_mean, pred_obs_cov = self.predict_obs(bel, X), self.predict_obs_cov(bel, X)
        emission_dist = self.params.emission_dist(pred_obs_mean, pred_obs_cov)
        log_prob = emission_dist.log_prob(y)
        
        return log_prob

    @partial(jit, static_argnums=(0,))
    def update_state(
        self,
        bel: Belief,
        X: Float[Array, "input_dim"],
        y: Float[Array, "obs_dim"]
    ) -> Belief:
        """Return bel(t|t) = p(z(t) | X(t), y(t), D(1:t-1)) using bel(t|t-1) and Yt"""
        raise NotImplementedError
    
    def scan(
        self,
        X: Float[Array, "ntime input_dim"],
        Y: Float[Array, "ntime emission_dim"],
        callback=None,
        bel=None,
        progress_bar=False,
        debug=False,
        Xinit=None,
        Yinit=None,
        **kwargs
    ) -> Tuple[Belief, Any]:
        """Apply filtering to entire sequence of data. Return final belief state and outputs from callback."""
        num_timesteps = X.shape[0]
        def step(bel, t):
            bel_pred = self.predict_state(bel)
            pred_obs = self.predict_obs(bel, X[t])
            bel = self.update_state(bel_pred, X[t], Y[t])
            out = None
            if callback is not None:
                out = callback(bel, pred_obs, t, X[t], Y[t], bel_pred, **kwargs)
            return bel, out
        carry = bel
        if bel is None:
            if Xinit is not None:
                carry = self.init_bel(Xinit, Yinit)
            else:
                carry = self.init_bel()
        if progress_bar:
            step = scan_tqdm(num_timesteps)(step)
        if debug:
            outputs = []
            for t in range(num_timesteps):
                carry, out = step(carry, t)
                outputs.append(out)
            bel = carry
            outputs = jnp.stack(outputs)
        else:
            bel, outputs = scan(step, carry, jnp.arange(num_timesteps))
        return bel, outputs
    
    @partial(jit, static_argnums=(0, 4))
    def update_state_batch(
        self,
        bel: Belief, 
        X: Float[Array, "batch_size input_dim"],
        Y: Float[Array, "batch_size emission_dim"],
        progress_bar=False
    ) -> Tuple[Belief, Any]:
        bel, _ = self.scan(X, Y, bel=bel, progress_bar=progress_bar)
        return bel
    
    def update_state_batch_with_callback(
        self,
        i: int,
        bel: Belief, 
        X: Float[Array, "batch_size input_dim"],
        Y: Float[Array, "batch_size emission_dim"],
        callback=None,
        progress_bar=False,
        **kwargs
    ) -> Tuple[Belief, Any]:
        if callback is not None:
            callback = partial(callback, i=i)
        bel, outputs = self.scan(X, Y, callback=callback, bel=bel, progress_bar=progress_bar, **kwargs)
        return bel, outputs
    
    def scan_dataloader(
        self,
        data_loader,
        callback=None,
        bel=None,
        callback_at_end=True,
        progress_bar=False,
        **kwargs,
    ) -> Tuple[Belief, Any]:
        if bel is None:
            bel = self.init_bel()
        outputs = []    
        for i, batch in enumerate(data_loader):
            bel_pre_update = bel
            Xtr, Ytr = batch[0], batch[1]
            if callback_at_end:
                bel = self.update_state_batch(bel, Xtr, Ytr, progress_bar)
                if callback is None:
                    out = None
                else:
                    out = callback(i, bel_pre_update, bel, batch, **kwargs)
                    outputs.append(out)
            else:
                bel, out = self.update_state_batch_with_callback(
                    i, bel, Xtr, Ytr, callback, progress_bar, **kwargs
                )
                outputs.append(out)
        return bel, outputs