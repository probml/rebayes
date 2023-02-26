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


@chex.dataclass
class Gaussian:
    mean: chex.Array
    cov: chex.Array

@chex.dataclass
class Belief:
    dummy: float
    # The belief state can be a Gaussian or some other representation (eg samples)
    # This must be a chex dataclass so that it works with lax.scan as a return type for carry

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


    def init_bel(self) -> Belief:
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
        """Given bel(t|t-1) = p(z(t) | D(1:t-1)), return predicted-obs(t|t-1) = E(y(t) | u(t), D(1:t-1))"""
        return None

    def update_state(
        self,
        bel: Belief,
        X: Float[Array, "input_dim"],
        y: Float[Array, "obs_dim"]
    ) -> Belief:
        """Return bel(t|t) = p(z(t) | X(t), y(t), D(1:t-1)) using bel(t|t-1) and Yt"""
        raise NotImplementedError

    def update_state_batch(
        self,
        bel: Belief, 
        X: Float[Array, "batch_size input_dim"],
        Y: Float[Array, "batch_size emission_dim"]
    ) -> Tuple[Belief, Any]:
        batch_size = X.shape[0]
        def step(bel, i):
            bel = self.predict_state(bel)
            bel = self.update_state(bel, X[i], Y[i])
            return bel, None
        bel, _ = scan(step, bel, jnp.arange(batch_size))
        return bel
    
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
    
    def scan_dataloader(
        self,
        train_dl,
        test_dl,
        callback=None,
        bel=None,
        verbose=False,
        **kwargs
    ) -> Tuple[Belief, Any]:
        """Scan over pytorch dataloaders."""
        train_iter = iter(train_dl)
        num_batches = len(train_iter)
        if test_dl is not None:
            test_iter = cycle(iter(test_dl)) # ensure test stream is infinite
        else:
            test_iter = None
        def get_batch():
            # convert from pytorch tensor to jax array
            Xtr, Ytr = next(train_iter)
            Xtr, Ytr = jnp.array(Xtr.numpy()), jnp.array(Ytr.numpy())
            if test_iter is not None:
                    Xte, Yte = next(test_iter)
                    Xte, Yte = jnp.array(Xte.numpy()), jnp.array(Yte.numpy())
            else:
                    Xte, Yte = None, None
            return Xtr, Ytr, Xte, Yte
        
        if bel is None:
            bel = self.init_bel()
        outputs = []
        for b in jnp.arange(num_batches):
            if verbose: print('batch ', b)
            bel_pre_update = bel
            Xtr, Ytr, Xte, Yte = get_batch()
            bel = self.update_state_batch(bel, Xtr, Ytr)
            if callback is None:
                out = None
            else:
                out = callback(bel, bel_pre_update, b, Xtr, Ytr, Xte, Yte, **kwargs)
                outputs.append(out)
        return bel, outputs