import jax
import chex
import jax.numpy as jnp
from jax.lax import scan
import tensorflow_probability.substrates.jax as tfp

from jax import jit
from jaxtyping import Float, Array
from jax_tqdm import scan_tqdm
from functools import partial
from abc import ABC, abstractmethod
from typing import Callable, Union, Tuple, Any

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

_jacrev_2d = lambda f, x: jnp.atleast_2d(jax.jacrev(f)(x))


@chex.dataclass
class Belief:
    dummy: float
    # The belief state can be a Gaussian or some other representation (eg samples)
    # This must be a chex dataclass so that it works with lax.scan as a return type for carry


class Rebayes(ABC):
    def __init__(
        self,
        dynamics_covariance: CovMat,
        emission_mean_function: Union[FnStateToEmission, FnStateAndInputToEmission],
        emission_cov_function: Union[FnStateToEmission2, FnStateAndInputToEmission2],
        emission_dist: EmissionDistFn,
    ):
        self.dynamics_covariance = dynamics_covariance
        self.emission_mean_function = emission_mean_function
        self.emission_cov_function = emission_cov_function
        self.emission_dist = emission_dist


    @abstractmethod
    def init_bel(
        self,
        initial_mean: Float[Array, "state_dim"],
        initial_covariance: CovMat,
        Xinit = None,
        Yinit = None,
    ) -> Belief:
        ...

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
        """eturn E(y(t) | X(t), D(1:t-1))"""
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
        emission_dist = self.emission_dist(pred_obs_mean, pred_obs_cov)
        log_prob = emission_dist.log_prob(y)
        
        return log_prob

    @abstractmethod
    def update_state(
        self,
        bel: Belief,
        X: Float[Array, "input_dim"],
        y: Float[Array, "obs_dim"]
    ) -> Belief:
        """Return bel(t|t) = p(z(t) | X(t), y(t), D(1:t-1)) using bel(t|t-1) and Yt"""
        ...
    
    @abstractmethod
    def sample_state(
        self,
        bel: Belief,
        key: Float[Array, "key_dim"],
        n_samples: int = 100,
    ) -> Float[Array, "n_samples state_dim"]:
        """Return samples from p(z(t) | D(1:t))"""
        ...
    
    @partial(jax.jit, static_argnames=("self", "n_samples", "glm_predictive"))
    def nlpd_mc(self, key, bel, x, y, n_samples=30, glm_predictive=False):
        """
        Compute the negative log predictive density (nlpd) as
        a Monte Carlo estimate.
        llfn: log likelihood function
            Takes mean, x, y
        """
        x = jnp.atleast_2d(x)
        y = jnp.atleast_1d(y)
        bel = self.predict_state(bel)

        params_sample = self.sample_state(bel, key, n_samples)
        
        def llfn(params, x, y):
            y = y.ravel()
            args = self.emission_mean_function(params, x)
            log_likelihood = self.emission_dist(*args).log_prob(y)
            
            return log_likelihood.sum()

        def llfn_glm_predictive(params, x, y):
            y = y.ravel()
            m_Y = lambda w: self.emission_mean_function(w, x)
            F = _jacrev_2d(m_Y, bel.mean)
            mean = (m_Y(bel.mean) + F @ (params - bel.mean)).ravel()
            log_likelihood = self.emission_dist(mean, scale).log_prob(y) # TODO: FIx undefined scale
            
            return log_likelihood.sum()

        # Compute vectorised nlpd
        if glm_predictive:
            llfn = llfn_glm_predictive

        vnlpd = lambda w: jax.vmap(llfn, (None, 0, 0))(w, x, y)
        nlpd_vals = -jax.lax.map(vnlpd, params_sample).squeeze()
        nlpd_mean = nlpd_vals.mean()

        return nlpd_mean

    
    def scan(
        self,
        initial_mean: Float[Array, "state_dim"],
        initial_covariance: CovMat,
        X: Float[Array, "ntime input_dim"],
        Y: Float[Array, "ntime emission_dim"],
        callback=None,
        bel=None, # TODO: I don't think we need bel as an argument
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
            # Make initial_covariance optional? for exampl, point-estimate RSGD
            carry = self.init_bel(initial_mean, initial_covariance, Xinit, Yinit)
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
