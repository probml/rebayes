from abc import ABC, abstractmethod
from functools import partial
from typing import Callable, Union, Tuple, Any

import chex
import jax
from jax import jit, vmap
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
class Belief:
    mean: Float[Array, "state_dim"]
    obs_noise_var: float
    # The belief state can be a Gaussian or some other representation (eg samples)
    # This must be a chex dataclass so that it works with lax.scan as a return type for carry


class Rebayes(ABC):
    def __init__(
        self,
        dynamics_covariance: CovMat,
        emission_mean_function: Union[FnStateToEmission, FnStateAndInputToEmission],
        emission_cov_function: Union[FnStateToEmission2, FnStateAndInputToEmission2],
        emission_dist: EmissionDistFn,
        adaptive_emission_covariance: bool = False,
    ):
        self.dynamics_covariance = dynamics_covariance
        self.emission_mean_function = emission_mean_function
        self.emission_cov_function = emission_cov_function
        self.emission_dist = emission_dist
        self.adaptive_emission_covariance = adaptive_emission_covariance

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

    @abstractmethod
    def predict_obs(
        self,
        bel: Belief,
        X: Float[Array, "input_dim"]
    ) -> Union[Float[Array, "output_dim"], Any]: 
        """Return E(y(t) | X(t), D(1:t-1))"""
        ...
    
    @partial(jit, static_argnums=(0,))
    def predict_obs_cov(
        self,
        bel: Belief,
        X: Float[Array, "input_dim"],
        aleatoric_factor: float = 1.0,
        apply_fn: Callable = None,
    ) -> Union[Float[Array, "output_dim output_dim"], Any]: 
        """Return Cov(y(t) | X(t), D(1:t-1))"""
        return None
    
    @partial(jit, static_argnums=(0,))
    def obs_cov(
        self,
        bel: Belief,
        X: Float[Array, "input_dim"]
    ) -> Union[Float[Array, "output_dim output_dim"], Any]:
        """Return R(t)"""
        y_pred = jnp.atleast_1d(self.predict_obs(bel, X))
        C, *_ = y_pred.shape
        if self.adaptive_emission_covariance:
            R = bel.obs_noise_var * jnp.eye(C)
        else:
            R = jnp.atleast_2d(self.emission_cov_function(bel.mean, X))
        
        return R
    
    @partial(jit, static_argnums=(0,))
    def evaluate_log_prob(
        self,
        bel: Belief,
        X: Float[Array, "input_dim"],
        y: Float[Array, "obs_dim"],
        aleatoric_factor: float = 1.0,
    ) -> float:
        """Return log p(y(t) | X(t), D(1:t-1))"""
        X = jnp.atleast_2d(X)
        y = jnp.atleast_1d(y)
        
        def llfn(x, y):
            m = self.predict_obs(bel, x)
            V = self.predict_obs_cov(bel, x, aleatoric_factor)
            return self.emission_dist(m, V).log_prob(y)
        
        log_prob = vmap(llfn)(X, y)
        
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
    
    @partial(jit, static_argnums=(0,))
    def update_hyperparams_prepred(
        self,
        bel: Belief,
        x: Float[Array, "input_dim"],
        y: Float[Array, "output_dim"],
    ) -> Belief:
        """Return bel(t|t-1) = p(noise(t) | X(t), y(t), D(1:t-1)) using bel(t|t-1) and Yt"""
        return bel
    
    @partial(jit, static_argnums=(0,))
    def update_hyperparams_postpred(
        self,
        bel: Belief,
        x: Float[Array, "input_dim"],
        y: Float[Array, "output_dim"],
    ) -> Belief:
        """Return bel(t|t-1) = p(noise(t) | X(t), y(t), D(1:t-1)) using bel(t|t-1) and Yt"""
        return bel
    
    @abstractmethod
    def sample_state(
        self,
        bel: Belief,
        key: Float[Array, "key_dim"],
        n_samples: int = 100,
        temperature: float = 1.0,
    ) -> Float[Array, "n_samples state_dim"]:
        """Return samples from p(z(t) | D(1:t-1))"""
        ...
    
    @partial(jax.jit, static_argnames=("self", "n_samples", "temperature", "glm_callback"))
    def nlpd_mc(
        self,
        bel: Belief,
        key: Float[Array, "key_dim"],
        x: Float[Array, "ntime input_dim"],
        y: Float[Array, "ntime emission_dim"],
        n_samples: int=30,
        temperature: float=1.0,
        glm_callback=None,
    ) -> float:
        """
        Compute the negative log predictive density (nlpd) as
        a Monte Carlo estimate.
        """
        x = jnp.atleast_2d(x)
        y = jnp.atleast_1d(y)
        bel = self.predict_state(bel)
        params_sample = self.sample_state(bel, key, n_samples, temperature)
        mean_fn = self.emission_mean_function if glm_callback is None else glm_callback
        
        def llfn(params, x, y):
            y = y.ravel()
            mean = mean_fn(params, x)
            R = self.obs_cov(bel, x)
            log_likelihood = self.emission_dist(mean, R).log_prob(y)
            
            return log_likelihood.mean()

        # Compute vectorised nlpd
        lpd = vmap(vmap(llfn, (None, 0, 0)), (0, None, None))(params_sample, x, y)
        nlpd = -lpd

        return nlpd

    def scan(
        self,
        initial_mean: Float[Array, "state_dim"],
        initial_covariance: CovMat,
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
            bel_pred = self.update_hyperparams_prepred(bel, X[t], Y[t])
            bel_pred = self.predict_state(bel_pred)
            bel_pred = self.update_hyperparams_postpred(bel, X[t], Y[t])
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
    def scan_state_batch(
        self,
        bel: Belief, 
        X: Float[Array, "batch_size input_dim"],
        Y: Float[Array, "batch_size emission_dim"],
        progress_bar=False
    ) -> Tuple[Belief, Any]:
        bel, _ = self.scan(None, None, X, Y, bel=bel, progress_bar=progress_bar)
        return bel
    
    def scan_state_batch_with_callback(
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
        bel, outputs = self.scan(None, None, X, Y, callback=callback, bel=bel, 
                                 progress_bar=progress_bar, **kwargs)
        return bel, outputs
    
    def scan_dataloader(
        self,
        initial_mean: Float[Array, "state_dim"],
        initial_covariance: CovMat,
        data_loader: Any,
        callback=None,
        bel=None,
        callback_at_end=True,
        progress_bar=False,
        **kwargs,
    ) -> Tuple[Belief, Any]:
        if bel is None:
            bel = self.init_bel(initial_mean, initial_covariance)
        outputs = []
        for i, batch in enumerate(data_loader):
            bel_pre_update = bel
            Xtr, Ytr = batch[0], batch[1]
            if callback_at_end:
                bel = self.scan_state_batch(bel, Xtr, Ytr, progress_bar)
                if callback is None:
                    out = None
                else:
                    out = callback(i, bel_pre_update, bel, batch, **kwargs)
                    outputs.append(out)
            else:
                bel, out = self.scan_state_batch_with_callback(
                    i, bel, Xtr, Ytr, callback, progress_bar, **kwargs
                )
                outputs.append(out)
        return bel, outputs