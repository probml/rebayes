"""
Hamiltonian Monte Carlo for Bayesian Neural Network
"""

import jax
import distrax
import blackjax
import jax.numpy as jnp
import flax.linen as nn
from chex import dataclass
from tqdm.auto import tqdm
from functools import partial
from jax_tqdm import scan_tqdm
from typing import Callable, Union, List
from jaxtyping import Float, Array, PyTree
from jax.flatten_util import ravel_pytree

@dataclass
class PriorParam:
    scale_obs: float
    scale_weight: float


def get_leaves(params):
    flat_params, _ = ravel_pytree(params)
    return flat_params


def log_joint(
    params: nn.FrozenDict,
    X: Float[Array, "num_obs dim_obs"],
    y: Float[Array, "num_obs"],
    apply_fn: Callable[[PyTree[float], Float[Array, "num_obs dim_obs"]], Float[Array, "num_obs"]],
    priors: PriorParam,
):
    """
    We sample from a BNN posterior assuming
        p(w{i}) = N(0, scale_prior) ∀ i
        P(y | w, X) = N(apply_fn(w, X), scale_obs)

    TODO:
    * Add more general way to compute observation-model log-probability
    """
    scale_obs = priors.scale_obs
    scale_prior = priors.scale_weight
    
    params_flat = get_leaves(params)
    
    # Prior log probability (use initialised vals for mean?)
    logp_prior = distrax.Normal(loc=0.0, scale=scale_prior).log_prob(params_flat).sum()
    
    # Observation log-probability
    mu_obs = apply_fn(params, X).ravel()
    logp_obs = distrax.Normal(loc=mu_obs, scale=scale_obs).log_prob(y).sum()
    
    logprob = logp_prior + logp_obs
    return logprob


def inference_loop(rng_key, kernel, initial_state, num_samples, tqdm=True):
    def one_step(state, num_step):
        key = jax.random.fold_in(rng_key, num_step)
        state, _ = kernel(key, state)
        return state, state
    
    if tqdm:
        one_step = scan_tqdm(num_samples)(one_step)

    steps = jnp.arange(num_samples)
    _, states = jax.lax.scan(one_step, initial_state, steps)

    return states


def inference(
    key: jax.random.PRNGKey,
    apply_fn: Callable,
    log_joint: Callable,
    params_init: nn.FrozenDict,
    priors: PriorParam,
    X: Float[Array, "num_obs ..."],
    y: Float[Array, "num_obs"],
    num_warmup: int,
    num_steps: int,
    tqdm: bool = True,
):
    key_warmup, key_train = jax.random.split(key)
    potential = partial(
        log_joint,
        priors=priors, X=X, y=y, apply_fn=apply_fn
    )

    adapt = blackjax.window_adaptation(blackjax.nuts, potential, num_warmup)
    final_state, kernel, _ = adapt.run(key_warmup, params_init)
    states = inference_loop(key_train, kernel, final_state, num_steps, tqdm)

    return states


class RebayesHMC:
    def __init__(self, apply_fn, priors, log_joint, num_samples, num_warmup):
        self.apply_fn = apply_fn
        self.priors = priors
        self.log_joint = log_joint
        self.num_samples = num_samples
        self.num_warmup = num_warmup
    

    @partial(jax.jit, static_argnums=(0,))
    def eval(self, bel, X):
        """
        Evaluate the model at the given parameters
        """
        yhat_samples = jax.vmap(self.apply_fn, (0, None))(bel, X)
        return yhat_samples

    def predict_obs(self, bel, X):
        """
        Estimate posterior predictive
        """
        yhat_samples = self.eval(bel, X)
        yhat = yhat_samples.mean(axis=0)
        return yhat
    
    def predict_state(self, bel, X):
        return bel

    def update_state(self, bel, X, y, key, tqdm=False):
        state = inference(
            key, self.apply_fn, self.log_joint, bel, self.priors,
            X, y, self.num_warmup, self.num_samples,
            tqdm=tqdm
        )
        return state.position

    def scan(
        self,
        key: jax.random.PRNGKey,
        params_init: nn.FrozenDict,
        X: Float[Array, "ntime ..."],
        y: Float[Array, "ntime emission_dim"],
        eval_steps: Union[List, None] = None,
        callback: Callable = None,
    ):
        num_samples = len(y)
        if eval_steps is None:
            eval_steps = list(range(num_samples))

        params_hist = {} 
        for n_eval in tqdm(eval_steps):
            X_eval = X[:n_eval]
            y_eval = y[:n_eval]

            bel_update = self.update_state(params_init, X_eval, y_eval, key)
            params_hist[n_eval] = bel_update

            if callback is not None:
                callback(bel_update, n_eval)
        
        return params_hist
