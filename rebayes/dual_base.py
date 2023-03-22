from collections import namedtuple
from typing import Tuple, Any

import chex
from jax.lax import scan
import jax.numpy as jnp
from jaxtyping import Float, Array


RebayesEstimator = namedtuple("RebayesEstimator", ["init", "predict_state", "update_state", "predict_obs", "predict_obs_cov", "update_params"])

# immutable set of functions for observation model
ObsModel = namedtuple("ObsModel", ["emission_mean_function", "emission_cov_function"])


@chex.dataclass
class DualRebayesParams:
    mu0: chex.Array
    eta0: float
    dynamics_scale_factor: float = 1.0 # gamma
    dynamics_noise: float = 0.0 # Q
    obs_noise: float = 1.0 # R
    cov_inflation_factor: float = 0.0 # alpha  
    nobs: int = 0 # counts number of observations seen so far (for adaptive estimation)


def make_dual_rebayes_params():
    # dummy constructor
    params = DualRebayesParams(
        mu0 = None, 
        eta0 = None, 
        dynamics_scale_factor = None, 
        dynamics_noise = None, 
        obs_noise = None, 
        cov_inflation_factor = None, 
        nobs = None
    )
    obs = ObsModel(
        emission_mean_function = None, 
        emission_cov_function = None
    )
    
    return params, obs


def dual_rebayes_scan(
        estimator,
        X: Float[Array, "ntime input_dim"],
        Y: Float[Array, "ntime emission_dim"],
        callback=None,
        params = None,
        bel = None,
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
        if params is None or bel is None:
            params, bel = estimator.init()
        carry, outputs = scan(step, (params, bel), jnp.arange(num_timesteps))
        return carry, outputs
    

def rebayes_scan_dataloader( # not tested!
        estimator,
        data_loader,
        callback=None,
        callback_on_batch_end=None,
    ) -> Tuple[Any, Any]:
        outputs = []   
        params, bel = estimator.init() 
        for i, batch in enumerate(data_loader):
            Xtr, Ytr = batch[0], batch[1]
            (params, bel), out = dual_rebayes_scan(estimator, Xtr, Ytr, callback, params, bel)
            outputs.append(out)
            if callback_on_batch_end is not None:
                 out = callback_on_batch_end(params, bel, i, batch)
                 outputs.append(out)
        return (params, bel), outputs