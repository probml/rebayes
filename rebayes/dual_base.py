from collections import namedtuple
from typing import Any, Tuple, Union

import chex
from jax.lax import scan
import jax.numpy as jnp
from jaxtyping import Float, Array
from tqdm import trange


RebayesEstimator = namedtuple(
    "RebayesEstimator", 
    ["init", "init_optimizer_params", "predict_state", "update_state", "predict_obs", 
     "predict_obs_cov", "update_params", "update_optimizer_params",]
)


# immutable set of functions for observation model
ObsModel = namedtuple("ObsModel", ["emission_mean_function", "emission_cov_function"])


@chex.dataclass
class DualRebayesParams:
    mu0: chex.Array
    eta0: float
    dynamics_scale_factor: float = 1.0 # gamma
    dynamics_noise: float = 0.0 # Q
    obs_noise: Float[Array, "emission_dim emission_dim"] = jnp.eye(1) # R
    cov_inflation_factor: float = 0.0 # alpha  
    nobs: int = 0 # counts number of observations seen so far (for adaptive estimation)


def form_tril_matrix(theta, C):
    """Form a lower triangular matrix from a vector of parameters.
    The resulting matrix M has the following form:
               0 if i < j
    M[i, j] =  exp(theta[(i+1)*(i+2)/2 - 1]) if i == j
               theta[(i)*(i+1)/2 + j] if i > j

    Args:
        theta (C*(C+1)/2): vector of parameters.
        C (int): number of rows/columns in the resulting matrix.

    Returns:
        M: lower triangular matrix.
    """
    assert int(C*(C+1)/2) == theta.shape[0]
    M = jnp.zeros((C, C))
    
    tril_index = jnp.tril_indices(C)
    M = M.at[tril_index].set(theta)

    diag_index = jnp.diag_indices(C)
    M = M.at[diag_index].set(jnp.exp(M[diag_index]))
    
    return M


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
    optimize_params = False,
    **kwargs,
) -> Tuple[Any, Any]:
    """Apply filtering to entire sequence of data. Return final belief state and list of outputs from callback."""
    num_timesteps = X.shape[0]
    def step(carry, t):
        params, bel = carry
        pred_bel = estimator.predict_state(params, bel)
        pred_obs = estimator.predict_obs(params, bel, X[t])
        bel = estimator.update_state(params, pred_bel, X[t], Y[t])
        out = None
        if not optimize_params:
            params = estimator.update_params(params, t,  X[t], Y[t], pred_obs, bel)
        if callback is not None:
            out = callback(params, bel, pred_obs, t, X[t], Y[t], pred_bel, **kwargs)
        return (params, bel), out
    if params is None or bel is None:
        params, bel = estimator.init()
    carry, outputs = scan(step, (params, bel), jnp.arange(num_timesteps))
    return carry, outputs
    

def dual_rebayes_scan_dataloader( # not tested!
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
            params, out = callback_on_batch_end(params, bel, i, batch)
            outputs.append(out)
    return (params, bel), outputs
    

def dual_rebayes_optimize_scan(
    estimator,
    data_loader,
    num_epochs,
    tx,
    grad_callback,
    progress_bar = True,
) -> Tuple[Any, Any]:
    _, Y = next(iter(data_loader))
    emission_dim = 1 if len(Y.shape) == 1 else Y.shape[1]
    
    params, bel = estimator.init()
    params_bel = estimator.init_optimizer_params(tx, emission_dim)
    
    epoch_range = trange(num_epochs, desc="Epoch 0 average loss: 0.0") if progress_bar else range(num_epochs)
    for epoch in epoch_range:
        losses = []
        for batch in data_loader:
            Xtr, Ytr = jnp.array(batch[0]), jnp.array(batch[1])
            (params, bel), out = dual_rebayes_scan(estimator, Xtr, Ytr, grad_callback,
                                                   params, bel, optimize_params=True,
                                                   params_bel=params_bel, 
                                                   update_fn=estimator.update_state,
                                                   predict_fn=estimator.predict_obs)
            loss, grads = out
            grads = grads.sum()
            params_bel = params_bel.apply_gradients(grads=grads)

            params = estimator.update_optimizer_params(params, params_bel, emission_dim)
            losses.append(loss.mean())
            if progress_bar:
                epoch_range.set_description(f"Epoch {epoch} average loss: {jnp.mean(jnp.array(losses)):.4f}")
    
    return params