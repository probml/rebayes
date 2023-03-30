
from collections import namedtuple
from typing import Any, Tuple, Union

import jax
import chex
from jax.lax import scan
import jax.numpy as jnp
from jaxtyping import Float, Array
from tqdm import trange

import tensorflow_probability.substrates.jax as tfp
tfd = tfp.distributions
tfb = tfp.bijectors
MVN = tfd.MultivariateNormalFullCovariance
MVND = tfd.MultivariateNormalDiag


import optax


def predict_update_batch(
        params,
        bel,
    emission_mean_fn,
    predict_state_fn,
    update_state_fn,
    Xs,
    Ys
):
    num_timesteps = Xs.shape[0]
    is_classification = params['is_classification']
    def step(bel, t):
        bel = carry
        pred_bel = predict_state_fn(params['dyn_noise'], params['dyn_weights'], params['inflation'], bel)
        x = Xs[t]
        yhat = emission_mean_fn(pred_bel.mean, x)
        ytrue = Ys[t]
        if is_classification:
            #obs_cov = emission_cov_fn(pred_bel.mean)
            ps = yhat # probabilities
            obs_cov =  jnp.diag(ps) - jnp.outer(ps, ps) + 1e-3 * jnp.eye(len(ps)) # Add diagonal to avoid singularity
        else:
            obs_cov = params['obs_noise']
        bel = update_state_fn(pred_bel, x, ytrue, emission_mean_fn, obs_cov)
        if is_classification:
            logits = jnp.log(yhat)
            loss = optax.softmax_cross_entropy_loss_with_integer_labels(logits, ytrue)
        else:
            #loss = jnp.square(Ytr - yhat).mean()
            pdf = MVN(yhat, obs_cov)
            loss = -pdf.logpdf(ytrue)
        return bel, loss
    carry, losses = scan(step, bel, jnp.arange(num_timesteps))
    loss = jnp.sum(losses)
    return carry, loss
    
#apply_fn = lambda w, x: cnn.apply({'params': unflatten_fn(w)}, x).ravel()
#emission_mean_function=lambda w, x: jax.nn.softmax(apply_fn(w, x))

def dual_rebayes_simple(
        init_fn, 
        nstates,
    emission_mean_fn,
    is_classification,
    predict_state_fn,
    update_state_fn,
    constrain_params_fn,
    data_loader,
    optimizer
):
    X, Y = next(iter(data_loader))
    emission_dim = 1 if len(Y.shape) == 1 else Y.shape[1]
    params_unc, bel = init_fn(nstates, emission_dim, is_classification)
    opt_state = optimizer.init(params_unc)
    def step(carry, b):
        params_unc, bel, opt_state = carry
        batch = data_loader[b]
        Xs, Ys = batch[0], batch[1]
        def lossfn(params_unc):
            params_con = constrain_params_fn(params_unc)
            bel_post, loss = predict_update_batch(
                params_con, bel,
                emission_mean_fn, is_classification,
                predict_state_fn, update_state_fn, 
                Xs, Ys)
            return loss, bel_post
        (loss_value, bel), grads = jax.value_and_grad(lossfn, has_aux=True)(params_unc)
        param_updates, opt_state = optimizer.update(grads, opt_state, params_unc)
        params_unc = optax.apply_updates(params_unc, param_updates)
        return (params_unc, bel, opt_state), loss_value
    carry, losses = scan(step, (params_unc, bel, opt_state), jnp.arange(len(data_loader)))
    return carry, losses
