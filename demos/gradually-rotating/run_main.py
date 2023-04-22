"""
In this script, we consider the gradually-rotating
MNIST problem for classification. We analyse the
effect of the dynamics_weights (gamma) parameter
and the dynamics_covariance (Q) parameter.
We take an inflation factor of 0.0
"""

import jax
import optax
import numpy as np
import jax.numpy as jnp
import flax.linen as nn
from functools import partial
from typing import Callable
from jax.flatten_util import ravel_pytree

from rebayes import base
from rebayes.low_rank_filter import lofi
from rebayes.sgd_filter import replay_sgd as rsgd


class MLP(nn.Module):
    n_out: int = 1
    n_hidden: int = 100
    activation: Callable = nn.elu

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.n_hidden)(x)
        x = self.activation(x)
        x = nn.Dense(self.n_hidden)(x)
        x = self.activation(x)
        x = nn.Dense(self.n_hidden)(x)
        x = self.activation(x)
        x = nn.Dense(self.n_out, name="last-layer")(x)
        return x


def make_bnn_flax(dim_in, dim_out, nhidden=50):
    key = jax.random.PRNGKey(314)
    model = MLP(dim_out, nhidden)
    params = model.init(key, jnp.ones((1, dim_in)))
    flat_params, recfn = ravel_pytree(params)
    return model, params, flat_params, recfn


def apply_fn_flat(flat_params, x, model, recfn):
    return model.apply(recfn(flat_params), x)


def apply_fn_unflat(params, x, model):
    return model.apply(params, x)


def apply_fn_flat(flat_params, x, model, recfn):
    return model.apply(recfn(flat_params), x)


def load_lofi_agent(
    cfg,
    mean_init,
    emission_mean_fn,
    emission_cov_fn,
):
    ssm_params = base.RebayesParams(
            initial_mean=mean_init,
            initial_covariance=cfg.lofi.initial_covariance,
            dynamics_weights=cfg.lofi.dynamics_weight,
            dynamics_covariance=cfg.lofi.dynamics_covariance,
            emission_mean_function=emission_mean_fn,
            emission_cov_function=emission_cov_fn,
            dynamics_covariance_inflation_factor=0.0
    )

    lofi_params = lofi.LoFiParams(memory_size=cfg.memory, steady_state=False, inflation="hybrid")

    agent = lofi.RebayesLoFiDiagonal(ssm_params, lofi_params)
    return agent


def load_rsgd_agent(
    cfg,
    mean_init,
    apply_fn,
    lossfn,
    dim_in,
    dim_out,
    tx=None
):
    if tx is None:
        tx = optax.adam(learning_rate=cfg.rsgd.learning_rate)

    agent = rsgd.FifoSGD(lossfn, 
        apply_fn=apply_fn,
        init_params=mean_init,
        tx=tx,
        buffer_size=cfg.memory,
        dim_features=dim_in,
        dim_output=dim_out,
        n_inner=cfg.rsgd.n_inner
        )
    
    return agent
