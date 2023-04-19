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
from rebayes.utils.utils import tree_to_cpu
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


def callback(bel, pred_obs, t, X, y, bel_pred, apply_fn, lagn=20, store_fro=True, **kwargs):
    X_test, y_test = kwargs["X_test"], kwargs["y_test"]
    recfn = kwargs["recfn"]
    
    slice_ix = jnp.arange(0, lagn) + t
    
    X_test = jnp.take(X_test, slice_ix, axis=0, fill_value=0)
    y_test = jnp.take(y_test, slice_ix, axis=0, fill_value=0)

    y_next = y.squeeze().argmax()
    phat_next = pred_obs.squeeze()
    yhat_next = phat_next.argmax()

    yhat_test = apply_fn(bel.mean, X_test).squeeze().argmax()

    # Compute errors
    err_test = (y_test == yhat_test).mean()
    err = (y_next == yhat_next).mean()
    
    if store_fro:
        mean_params = recfn(bel.mean)
        params_magnitude = jax.tree_map(lambda A: A["kernel"], mean_params, is_leaf=lambda k: "kernel" in k)
        params_magnitude = jax.tree_map(lambda A: jnp.linalg.norm(A, ord="fro"), params_magnitude)
    else:
        params_magnitude = None
    
    res = {
        "n-step-pred": yhat_test,
        "nsa-error": err_test,
        "osa-error": err,
        "phat": phat_next,
        "params_magnitude": params_magnitude
    }
    return res


@partial(jax.jit, static_argnames=("apply_fn",))
def lossfn_fifo(params, counter, X, y, apply_fn):
    yhat = apply_fn(params, X)
    yhat = jax.nn.softmax(yhat).squeeze()
    y = y.squeeze()

    logits = jnp.log(yhat) # B x K
    loss = -jnp.einsum("bk,bk,b->", logits, y, counter) / counter.sum()
    return loss


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


def damp_angle(n_configs, minangle, maxangle):
    t = np.linspace(-0.5, 1.5, n_configs)
    
    angles = np.exp(t) * np.sin(55 * t)
    angles = np.sin(55 * t)
    
    angles = (angles + 1) / 2 * (maxangle - minangle) + minangle + np.random.randn(n_configs) * 2
    
    return angles
    

def emission_cov_function(w, x, fn_mean):
    """
    Compute the covariance matrix of the emission distribution.
    fn_mean: emission mean function
    """
    ps = fn_mean(w, x)
    n_classes = len(ps)
    I = jnp.eye(n_classes)
    return jnp.diag(ps) - jnp.outer(ps, ps) + 1e-3 * I


def categorise(labels):
    """
    Labels is taken to be a list of ordinal numbers
    """
    # One-hot-encoded
    n_classes = max(labels) + 1

    ohed = jax.nn.one_hot(labels, n_classes)
    filter_columns = ~(ohed == 0).all(axis=0)
    ohed = ohed[:, filter_columns]
    return ohed


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
):
    agent = rsgd.FifoSGD(lossfn, 
        apply_fn=apply_fn,
        init_params=mean_init,
        tx=optax.adam(learning_rate=cfg.rsgd.learning_rate),
        buffer_size=cfg.memory,
        dim_features=dim_in,
        dim_output=dim_out,
        n_inner=cfg.rsgd.n_inner
        )
    
    return agent


if __name__ == "__main__":
    from cfg_main import get_config
    from rebayes.datasets import rotating_mnist_data as data
    target_digits = [2, 3]
    n_classes = len(target_digits)
    num_train = 6_000

    data = data.load_and_transform(damp_angle, target_digits, num_train, sort_by_angle=False)
    X_train, signal_train, labels_train = data["dataset"]["train"]
    X_test, signal_test, labels_test = data["dataset"]["test"]
    Y_train = categorise(labels_train)
    Y_test = categorise(labels_test)

    cfg = get_config()

    _, dim_in = X_train.shape
    model, tree_params, flat_params, recfn = make_bnn_flax(dim_in, n_classes)
    apply_fn = partial(apply_fn_flat, model=model, recfn=recfn)
    def emission_mean_fn(w, x): return nn.softmax(apply_fn(w, x))
    emission_cov_fn = partial(emission_cov_function, fn_mean=emission_mean_fn)

    _, dim_in = data["dataset"]["train"][0].shape

    callback_part = partial(callback,
                            apply_fn=emission_mean_fn,
                            X_test=X_train, y_test=Y_train,
                            )

    ### Lofi---load and train
    agent = load_lofi_agent(cfg, flat_params, emission_mean_fn, emission_cov_fn)
    callback_lofi = partial(callback_part, recfn=recfn)
    bel, outputs_lofi = agent.scan(X_train, Y_train, progress_bar=True, callback=callback_lofi)
    bel = jax.block_until_ready(bel)
    outputs_lofi = tree_to_cpu(outputs_lofi)

    ### RSGD---load and train
    callback_rsgd = partial(callback_part, recfn=lambda x: x)
    apply_fn = partial(apply_fn_unflat, model=model)
    agent = load_rsgd_agent(cfg, tree_params, apply_fn, lossfn_fifo, dim_in, n_classes)
    bel, outputs_rsgd = agent.scan(X_train, Y_train, progress_bar=True, callback=callback_rsgd)
    bel = jax.block_until_ready(bel)
    outputs_rsgd = tree_to_cpu(outputs_rsgd)
