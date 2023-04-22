"""
In this notebook, we consider the gradually-rotating MNIST
problem for regression. The angle of rotation is a growing
sinusoid.
"""

import jax
import optax
import numpy as np

import run_main as ev
from functools import partial
from cfg_main import get_config
from rebayes.utils.utils import tree_to_cpu
from rebayes.utils.callbacks import cb_reg_sup
from rebayes.sgd_filter import replay_sgd as rsgd
from rebayes.datasets import rotating_mnist_data as data


def damp_angle(n_configs, minangle, maxangle):
    t = np.linspace(0, 1.5, n_configs)
    angles = np.exp(t) * np.sin(35 * t)
    angles = (angles + 1) / 2 * (maxangle - minangle) + minangle + np.random.randn(n_configs) * 2
    return angles


if __name__ == "__main__":
    cfg = get_config()

    num_train = None
    frac_train = 1.0
    target_digit = 2
    data = data.load_and_transform(
        damp_angle, target_digit, num_train, frac_train, sort_by_angle=False
    )

    X_train, Y_train, labels_train = data["dataset"]["train"]


    # TODO: Refactor into LoFi regression
    dim_out = 1
    _, dim_in = X_train.shape
    model, tree_params, flat_params, recfn = ev.make_bnn_flax(dim_in, dim_out)
    apply_fn_flat = partial(ev.apply_fn_flat, model=model, recfn=recfn)
    apply_fn_tree = partial(ev.apply_fn_unflat, model=model)
    def emission_mean_fn(w, x): return apply_fn_flat(w, x)
    def emission_cov_fn(w, x): return 0.02

    ymean, ystd = data["ymean"], data["ystd"]
    callback = partial(cb_reg_sup,
                            X_test=X_train, y_test=Y_train,
                            ymean=ymean, ystd=ystd,
                            )

    callback_lofi = partial(callback, apply_fn=emission_mean_fn)
    callback_rsgd = partial(callback, apply_fn=apply_fn_tree)

    ### LoFi---load and train
    agent = ev.load_lofi_agent(cfg, flat_params, emission_mean_fn, emission_cov_fn)
    bel, output_lofi = agent.scan(X_train, Y_train, progress_bar=True, callback=callback_lofi)
    bel = jax.block_until_ready(bel)
    bel = tree_to_cpu(bel)
    output_lofi = tree_to_cpu(output_lofi)


    ### RSGD---load and train
    lr = 1e-2
    tx = optax.sgd(lr)
    agent = ev.load_rsgd_agent(cfg, tree_params, apply_fn_tree, rsgd.lossfn_rmse, dim_in, dim_out, tx=tx)
    bel, output_rsgd = agent.scan(X_train, Y_train, progress_bar=True, callback=callback_rsgd)
    bel = jax.block_until_ready(bel)
    output_rsgd = tree_to_cpu(output_rsgd)


    ### RSGD (ADAM)---load and train
    lr = 5e-3
    tx = optax.adam(lr)
    agent = ev.load_rsgd_agent(cfg, tree_params, apply_fn_tree, rsgd.lossfn_rmse, dim_in, dim_out, tx=tx)
    bel, output_rsgd_adam = agent.scan(X_train, Y_train, progress_bar=True, callback=callback_rsgd)
    bel = jax.block_until_ready(bel)
    output_rsgd_adam = tree_to_cpu(output_rsgd_adam)
