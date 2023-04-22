import jax
import numpy as np
import jax.numpy as jnp

import run_main as ev
from functools import partial
from cfg_main import get_config
from rebayes.utils.utils import tree_to_cpu
from rebayes.utils.callbacks import cb_clf_sup
from rebayes.sgd_filter import replay_sgd as rsgd
from rebayes.datasets import rotating_mnist_data as data


def damp_angle(n_configs, minangle, maxangle):
    t = np.linspace(-0.5, 1.5, n_configs)
    
    angles = np.exp(t) * np.sin(55 * t)
    angles = np.sin(55 * t)
    
    angles = (angles + 1) / 2 * (maxangle - minangle) + minangle + np.random.randn(n_configs) * 2
    
    return angles


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


def emission_cov_function(w, x, fn_mean):
    """
    Compute the covariance matrix of the emission distribution.
    fn_mean: emission mean function
    """
    ps = fn_mean(w, x)
    n_classes = len(ps)
    I = jnp.eye(n_classes)
    return jnp.diag(ps) - jnp.outer(ps, ps) + 1e-3 * I


if __name__  == "__main__":
    target_digits = [2, 3]
    n_classes = len(target_digits)
    num_train = 6_000

    data = data.load_and_transform(damp_angle, target_digits, num_train, sort_by_angle=False)
    X_train, signal_train, labels_train = data["dataset"]["train"]
    X_test, signal_test, labels_test = data["dataset"]["test"]
    Y_train = categorise(labels_train)
    Y_test = categorise(labels_test)

    cfg = get_config()
    cfg.lofi.dynamics_covariance = 0.0
    cfg.lofi.dynamics_weights = 1.0
    cfg.lofi.initial_covariance = 0.1

    _, dim_in = X_train.shape
    model, tree_params, flat_params, recfn = ev.make_bnn_flax(dim_in, n_classes)
    apply_fn_flat = partial(ev.apply_fn_flat, model=model, recfn=recfn)
    apply_fn_tree = partial(ev.apply_fn_unflat, model=model)
    def emission_mean_fn(w, x): return jax.nn.softmax(apply_fn_flat(w, x))
    emission_cov_fn = partial(emission_cov_function, fn_mean=emission_mean_fn)


    _, dim_in = data["dataset"]["train"][0].shape

    callback_part = partial(cb_clf_sup,
                            X_test=X_train, y_test=Y_train,
                            )

    ### Lofi---load and train
    agent = ev.load_lofi_agent(cfg, flat_params, emission_mean_fn, emission_cov_fn)
    callback_lofi = partial(callback_part, recfn=recfn, apply_fn=apply_fn_flat)
    bel, outputs_lofi = agent.scan(X_train, Y_train, progress_bar=True, callback=callback_lofi)
    bel = jax.block_until_ready(bel)
    outputs_lofi = tree_to_cpu(outputs_lofi)

    ### RSGD---load and train
    apply_fn = partial(apply_fn_tree, model=model)
    callback_rsgd = partial(callback_part, recfn=lambda x: x, apply_fn=apply_fn)
    agent = ev.load_rsgd_agent(cfg, tree_params, apply_fn, rsgd.lossfn_xentropy, dim_in, n_classes)
    bel, outputs_rsgd = agent.scan(X_train, Y_train, progress_bar=True, callback=callback_rsgd)
    bel = jax.block_until_ready(bel)
    outputs_rsgd = tree_to_cpu(outputs_rsgd)
