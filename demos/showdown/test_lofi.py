import jax
import numpy as np
import jax.numpy as jnp
from rebayes import base
from functools import partial
from jax.flatten_util import ravel_pytree
from bayes_opt import BayesianOptimization
from rebayes.low_rank_filter import lofi

def bbf(
    log_init_cov,
    dynamics_weights,
    log_emission_cov,
    log_dynamics_cov,
    # Specify before running
    train,
    test,
    flat_params,
    callback,
    apply_fn,
    params_lofi,
    method="lofi"
):
    """
    Black-box function for Bayesian optimization.
    """
    X_train, y_train = train
    X_test, y_test = test

    test_callback_kwargs = {"X_test": X_test, "y_test": y_test, "apply_fn": apply_fn}
    params_rebayes = base.RebayesParams(
        initial_mean=flat_params,
        initial_covariance=jnp.exp(log_init_cov).item(),
        dynamics_weights=dynamics_weights,
        dynamics_covariance=jnp.exp(log_dynamics_cov),
        emission_mean_function=apply_fn,
        emission_cov_function=lambda w, x: jnp.exp(log_emission_cov),
    )

    estimator = lofi.RebayesLoFi(params_rebayes, params_lofi, method=method)

    bel, _ = estimator.scan(X_train, y_train, progress_bar=False)
    metric = callback(bel, **test_callback_kwargs)["test"].item()
    isna = np.isnan(metric)
    metric = 10 if isna else metric
    return -metric


def apply(flat_params, x, model, unflatten_fn):
    return model.apply(unflatten_fn(flat_params), x)


def create_optimizer(
    model,
    bounds,
    random_state,
    train,
    test,
    params_lofi,
    callback=None,
    method="lofi"
):
    key = jax.random.PRNGKey(random_state)
    X_train, _ = train
    _, n_params = X_train.shape

    batch_init = jnp.ones((1, n_params))
    params_init = model.init(key, batch_init)
    flat_params, recfn = ravel_pytree(params_init)

    apply_fn = partial(apply, model=model, unflatten_fn=recfn)
    bbf_partial = partial(
        bbf,
        train=train,
        test=test,
        flat_params=flat_params,
        callback=callback,
        apply_fn=apply_fn,
        params_lofi=params_lofi,
        method=method
    )

    optimizer = BayesianOptimization(
        f=bbf_partial,
        pbounds=bounds,
        random_state=random_state,
    )

    return optimizer, apply_fn, n_params


def get_best_params(n_params, optimizer):
    max_params = optimizer.max["params"].copy()

    init_cov = np.exp(max_params["log_init_cov"]).item()
    emission_cov = np.exp(max_params["log_emission_cov"])
    dynamics_cov = np.exp(max_params["log_dynamics_cov"])
    dynamics_weights = max_params["dynamics_weights"]

    hparams = {
        "initial_covariance": init_cov,
        "dynamics_covariance": dynamics_cov,
        "dynamics_weights": dynamics_weights,
        "emission_cov_function": lambda w, x: emission_cov,
    }

    return hparams


def build_estimator(init_mean, hparams, params_lofi, apply_fn, method="lofi"):
    params = base.RebayesParams(
        initial_mean=init_mean,
        emission_mean_function=apply_fn,
        **hparams,
    )

    estimator = lofi.RebayesLoFi(params, params_lofi, method)
    return estimator