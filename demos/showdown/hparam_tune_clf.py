import jax
import numpy as np
import jax.numpy as jnp
from rebayes import base
from functools import partial
from jax.flatten_util import ravel_pytree
from bayes_opt import BayesianOptimization
from rebayes.extended_kalman_filter import ekf
from rebayes.low_rank_filter import lofi


def apply(flat_params, x, model, unflatten_fn):
    return model.apply(unflatten_fn(flat_params), x)


def bbf_lofi(
    log_init_cov,
    dynamics_weights,
    # Specify before running
    emission_cov_fn,
    train,
    test,
    flat_params,
    callback,
    apply_fn,
    params_lofi,
    method,
):
    """
    Black-box function for Bayesian optimization.
    """
    X_train, y_train = train
    X_test, y_test = test

    dynamics_covariance = None
    initial_covariance = jnp.exp(log_init_cov).item()

    test_callback_kwargs = {"X_test": X_test, "y_test": y_test, "apply_fn": apply_fn}
    params_rebayes = base.RebayesParams(
        initial_mean=flat_params,
        initial_covariance=initial_covariance,
        dynamics_weights=dynamics_weights,
        dynamics_covariance=dynamics_covariance,
        emission_mean_function=apply_fn,
        emission_cov_function=emission_cov_fn,
    )

    estimator = lofi.RebayesLoFi(params_rebayes, params_lofi, method=method)

    bel, _ = estimator.scan(X_train, y_train, progress_bar=False)
    metric = callback(bel, **test_callback_kwargs)["test"].item()
    return metric



def bbf_ekf(
    log_init_cov,
    dynamics_weights,
    # Specify before running
    emission_cov_fn,
    train,
    test,
    flat_params,
    callback,
    apply_fn,
    method="fdekf",
):
    """
    Black-box function for Bayesian optimization.
    """
    X_train, y_train = train
    X_test, y_test = test

    dynamics_covariance = None
    initial_covariance = jnp.exp(log_init_cov).item()

    test_callback_kwargs = {"X_test": X_test, "y_test": y_test, "apply_fn": apply_fn}
    params_rebayes = base.RebayesParams(
        initial_mean=flat_params,
        initial_covariance=initial_covariance,
        dynamics_weights=dynamics_weights,
        dynamics_covariance=dynamics_covariance,
        emission_mean_function=apply_fn,
        emission_cov_function=emission_cov_fn,
    )

    estimator = ekf.RebayesEKF(params_rebayes, method=method)

    bel, _ = estimator.scan(X_train, y_train, progress_bar=False)
    metric = callback(bel, **test_callback_kwargs)["test"].item()
    return metric


def create_optimizer(
    model,
    bounds,
    random_state,
    train,
    test,
    callback=None,
    method="fdekf",
    **kwargs
):
    key = jax.random.PRNGKey(random_state)
    X_train, _ = train
    _, n_features = X_train.shape

    batch_init = jnp.ones((1, n_features))
    params_init = model.init(key, batch_init)
    flat_params, recfn = ravel_pytree(params_init)

    apply_fn = partial(apply, model=model, unflatten_fn=recfn)
    def emission_cov_fn(w, x):
        return apply_fn(w, x) * (1 - apply_fn(w, x))
    
    if "ekf" in method:
        bbf = bbf_ekf
    elif "lofi" in method:
        bbf = bbf_lofi

    bbf_partial = partial(
        bbf,
        train=train,
        test=test,
        flat_params=flat_params,
        callback=callback,
        apply_fn=apply_fn,
        emission_cov_fn=emission_cov_fn,
        method=method,
        **kwargs # Must include params_lofi if method is lofi
    )

    optimizer = BayesianOptimization(
        f=bbf_partial,
        pbounds=bounds,
        random_state=random_state,
    )

    return optimizer, apply_fn, n_features


def get_best_params(num_params, optimizer, method):
    max_params = optimizer.max["params"].copy()

    dynamics_covariance = None
    initial_covariance = np.exp(max_params["log_init_cov"])
    dynamics_weights = max_params["dynamics_weights"]

    hparams = {
        "initial_covariance": initial_covariance,
        "dynamics_covariance": dynamics_covariance,
        "dynamics_weights": dynamics_weights,
    }

    return hparams


def build_estimator(init_mean, hparams, apply_fn, method, **kwargs):
    """
    _ is a dummy parameter for compatibility with lofi 
    """
    def emission_cov_fn(w, x):
        return apply_fn(w, x) * (1 - apply_fn(w, x))

    params = base.RebayesParams(
        initial_mean=init_mean,
        emission_mean_function=apply_fn,
        emission_cov_function=emission_cov_fn,
        **hparams,
    )

    if "ekf" in method:
        estimator = ekf.RebayesEKF(params, method=method)
    elif "lofi" in method:
        estimator = lofi.RebayesLoFi(params, method=method, **kwargs)
    return estimator
