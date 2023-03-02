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
    log_dynamics_weights,
    log_dynamics_cov,
    log_alpha,
    # Specify before running
    emission_mean_fn,
    emission_cov_fn,
    train,
    test,
    flat_params,
    callback,
    apply_fn,
    params_lofi,
    method="lofi",
):
    """
    Black-box function for Bayesian optimization.
    """
    X_train, y_train = train
    X_test, y_test = test

    dynamics_weights = 1 - jnp.exp(log_dynamics_weights).item()
    dynamics_covariance = jnp.exp(log_dynamics_cov).item()
    initial_covariance = jnp.exp(log_init_cov).item()
    alpha = jnp.exp(log_alpha).item()

    test_callback_kwargs = {"X_test": X_test, "y_test": y_test, "apply_fn": apply_fn}
    params_rebayes = base.RebayesParams(
        initial_mean=flat_params,
        initial_covariance=initial_covariance,
        dynamics_weights=dynamics_weights,
        dynamics_covariance=dynamics_covariance,
        emission_mean_function=emission_mean_fn,
        emission_cov_function=emission_cov_fn,
        dynamics_covariance_inflation_factor=alpha,
    )

    estimator = lofi.RebayesLoFi(params_rebayes, params_lofi, method=method)

    bel, _ = estimator.scan(X_train, y_train, progress_bar=False)
    metric = callback(bel, **test_callback_kwargs)["test"]
    return metric



def bbf_ekf(
    log_init_cov,
    log_dynamics_weights,
    log_dynamics_cov,
    log_alpha,
    # Specify before running
    emission_mean_fn,
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

    dynamics_weights = 1 - jnp.exp(log_dynamics_weights).item()
    dynamics_covariance = jnp.exp(log_dynamics_cov).item()
    initial_covariance = jnp.exp(log_init_cov).item()
    alpha = jnp.exp(log_alpha).item()

    test_callback_kwargs = {"X_test": X_test, "y_test": y_test, "apply_fn": apply_fn}
    params_rebayes = base.RebayesParams(
        initial_mean=flat_params,
        initial_covariance=initial_covariance,
        dynamics_weights=dynamics_weights,
        dynamics_covariance=dynamics_covariance,
        emission_mean_function=emission_mean_fn,
        emission_cov_function=emission_cov_fn,
        dynamics_covariance_inflation_factor=alpha,
    )

    estimator = ekf.RebayesEKF(params_rebayes, method=method)

    bel, _ = estimator.scan(X_train, y_train, progress_bar=False)
    metric = callback(bel, **test_callback_kwargs)["test"]
    return metric


def create_optimizer(
    model,
    bounds,
    random_state,
    train,
    test,
    emission_mean_fn,
    emission_cov_fn,
    callback=None,
    method="fdekf",
    **kwargs
):
    key = jax.random.PRNGKey(random_state)
    X_train, _ = train
    _, *n_features = X_train.shape

    batch_init = jnp.ones((1, *n_features))
    params_init = model.init(key, batch_init)
    flat_params, recfn = ravel_pytree(params_init)

    apply_fn = partial(apply, model=model, unflatten_fn=recfn)
    
    if "ekf" in method:
        bbf = bbf_ekf
    elif "lofi" in method:
        bbf = bbf_lofi

    bbf_partial = partial(
        bbf,
        emission_mean_fn=emission_mean_fn,
        emission_cov_fn=emission_cov_fn,
        train=train,
        test=test,
        flat_params=flat_params,
        callback=callback,
        apply_fn=apply_fn,
        method=method,
        **kwargs # Must include params_lofi if method is lofi
    )

    optimizer = BayesianOptimization(
        f=bbf_partial,
        pbounds=bounds,
        random_state=random_state,
    )

    return optimizer, apply_fn, n_features


def get_best_params(optimizer):
    max_params = optimizer.max["params"].copy()

    initial_covariance = np.exp(max_params["log_init_cov"])
    dynamics_weights = 1 - np.exp(max_params["log_dynamics_weights"])
    dynamics_covariance = np.exp(max_params["log_dynamics_cov"])
    alpha = np.exp(max_params["log_alpha"])

    hparams = {
        "initial_covariance": initial_covariance,
        "dynamics_weights": dynamics_weights,
        "dynamics_covariance": dynamics_covariance,
        "dynamics_covariance_inflation_factor": alpha,
    }

    return hparams


def build_estimator(init_mean, hparams, emission_mean_fn, emission_cov_fn, method, **kwargs):
    """
    _ is a dummy parameter for compatibility with lofi 
    """
    params = base.RebayesParams(
        initial_mean=init_mean,
        emission_mean_function=emission_mean_fn,
        emission_cov_function=emission_cov_fn,
        **hparams,
    )

    if "ekf" in method:
        estimator = ekf.RebayesEKF(params, method=method)
    elif "lofi" in method:
        estimator = lofi.RebayesLoFi(params, method=method, **kwargs)
    return estimator