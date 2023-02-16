import jax
import numpy as np
import jax.numpy as jnp
from rebayes import base
from functools import partial
from jax.flatten_util import ravel_pytree
from bayes_opt import BayesianOptimization
from rebayes.extended_kalman_filter import ekf


def bbf(
    log_init_cov,
    dynamics_weights,
    log_emission_cov,
    # Specify before running
    train,
    test,
    flat_params,
    callback,
    apply_fn,
    method="fdekf",
    emission_mean_function=None,
    emission_cov_function=None,
    adaptive_variance=False
):
    """
    Black-box function for Bayesian optimization.
    """
    X_train, y_train = train
    X_test, y_test = test

    dynamics_covariance = None
    initial_covariance = jnp.exp(log_init_cov).item()
    if emission_mean_function is None:
        emission_mean_function = apply_fn
    if emission_cov_function is None:
        emission_cov_function = lambda w, x: jnp.exp(log_emission_cov)

    test_callback_kwargs = {"X_test": X_test, "y_test": y_test, "apply_fn": apply_fn}
    params_rebayes = base.RebayesParams(
        initial_mean=flat_params,
        initial_covariance=initial_covariance,
        dynamics_weights=dynamics_weights,
        dynamics_covariance=dynamics_covariance,
        emission_mean_function=emission_mean_function,
        emission_cov_function=emission_cov_function,
    )

    estimator = ekf.RebayesEKF(params_rebayes, method=method, adaptive_variance=adaptive_variance)

    bel, _ = estimator.scan(X_train, y_train, progress_bar=False)
    metric = callback(bel, **test_callback_kwargs)["test"].item()
    return -metric


def apply(flat_params, x, model, unflatten_fn):
    return model.apply(unflatten_fn(flat_params), x)


def create_optimizer(
    model,
    bounds,
    random_state,
    train,
    test,
    callback=None,
    method="fdekf",
    emission_mean_function=None,
    emission_cov_function=None,
    adaptive_variance=False,
):
    key = jax.random.PRNGKey(random_state)
    X_train, _ = train
    _, *n_features = X_train.shape

    batch_init = jnp.ones((1, *n_features))
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
        method=method,
        emission_mean_function=emission_mean_function,
        emission_cov_function=emission_cov_function,
        adaptive_variance=adaptive_variance,
    )
    if emission_cov_function is not None or adaptive_variance == True:
        bbf_partial = partial(
            bbf_partial,
            log_emission_cov=0.0,
        )

    optimizer = BayesianOptimization(
        f=bbf_partial,
        pbounds=bounds,
        random_state=random_state,
    )

    return optimizer, apply_fn, n_features


def get_best_params(num_params, optimizer, method="fdekf"):
    max_params = optimizer.max["params"].copy()

    dynamics_covariance = None
    initial_covariance = np.exp(max_params["log_init_cov"])
    dynamics_weights = max_params["dynamics_weights"]
    emission_cov = np.exp(max_params.get("log_emission_cov", 0.0))

    hparams = {
        "initial_covariance": initial_covariance,
        "dynamics_covariance": dynamics_covariance,
        "dynamics_weights": dynamics_weights,
        "emission_cov_function": lambda w, x: emission_cov,
    }

    return hparams

def build_estimator(init_mean, hparams, _, apply_fn, method="fdekf",
                    emission_mean_function=None, emission_cov_function=None,
                    adaptive_variance=False):
    """
    _ is a dummy parameter for compatibility with lofi 
    """
    if emission_mean_function is None:
        emission_mean_function = apply_fn
    if emission_cov_function is None or adaptive_variance == True:
        params = base.RebayesParams(
            initial_mean=init_mean,
            emission_mean_function=emission_mean_function,
            **hparams,
        )
    else:
        params = base.RebayesParams(
            initial_mean=init_mean,
            emission_mean_function=emission_mean_function,
            emission_cov_function=emission_cov_function,
            **hparams,
        )

    estimator = ekf.RebayesEKF(params, method=method, adaptive_variance=adaptive_variance)
    return estimator
