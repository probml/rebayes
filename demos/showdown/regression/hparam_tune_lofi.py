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
    log_dynamics_weights,
    dynamics_log_cov,
    log_emission_cov,
    log_inflation,
    # Specify before running
    train,
    test,
    flat_params,
    callback,
    apply_fn,
    params_lofi,
    method="full_svd_lofi", # TODO: Deprecate this
    emission_mean_function=None,
    emission_cov_function=None,
):
    """
    Black-box function for Bayesian optimization.
    """
    X_train, y_train = train
    X_test, y_test = test

    initial_covariance = jnp.exp(log_init_cov).item()
    inflation = jnp.exp(log_inflation)
    dynamics_weights = jnp.exp(log_dynamics_weights)
    dynamics_covariance = jnp.exp(dynamics_log_cov)
    if emission_mean_function is None:
        emission_mean_function = apply_fn
    if emission_cov_function is None:
        def emission_cov_function(w, x): return jnp.exp(log_emission_cov)
    

    test_callback_kwargs = {"X_test": X_test, "y_test": y_test, "apply_fn": apply_fn}
    params_rebayes = base.RebayesParams(
        initial_mean=flat_params,
        initial_covariance=initial_covariance,
        dynamics_weights=dynamics_weights,
        dynamics_covariance=dynamics_covariance,
        emission_mean_function=emission_mean_function,
        emission_cov_function=emission_cov_function,
        dynamics_covariance_inflation_factor=inflation,
    )

    estimator = lofi.RebayesLoFiDiagonal(params_rebayes, params_lofi)

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
    method="full_svd_lofi",
    emission_mean_function=None,
    emission_cov_function=None,
):
    bounds = bounds.copy()
    key = jax.random.PRNGKey(random_state)
    X_train, _ = train
    _, *n_params = X_train.shape

    batch_init = jnp.ones((1, *n_params))
    params_init = model.init(key, batch_init)
    flat_params, recfn = ravel_pytree(params_init)

    kwargs = {}
    if bounds.get("log_inflation") is None:
        bounds.pop("log_inflation", None)
        kwargs["log_inflation"] = -np.inf


    apply_fn = partial(apply, model=model, unflatten_fn=recfn)
    bbf_partial = partial(
        bbf,
        train=train,
        test=test,
        flat_params=flat_params,
        callback=callback,
        apply_fn=apply_fn,
        params_lofi=params_lofi,
        method=method,
        emission_mean_function=emission_mean_function,
        emission_cov_function=emission_cov_function,
        **kwargs,
    )
    
    # Fix log-emission-covariance to dummy if adaptive
    if emission_cov_function is not None:
        bbf_partial = partial(
            bbf_partial,
            log_emission_cov=0.0,
        )



    optimizer = BayesianOptimization(
        f=bbf_partial,
        pbounds=bounds,
        random_state=random_state,
        allow_duplicate_points=True,
    )

    return optimizer, apply_fn, n_params


def get_best_params(n_params, optimizer):
    if type(optimizer) is dict:
        max_params = optimizer.copy()
    else:
        max_params = optimizer.max["params"].copy()

    init_cov = np.exp(max_params["log_init_cov"]).item()
    emission_cov = np.exp(max_params.get("log_emission_cov", 0.0))
    dynamics_weights = np.exp(max_params["log_dynamics_weights"])
    dynamics_cov = np.exp(max_params.get("dynamics_log_cov"))
    inflation = np.exp(max_params.get("log_inflation", -np.inf))

    def emission_cov_function(w, x): return emission_cov
    hparams = {
        "initial_covariance": init_cov,
        "dynamics_covariance": dynamics_cov,
        "dynamics_weights": dynamics_weights,
        "emission_cov_function": emission_cov_function,
        "dynamics_covariance_inflation_factor": inflation,
    }

    return hparams


def build_estimator(init_mean, hparams, params_lofi, apply_fn, method="full_svd_lofi",
                    emission_mean_function=None, emission_cov_function=None):
    if emission_mean_function is None:
        emission_mean_function = apply_fn
    if emission_cov_function is None:
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

    estimator = lofi.RebayesLoFiDiagonal(params, params_lofi)
    return estimator
