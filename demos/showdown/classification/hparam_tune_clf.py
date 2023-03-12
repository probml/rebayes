from functools import partial

import jax
import numpy as np
import jax.numpy as jnp
from rebayes import base
from jax.flatten_util import ravel_pytree
from jax import vmap, jit
from bayes_opt import BayesianOptimization
import optax

from rebayes.extended_kalman_filter import ekf
from rebayes.low_rank_filter import lofi
from rebayes.sgd_filter import replay_sgd as rsgd


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
    lofi_params,
    method="lofi",
    callback_at_end=True,
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

    estimator = lofi.RebayesLoFi(params_rebayes, lofi_params, method=method)

    if callback_at_end:
        bel, _ = estimator.scan(X_train, y_train, progress_bar=False)
        metric = callback(bel, **test_callback_kwargs)
    else:
        _, metric = estimator.scan(X_train, y_train, progress_bar=False, callback=callback, **test_callback_kwargs)
        metric = metric.mean()
        
    if jnp.isnan(metric) or jnp.isinf(metric):
        metric = -1e8
        
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
    callback_at_end=True,
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

    if callback_at_end:
        bel, _ = estimator.scan(X_train, y_train, progress_bar=False)
        metric = callback(bel, **test_callback_kwargs)
    else:
        _, metric = estimator.scan(X_train, y_train, progress_bar=False, callback=callback, **test_callback_kwargs)
        metric = metric.mean()
        
    if jnp.isnan(metric) or jnp.isinf(metric):
        metric = -1e8

    return metric


def bbf_rsgd(
    learning_rate,
    # Specify before running
    train,
    test,
    flat_params,
    callback,
    apply_fn,
    loss_fn,
    buffer_size,
    dim_output,
    callback_at_end=True,
):
    """
    Black-box function for Bayesian optimization.
    """
    X_train, y_train = train
    X_test, y_test = test

    tx = optax.sgd(learning_rate=learning_rate)

    test_callback_kwargs = {"X_test": X_test, "y_test": y_test, "apply_fn": apply_fn}
    
    @partial(jit, static_argnames=("applyfn",))
    def lossfn_fifo(params, counter, X, y, applyfn):
        logits = vmap(applyfn, (None, 0))(params, X).ravel()
        nll = loss_fn(logits=logits, labels=y.ravel())
        nll = nll.sum()
        loss = (nll * counter).sum() / counter.sum()
        return loss
    
    estimator = rsgd.FifoSGD(
        lossfn_fifo,
        apply_fn=apply_fn,
        init_params=flat_params,
        tx=tx,
        buffer_size=buffer_size,
        dim_features=[1, 28, 28, 1],
        dim_output=dim_output,
        n_inner=1,
    )
    
    if callback_at_end:
        bel, _ = estimator.scan(X_train, y_train, progress_bar=False)
        metric = callback(bel, **test_callback_kwargs)
    else:
        _, metric = estimator.scan(X_train, y_train, progress_bar=False, callback=callback, **test_callback_kwargs)
        metric = metric.mean()
        
    if jnp.isnan(metric) or jnp.isinf(metric):
        metric = -1e6
        
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
    verbose=2,
    callback_at_end=True,
    **kwargs
):
    key = jax.random.PRNGKey(random_state)
    X_train, _ = train
    _, *n_features = X_train.shape

    batch_init = jnp.ones((1, *n_features))
    params_init = model.init(key, batch_init)
    flat_params, recfn = ravel_pytree(params_init)

    apply_fn = partial(apply, model=model, unflatten_fn=recfn)
    
    if "sgd" not in method:
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
            callback_at_end=callback_at_end,
            **kwargs # Must include lofi_params if method is lofi
        )
    else:
        bbf_partial = partial(
            bbf_rsgd,
            train=train,
            test=test,
            flat_params=flat_params,
            callback=callback,
            apply_fn=apply_fn,
            callback_at_end=callback_at_end,
            **kwargs # Must include loss_fn, buffer_size, dim_output if method is rsgd
        )

    optimizer = BayesianOptimization(
        f=bbf_partial,
        pbounds=bounds,
        random_state=random_state,
        verbose=verbose,
        allow_duplicate_points=True,
    )

    return optimizer, apply_fn, n_features


def get_best_params(optimizer, method):
    max_params = optimizer.max["params"].copy()

    if "sgd" not in method:
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
    else:
        hparams = {
            "learning_rate": max_params["learning_rate"],
        }

    return hparams


def build_estimator(init_mean, apply_fn, hparams, emission_mean_fn, emission_cov_fn, method, **kwargs):
    """
    _ is a dummy parameter for compatibility with lofi 
    """
    if "sgd" not in method:
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
        bel = None
    else:
        tx = optax.sgd(learning_rate=hparams["learning_rate"])
        
        @partial(jit, static_argnames=("applyfn",))
        def lossfn_fifo(params, counter, X, y, applyfn):
            logits = vmap(applyfn, (None, 0))(params, X)
            nll = kwargs["loss_fn"](logits=logits, labels=y)
            nll = nll.sum(axis=-1)
            loss = (nll * counter).sum() / counter.sum()
            return loss
        
        estimator = rsgd.FifoSGD(
            lossfn_fifo,
            apply_fn=apply_fn,
            init_params=init_mean,
            tx=tx,
            buffer_size=kwargs["buffer_size"],
            dim_features=[1, 28, 28, 1],
            dim_output=kwargs["dim_output"],
            n_inner=1,
        )
        
    return estimator